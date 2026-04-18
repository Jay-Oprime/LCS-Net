import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== 配置类 ==================
class ModelConfig:
    """模型配置类，用于控制各个模块的开关"""
    def __init__(self, 
                 use_cbam=False,
                 use_res=True,
                 use_bilinear=False,
                 use_ds=False,
                 use_ghost=False,
                 reduce_channels=False):
        self.USE_CBAM = use_cbam
        self.USE_RES = use_res
        self.USE_BILINEAR = use_bilinear
        self.USE_DS = use_ds  # 深度可分离卷积开关
        self.USE_GHOST = use_ghost  # Ghost模块开关
        self.REDUCE_CHANNELS = reduce_channels  # 通道数缩减开关

# ================== 全局配置（默认） ==================
# 默认配置，实际使用时通过config参数传递
default_config = ModelConfig()

# ================== 模型复杂度计算函数 ==================
def calculate_model_complexity(model, input_size=(1, 3, 256, 256), device='cpu'):
    """
    统一计算模型参数量和FLOPs的函数

    参数:
        model: PyTorch模型
        input_size: 输入张量大小，默认(1, 3, 256, 256)
        device: 计算设备

    返回:
        params_m: 参数量（百万）
        flops_g: FLOPs（十亿）
        total_params: 总参数量（原始值）
    """
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6

    # 计算FLOPs
    flops_g = 0.0
    try:
        from thop import profile
        # 使用深拷贝避免修改原模型
        from copy import deepcopy
        model_copy = deepcopy(model).to(device)
        dummy_input = torch.randn(input_size).to(device)

        # 计算FLOPs
        flops, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9

        # 清理
        del model_copy, dummy_input
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except ImportError:
        print("⚠️ thop库未安装，无法计算FLOPs。请运行: pip install thop")
        flops_g = 0.0
    except Exception as e:
        print(f"⚠️ FLOPs计算失败: {e}")
        flops_g = 0.0

    return params_m, flops_g, total_params

# 新增CBAM注意力模块
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
           nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
       # Spatial Attention
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        channel_out = avg_out + max_out
        channel_out = self.sigmoid(channel_out)
        x = x * channel_out

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv1(spatial_out)
        spatial_out = self.bn(spatial_out)
        spatial_out = self.sigmoid(spatial_out)
        x = x * spatial_out

        return x

# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.bn_pw = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        return x

# Ghost模块 - 使用廉价操作生成更多特征图
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, bias=False):
        super().__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=bias),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, 
                     groups=init_channels, bias=bias),
            nn.BatchNorm2d(new_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

# 优化后的Conv_Block
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, config=None):
        super().__init__()

        # 使用传入的配置或默认配置
        if config is None:
            config = default_config

        # 根据开关选择卷积类型
        if config.USE_DS and config.USE_GHOST:
            # 结合深度可分离卷积和Ghost模块
            self.conv1 = GhostModule(in_channel, out_channel // 2)
            self.conv2 = DepthwiseSeparableConv(out_channel // 2, out_channel)
        elif config.USE_DS:
            # 仅使用深度可分离卷积
            self.conv1 = DepthwiseSeparableConv(in_channel, out_channel)
            self.conv2 = DepthwiseSeparableConv(out_channel, out_channel)
        elif config.USE_GHOST:
            # 仅使用Ghost模块
            self.conv1 = GhostModule(in_channel, out_channel)
            self.conv2 = GhostModule(out_channel, out_channel)
        else:
            # 原始标准卷积
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False)
            self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False)

        # 依据开关插入 CBAM 或 Identity
        cbam = CBAMLayer(out_channel) if config.USE_CBAM else nn.Identity()

        # 构建层序列
        layers = [
            self.conv1,
            nn.LeakyReLU(inplace=True)
        ]

        if not (config.USE_DS or config.USE_GHOST):
            layers.insert(1, nn.BatchNorm2d(out_channel))

        layers.extend([
            self.conv2,
            nn.BatchNorm2d(out_channel) if not (config.USE_DS or config.USE_GHOST) else nn.Identity(),
            cbam,  # 开关控制
            nn.LeakyReLU(inplace=True)
        ])

        self.layer = nn.Sequential(*layers)

        # 残差连接
        if config.USE_RES:
            if in_channel != out_channel:
                if config.USE_DS:
                    self.residual = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, 1, bias=False),
                        nn.BatchNorm2d(out_channel)
                    )
                else:
                    self.residual = nn.Conv2d(in_channel, out_channel, 1)
            else:
                self.residual = nn.Identity()
        else:
            self.residual = None

    def forward(self, x):
        out = self.layer(x)
        if self.residual is not None:
            out = out + self.residual(x)
        return out

# 通道数缩减函数
def get_optimized_channels(base_channels, config=None):
    """根据开关缩减通道数"""
    if config is None:
        config = default_config

    if config.REDUCE_CHANNELS:
        # 缩减高分辨率特征图的通道数
        reduction_factors = {
            64: 48,    # 64 -> 48
            128: 96,   # 128 -> 96  
            256: 192,  # 256 -> 192
            512: 384,  # 512 -> 384
            1024: 768  # 1024 -> 768
        }
        return reduction_factors.get(base_channels, base_channels)
    return base_channels

class DownSample(nn.Module):
    def __init__(self, channel, config=None):
        super().__init__()
        if config is None:
            config = default_config

        if config.USE_DS:
            self.layer = nn.Sequential(
                nn.MaxPool2d(2),
                DepthwiseSeparableConv(channel, channel)
            )
        else:
            self.layer = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(channel, channel, 3, 1, 1, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(channel),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self, channel, config=None):
        super().__init__()
        if config is None:
            config = default_config

        out_channel = channel // 2

        # 开关控制上采样方式
        if config.USE_BILINEAR:
            if config.USE_DS:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DepthwiseSeparableConv(channel, out_channel)
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(channel, out_channel, 3, 1, 1, padding_mode='reflect')
                )
        else:
            self.up = nn.ConvTranspose2d(channel, out_channel, 2, 2)

    def forward(self, x, skip):
        x = self.up(x)
        return torch.cat([x, skip], dim=1)

# 优化后的UNet
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, config=None):
        super().__init__()

        # 使用传入的配置或默认配置
        self.config = config if config is not None else default_config

        # 获取优化后的通道数
        c64 = get_optimized_channels(64, self.config)
        c128 = get_optimized_channels(128, self.config)
        c256 = get_optimized_channels(256, self.config)
        c512 = get_optimized_channels(512, self.config)
        c1024 = get_optimized_channels(1024, self.config)

        # 编码器
        self.c1 = Conv_Block(n_channels, c64, self.config)
        self.d1 = DownSample(c64, self.config)
        self.c2 = Conv_Block(c64, c128, self.config)
        self.d2 = DownSample(c128, self.config)
        self.c3 = Conv_Block(c128, c256, self.config)
        self.d3 = DownSample(c256, self.config)
        self.c4 = Conv_Block(c256, c512, self.config)
        self.d4 = DownSample(c512, self.config)
        self.c5 = Conv_Block(c512, c1024, self.config)

        # 解码器
        self.u1 = UpSample(c1024, self.config)
        self.c6 = Conv_Block(c1024, c512, self.config)
        self.u2 = UpSample(c512, self.config)
        self.c7 = Conv_Block(c512, c256, self.config)
        self.u3 = UpSample(c256, self.config)
        self.c8 = Conv_Block(c256, c128, self.config)
        self.u4 = UpSample(c128, self.config)
        self.c9 = Conv_Block(c128, c64, self.config)

        self.out = nn.Sequential(
            nn.Conv2d(c64, n_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码器路径
        s1 = self.c1(x)
        s2 = self.c2(self.d1(s1))
        s3 = self.c3(self.d2(s2))
        s4 = self.c4(self.d3(s3))
        bottleneck = self.c5(self.d4(s4))

        # 解码器路径
        d1 = self.c6(self.u1(bottleneck, s4))
        d2 = self.c7(self.u2(d1, s3))
        d3 = self.c8(self.u3(d2, s2))
        d4 = self.c9(self.u4(d3, s1))

        return self.out(d4)


if __name__ == '__main__':
    # 测试不同配置的模型
    configs = [
        ("原始模型", ModelConfig(use_cbam=False, use_res=True, use_bilinear=False, use_ds=False, use_ghost=False, reduce_channels=False)),
        ("深度可分离卷积", ModelConfig(use_cbam=False, use_res=True, use_bilinear=False, use_ds=True, use_ghost=False, reduce_channels=False)),
        ("Ghost模块", ModelConfig(use_cbam=False, use_res=True, use_bilinear=False, use_ds=False, use_ghost=True, reduce_channels=False)),
        ("通道数缩减", ModelConfig(use_cbam=False, use_res=True, use_bilinear=False, use_ds=False, use_ghost=False, reduce_channels=True)),
        ("全部优化", ModelConfig(use_cbam=True, use_res=True, use_bilinear=True, use_ds=True, use_ghost=True, reduce_channels=True))
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print()

    for name, config in configs:
        print(f"\n{name}配置:")
        print(f"  CBAM: {config.USE_CBAM}")
        print(f"  残差连接: {config.USE_RES}")
        print(f"  双线性插值: {config.USE_BILINEAR}")
        print(f"  深度可分离卷积: {config.USE_DS}")
        print(f"  Ghost模块: {config.USE_GHOST}")
        print(f"  通道数缩减: {config.REDUCE_CHANNELS}")

        # 创建模型
        net = UNet(config=config).to(device)
        x = torch.randn(1, 3, 256, 256).to(device)

        # 使用统一的复杂度计算函数
        params_m, flops_g, total_params = calculate_model_complexity(net, input_size=(1, 3, 256, 256), device=device)

        print(f"  参数量: {total_params:,} ({params_m:.2f}M)")
        print(f"  FLOPs: {flops_g:.2f}G")

        # 测试前向传播
        with torch.no_grad():
            output = net(x)
            print(f"  输出形状: {output.shape}")

        print("-" * 50)
