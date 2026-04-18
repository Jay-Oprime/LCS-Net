#!/usr/bin/env python3
"""
优化后的U-Net 图像分割预测程序
用于加载优化后的UNet模型并进行图像分割预测
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import yaml

warnings.filterwarnings('ignore')

# 添加thop用于计算FLOPs
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️ thop库未安装，无法计算FLOPs。请运行: pip install thop")

# -------------- 模型定义（与net_optimized.py保持一致） -------------
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        channel_out = avg_out + max_out
        channel_out = self.sigmoid(channel_out)
        x = x * channel_out

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv1(spatial_out)
        spatial_out = self.bn(spatial_out)
        spatial_out = self.sigmoid(spatial_out)
        x = x * spatial_out
        return x

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
        x = nn.functional.leaky_relu(x, inplace=True)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        return x

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

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, config=None):
        super().__init__()

        # 根据开关选择卷积类型
        if config and config['use_ds'] and config['use_ghost']:
            self.conv1 = GhostModule(in_channel, out_channel // 2)
            self.conv2 = DepthwiseSeparableConv(out_channel // 2, out_channel)
        elif config and config['use_ds']:
            self.conv1 = DepthwiseSeparableConv(in_channel, out_channel)
            self.conv2 = DepthwiseSeparableConv(out_channel, out_channel)
        elif config and config['use_ghost']:
            self.conv1 = GhostModule(in_channel, out_channel)
            self.conv2 = GhostModule(out_channel, out_channel)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False)
            self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False)

        # 依据开关插入 CBAM 或 Identity
        cbam = CBAMLayer(out_channel) if config and config['use_cbam'] else nn.Identity()

        # 构建层序列
        layers = [
            self.conv1,
            nn.LeakyReLU(inplace=True)
        ]

        if not (config and (config['use_ds'] or config['use_ghost'])):
            layers.insert(1, nn.BatchNorm2d(out_channel))

        layers.extend([
            self.conv2,
            nn.BatchNorm2d(out_channel) if not (config and (config['use_ds'] or config['use_ghost'])) else nn.Identity(),
            cbam,
            nn.LeakyReLU(inplace=True)
        ])

        self.layer = nn.Sequential(*layers)

        # 残差连接
        if config and config['use_res']:
            if in_channel != out_channel:
                if config['use_ds']:
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

class DownSample(nn.Module):
    def __init__(self, channel, config=None):
        super().__init__()
        if config and config['use_ds']:
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
        out_channel = channel // 2

        if config and config['use_bilinear']:
            if config and config['use_ds']:
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

def get_optimized_channels(base_channels, config=None):
    """缩减通道数"""
    if config and config['reduce_channels']:
        reduction_factors = {
            64: 48,
            128: 96,
            256: 192,
            512: 384,
            1024: 768
        }
        return reduction_factors.get(base_channels, base_channels)
    return base_channels

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, config=None):
        super().__init__()

        # 获取优化后的通道数
        c64 = get_optimized_channels(64, config)
        c128 = get_optimized_channels(128, config)
        c256 = get_optimized_channels(256, config)
        c512 = get_optimized_channels(512, config)
        c1024 = get_optimized_channels(1024, config)

        # 编码器
        self.c1 = Conv_Block(n_channels, c64, config)
        self.d1 = DownSample(c64, config)
        self.c2 = Conv_Block(c64, c128, config)
        self.d2 = DownSample(c128, config)
        self.c3 = Conv_Block(c128, c256, config)
        self.d3 = DownSample(c256, config)
        self.c4 = Conv_Block(c256, c512, config)
        self.d4 = DownSample(c512, config)
        self.c5 = Conv_Block(c512, c1024, config)

        # 解码器
        self.u1 = UpSample(c1024, config)
        self.c6 = Conv_Block(c1024, c512, config)
        self.u2 = UpSample(c512, config)
        self.c7 = Conv_Block(c512, c256, config)
        self.u3 = UpSample(c256, config)
        self.c8 = Conv_Block(c256, c128, config)
        self.u4 = UpSample(c128, config)
        self.c9 = Conv_Block(c128, c64, config)

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

# --------------------------------------------------------

# 统一计算模型复杂度的函数
def calculate_model_complexity(model, input_size=(1, 3, 512, 512), device='cpu'):
    """
    统一计算模型参数量和FLOPs的函数

    参数:
        model: PyTorch模型
        input_size: 输入张量大小
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
    if THOP_AVAILABLE:
        try:
            from copy import deepcopy
            model_copy = deepcopy(model).to(device)
            dummy_input = torch.randn(input_size).to(device)

            # 计算FLOPs
            flops, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9

            # 清理
            del model_copy, dummy_input
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"⚠️ FLOPs计算失败: {e}")
            flops_g = 0.0
    else:
        print("⚠️ thop库未安装，无法计算FLOPs")
        flops_g = 0.0

    return params_m, flops_g, total_params

class UNetPredictor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu', config=None):
        self.device = device
        self.model_path = model_path
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.load_model()

    def load_model(self):
        try:
            self.model = UNet(n_channels=3, n_classes=1, config=self.config).to(self.device)
            ckpt = torch.load(self.model_path, map_location=self.device)

            # 统一取 state_dict
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                best_oa = ckpt.get('best_oa', ckpt.get('best_dice', 0.0))
                print(f'✓ 验证集 OA: {best_oa:.4f}')
            else:
                state_dict = ckpt

            # 去掉 torch.compile 可能带来的前缀
            state_dict = {k[10:] if k.startswith('_orig_mod.') else k: v
                          for k, v in state_dict.items()}

            # 严格加载
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

            # 使用统一函数计算参数量和FLOPs
            params_m, flops_g, total_params = calculate_model_complexity(self.model, input_size=(1, 3, 512, 512), device=self.device)
            print(f'✓ 模型加载成功')
            print(f'✓ 参数量: {total_params:,} ({params_m:.2f}M)')
            print(f'✓ FLOPs: {flops_g:.2f}G')
            print(f'✓ 使用设备: {self.device}')

            # 打印配置
            if self.config:
                print("🔧 模型配置:")
                print(f"  CBAM注意力: {self.config.get('use_cbam', False)}")
                print(f"  残差连接: {self.config.get('use_res', False)}")
                print(f"  双线性插值: {self.config.get('use_bilinear', False)}")
                print(f"  深度可分离卷积: {self.config.get('use_ds', False)}")
                print(f"  Ghost模块: {self.config.get('use_ghost', False)}")
                print(f"  通道数缩减: {self.config.get('reduce_channels', False)}")
        except Exception as e:
            print(f'✗ 模型加载失败: {e}')
            raise

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            tensor = self.transform(image).unsqueeze(0)
            return tensor, original_size, image
        except Exception as e:
            print(f'✗ 图像预处理失败 {image_path}: {e}')
            raise

    def predict(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            pred = (output > 0.5).float()
            return output.cpu().numpy(), pred.cpu().numpy()

    def postprocess_mask(self, mask, original_size):
        mask_img = Image.fromarray((mask[0, 0] * 255).astype(np.uint8))
        mask_img = mask_img.resize(original_size, Image.NEAREST)
        return np.array(mask_img)

    def load_ground_truth(self, mask_path):
        try:
            mask = Image.open(mask_path).convert('L')
            return (np.array(mask) > 0).astype(np.uint8)
        except Exception as e:
            print(f'✗ 加载真实标注失败 {mask_path}: {e}')
            return None

    def calculate_iou(self, pred_mask, true_mask):
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        return 1.0 if union == 0 else intersection / union

    def calculate_dice(self, pred_mask, true_mask):
        intersection = np.logical_and(pred_mask, true_mask).sum()
        total = pred_mask.sum() + true_mask.sum()
        return 1.0 if total == 0 else (2 * intersection) / total

    def calculate_pixel_accuracy(self, pred_mask, true_mask):
        return np.equal(pred_mask, true_mask).mean()

    def calculate_precision_recall(self, pred_mask, true_mask):
        tp = np.logical_and(pred_mask, true_mask).sum()
        fp = np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), true_mask).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall

    def calculate_f1_score(self, precision, recall):
        return 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    def visualize_results(self, image, pred_mask, true_mask, metrics, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('U-Net Prediction Result', fontsize=16, fontweight='bold')

        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image'), axes[0, 0].axis('off')

        axes[0, 1].imshow(true_mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth'), axes[0, 1].axis('off')

        axes[0, 2].imshow(pred_mask, cmap='gray')
        axes[0, 2].set_title('Prediction'), axes[0, 2].axis('off')

        overlay = image.copy()
        overlay[pred_mask > 0] = [255, 0, 0]
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Prediction Overlay'), axes[1, 0].axis('off')

        diff = np.zeros_like(image)
        tp = np.logical_and(pred_mask, true_mask)
        fp = np.logical_and(pred_mask, np.logical_not(true_mask))
        fn = np.logical_and(np.logical_not(pred_mask), true_mask)
        diff[tp] = [255, 255, 255]
        diff[fp] = [255, 0, 0]
        diff[fn] = [0, 0, 255]
        axes[1, 1].imshow(diff)
        axes[1, 1].set_title('Error Analysis (White:Correct, Red:FP, Blue:FN)'), axes[1, 1].axis('off')

        axes[1, 2].axis('off')
        metrics_text = f"""
Evaluation Metrics:
IOU: {metrics['iou']:.4f}
Dice: {metrics['dice']:.4f}
Pixel Accuracy: {metrics['pixel_accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1 Score: {metrics['f1_score']:.4f}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12,
                       verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def process_single_image(self, image_path, mask_path, output_dir):
        try:
            print(f"\n处理图像: {os.path.basename(image_path)}")
            image_tensor, original_size, original_image = self.preprocess_image(image_path)
            probs, binary_mask = self.predict(image_tensor)
            pred_mask = (self.postprocess_mask(binary_mask, original_size) > 127).astype(np.uint8)
            true_mask = self.load_ground_truth(mask_path)

            metrics = {}
            if true_mask is not None:
                metrics['iou'] = self.calculate_iou(pred_mask, true_mask)
                metrics['dice'] = self.calculate_dice(pred_mask, true_mask)
                metrics['pixel_accuracy'] = self.calculate_pixel_accuracy(pred_mask, true_mask)
                metrics['precision'], metrics['recall'] = self.calculate_precision_recall(pred_mask, true_mask)
                metrics['f1_score'] = self.calculate_f1_score(metrics['precision'], metrics['recall'])
                print(f"  IOU: {metrics['iou']:.4f}  Dice: {metrics['dice']:.4f}  F1: {metrics['f1_score']:.4f}")
            else:
                metrics = {k: 0.0 for k in ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall', 'f1_score']}

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"{base_name}_pred_mask.png"))
            Image.fromarray((self.postprocess_mask(probs, original_size) * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"{base_name}_prob_map.png"))
            self.visualize_results(np.array(original_image), pred_mask, true_mask, metrics,
                                   os.path.join(output_dir, f"{base_name}_visualization.png"))
            return metrics
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            return None

    def process_directory(self, images_dir, masks_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f"*{ext}"))
            image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
        if not image_files:
            print(f"✗ 在 {images_dir} 中未找到图像文件")
            return None
        print(f"找到 {len(image_files)} 个图像文件")
        all_metrics = []
        for image_path in image_files:
            base_name = image_path.stem
            mask_path = os.path.join(masks_dir, f"{base_name}.png")
            if not os.path.exists(mask_path):
                for ext in image_extensions:
                    alt_path = os.path.join(masks_dir, f"{base_name}{ext}")
                    if os.path.exists(alt_path):
                        mask_path = alt_path
                        break
            metrics = self.process_single_image(str(image_path), mask_path, output_dir)
            if metrics:
                all_metrics.append(metrics)

        if all_metrics:
            avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            print(f"\n=== 整体评估结果 ===")
            print(f"处理图像数量: {len(all_metrics)}")
            print(f"平均 IOU: {avg_metrics['iou']:.4f}")
            print(f"平均 Dice: {avg_metrics['dice']:.4f}")
            print(f"平均像素准确率: {avg_metrics['pixel_accuracy']:.4f}")
            print(f"平均 F1分数: {avg_metrics['f1_score']:.4f}")

            summary_path = os.path.join(output_dir, "evaluation_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_images': len(all_metrics),
                    'average_metrics': avg_metrics,
                    'individual_metrics': all_metrics,
                    'model_info': {
                        'type': 'Optimized U-Net',
                        'features': [
                            f"CBAM Attention: {self.config.get('use_cbam', False) if self.config else False}",
                            f"Depthwise Separable Conv: {self.config.get('use_ds', False) if self.config else False}",
                            f"Ghost Modules: {self.config.get('use_ghost', False) if self.config else False}",
                            f"Reduced Channels: {self.config.get('reduce_channels', False) if self.config else False}",
                            f"Residual Connections: {self.config.get('use_res', False) if self.config else False}",
                            f"Bilinear Upsampling: {self.config.get('use_bilinear', False) if self.config else False}"
                        ]
                    },
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            print(f"\n评估摘要已保存到: {summary_path}")
            return avg_metrics
        return None


def load_config_from_checkpoint(checkpoint_path):
    """从检查点加载配置"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            return checkpoint['config']
        else:
            print("⚠️ 检查点中没有配置信息，使用默认配置")
            return None
    except Exception as e:
        print(f"⚠️ 无法从检查点加载配置: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='优化后的U-Net 图像分割预测程序')
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的优化模型文件路径')
    parser.add_argument('--images', type=str, required=True,
                        help='输入图像目录路径，例如：./dataset/test/images')
    parser.add_argument('--masks', type=str, required=True,
                        help='真实标注目录路径，例如：./dataset/test/masks')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果目录路径')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备 (cuda 或 cpu)')
    parser.add_argument('--single-image', type=str, default=None,
                        help='处理单个图像文件（可选）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（YAML格式，包含model_config）')
    args = parser.parse_args()

    print("=" * 60)
    print("优化后的U-Net 图像分割预测程序")
    print("=" * 60)
    print(f"模型文件: {args.model}")

    # 设置输出目录
    if args.output is None:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"./output_{model_name}/"

    print(f"图像目录: {args.images}")
    print(f"掩码目录: {args.masks}")
    print(f"输出目录: {args.output}")
    print(f"计算设备: {args.device}")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"✗ 模型文件不存在: {args.model}")
        return
    if not os.path.exists(args.images):
        print(f"✗ 图像目录不存在: {args.images}")
        return
    if args.masks and not os.path.exists(args.masks):
        print(f"✗ 掩码目录不存在: {args.masks}")
        return

    try:
        # 加载配置
        config = None
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
                config = cfg.get('model_config', None)
                print("✓ 从配置文件加载模型配置")
        else:
            # 尝试从检查点加载配置
            config = load_config_from_checkpoint(args.model)
            if config:
                print("✓ 从检查点加载模型配置")

        if config:
            print("🔧 模型配置:")
            print(f"  CBAM注意力: {config.get('use_cbam', False)}")
            print(f"  残差连接: {config.get('use_res', False)}")
            print(f"  双线性插值: {config.get('use_bilinear', False)}")
            print(f"  深度可分离卷积: {config.get('use_ds', False)}")
            print(f"  Ghost模块: {config.get('use_ghost', False)}")
            print(f"  通道数缩减: {config.get('reduce_channels', False)}")
        else:
            print("⚠️ 未找到配置信息，使用默认配置")

        print("=" * 60)

        predictor = UNetPredictor(args.model, args.device, config)
        if args.single_image:
            image_name = os.path.basename(args.single_image)
            base_name = os.path.splitext(image_name)[0]
            mask_path = os.path.join(args.masks, f"{base_name}.png")
            predictor.process_single_image(args.single_image, mask_path, args.output)
        else:
            predictor.process_directory(args.images, args.masks, args.output)
        print("\n✓ 预测完成！")
        print(f"结果保存在: {args.output}")
    except Exception as e:
        print(f"\n✗ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
