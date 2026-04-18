import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from net_optimized import UNet, ModelConfig, calculate_model_complexity  # 保留原有导入
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import yaml
import argparse
import json
from datetime import datetime
import json
import numpy as np

# 新增：用于生成可序列化的JSON
class NpEncoder(json.JSONEncoder):
    """把 numpy 标量转成 Python 原生类型，供 json 序列化"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 新增：纯净的统计函数（与B代码完全一致）
def calculate_model_complexity_pure(model, input_size=(1, 3, 256, 256)):
    """纯粹的模型复杂度计算，禁用所有动态优化"""
    # 参数量计算
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6
    
    # FLOPs计算
    flops_g = 0.0
    try:
        from thop import profile
        # CPU上统计，确保环境纯净
        model_cpu = model.cpu()
        model_cpu.eval()
        dummy_input = torch.randn(input_size)
        flops, _ = profile(model_cpu, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
    except ImportError:
        print("⚠️ thop库未安装，无法计算FLOPs。请运行: pip install thop")
        flops_g = 0.0
    except Exception as e:
        print(f"⚠️ FLOPs计算失败: {e}")
        flops_g = 0.0
    
    return params_m, flops_g, total_params

# -------------------- 数据集类 --------------------
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform if mask_transform else transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.valid_pairs = []
        
        for img in self.images:
            mask = img.replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, mask)
            if os.path.exists(mask_path):
                try:
                    with Image.open(mask_path) as test_img:
                        test_img.verify()
                    self.valid_pairs.append((img, mask))
                except (IOError, SyntaxError) as e:
                    print(f"⚠️ 损坏掩码文件: {mask_path} - {e}")
            else:
                print(f"⚠️ 缺失掩码文件: {mask_path}")
        
        print(f"✅ 找到 {len(self.valid_pairs)} 个有效图像-掩码对")

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        mask = (mask > 0).float()
        
        return img, mask

    def __len__(self):
        return len(self.valid_pairs)

# -------------------- 指标计算 --------------------
def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def calculate_metrics(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()
    
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)
    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)
    
    oa = (tp + tn) / (tp + fp + fn + tn + smooth)
    iou = tp / (tp + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    return {
        'oa': oa.mean().item(),
        'iou': iou.mean().item(),
        'f1': f1.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }

# -------------------- 主训练函数 --------------------
def train_model(config, experiment_name, save_dir, resume_path=None,
                img_dir=None, mask_dir=None, val_img_dir=None, val_mask_dir=None):
    # 配置参数（完全不变）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_epochs = 300  # 注意：原代码是5，如需改为1000请自行调整
    lr = 1e-4
    img_size = 256
    gradient_clip = 1.0
    num_workers = 4
    patience = 50
    min_delta = 0.001
    
    print(f"🚀 使用设备: {device}")
    print(f"⚙️ 批大小: {batch_size}, 初始学习率: {lr}, 图像尺寸: {img_size}x{img_size}")
    print(f"🛑 早停参数: 耐心={patience} epochs, 最小改进={min_delta}")
    
    experiment_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    

    
    print(f"📂 数据目录: {img_dir} | {mask_dir}")
    print(f"💾 实验目录: {experiment_dir}")
    
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    train_dataset = SegDataset(img_dir, mask_dir, transform=img_transform, mask_transform=mask_transform)
    print(f"📊 训练集大小: {len(train_dataset)} 样本")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    

    val_dataset = SegDataset(val_img_dir, val_mask_dir, transform=img_transform, mask_transform=mask_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    
    # 初始化模型（完全不变）
    net = UNet(n_channels=3, n_classes=1, config=config).to(device)
    print(f"🧠 模型架构: {net.__class__.__name__}")
    
    # 打印配置（完全不变）
    print("🔧 模型配置:")
    print(f"  CBAM注意力: {config.USE_CBAM}")
    print(f"  残差连接: {config.USE_RES}")
    print(f"  双线性插值: {config.USE_BILINEAR}")
    print(f"  深度可分离卷积: {config.USE_DS}")
    print(f"  Ghost模块: {config.USE_GHOST}")
    print(f"  通道数缩减: {config.REDUCE_CHANNELS}")
    
    # ==================== 核心修复：纯净统计 ====================
    # 1. 创建用于统计的纯净模型副本（不影响训练用的net）
    import copy
    net_for_stat = copy.deepcopy(net)
    
    # 2. 移除所有动态优化包装
    if hasattr(net_for_stat, 'module'):  # 移除DataParallel
        net_for_stat = net_for_stat.module
    if hasattr(net_for_stat, '_orig_mod'):  # 移除torch.compile
        net_for_stat = net_for_stat._orig_mod
    
    # 3. 移动到CPU并设置为评估模式
    net_for_stat = net_for_stat.cpu().eval()
    
    # 4. 使用纯净函数计算（与B代码完全一致）
    params_m, flops_g, total_params = calculate_model_complexity_pure(net_for_stat)
    
    # 5. 打印统计结果（格式与B一致）
    print(f"参数数量: {total_params:,}")
    print(f"模型参数量: {params_m:.2f}M")
    print(f"模型FLOPs: {flops_g:.2f}G")  # 现在与B一致！
    # =========================================================
    
    # 训练时的并行和编译（完全不变）
    if torch.cuda.device_count() > 1:
        print(f"⚡ 使用 {torch.cuda.device_count()} GPU 进行数据并行")
        net = nn.DataParallel(net)
    
    try:
        net = torch.compile(net, mode="reduce-overhead")
        print("🔧 模型已编译优化")
    except Exception as e:
        print(f"⚠️ 模型编译失败: {e}, 继续未优化")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    print("🎛️ 启用混合精度训练")
    
    # 后续训练逻辑（完全不变）
    start_epoch = 0
    previous_best_dice = -np.inf
    
    # 新增：存储验证集最佳指标
    best_val_metrics = {
        'oa': -np.inf,
        'dice': 0.0,
        'iou': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }
    
    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                previous_best_dice = checkpoint.get('best_dice', -np.inf)
                start_epoch = checkpoint.get('epoch', 0)
                # 新增：加载验证集最佳指标
                best_val_metrics = checkpoint.get('best_val_metrics', {
                    'oa': checkpoint.get('best_oa', -np.inf),
                    'dice': 0.0,
                    'iou': 0.0,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                })
                print(f"✅ 加载检查点，最佳Dice: {previous_best_dice:.4f}, 起始epoch: {start_epoch}")
            else:
                state_dict = checkpoint
                
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            net.load_state_dict(state_dict, strict=False)
            print(f"✅ 成功加载检查点: {resume_path}")
        except Exception as e:
            print(f"⚠️ 加载检查点失败: {e}, 从头开始训练")
            previous_best_dice = -np.inf
            start_epoch = 0
    
    config_params = {
        'experiment_name': experiment_name,
        'arch': 'UNet_Optimized',
        'batch_size': batch_size,
        'dataset': 'dsb2018_96',
        'deep_supervision': False,
        'early_stopping': -1,
        'epochs': num_epochs,
        'factor': 0.1,
        'gamma': 0.6666666666666666,
        'gradient_ext': '.png',
        'img_ext': '.jpg',
        'input_channels': 3,
        'input_h': img_size,
        'input_w': img_size,
        'loss': 'BCEDiceLoss',
        'lr': lr,
        'mask_ext': '.png',
        'milestones': '50',
        'min_lr': 1.0e-05,
        'momentum': 0.9,
        'name': f'{experiment_name}_UNet_Optimized',
        'nesterov': False,
        'num_classes': 1,
        'num_workers': num_workers,
        'optimizer': 'Adam',
        'patience': 2,
        'scheduler': 'CosineAnnealingLR',
        'weight_decay': 0.001,
        'model_config': {
            'use_cbam': config.USE_CBAM,
            'use_res': config.USE_RES,
            'use_bilinear': config.USE_BILINEAR,
            'use_ds': config.USE_DS,
            'use_ghost': config.USE_GHOST,
            'reduce_channels': config.REDUCE_CHANNELS
        },
        'model_complexity': {
            'parameters': params_m,
            'flops': flops_g
        }
    }
    
    config_path = os.path.join(experiment_dir, "config.yml")
    with open(config_path, 'w') as f:
        yaml.dump(config_params, f, default_flow_style=False)
    print(f"✅ 配置文件已生成: {config_path}")
    
    # 训练循环（完全不变）
    dice_history = []
    loss_history = []
    oa_history = []
    iou_history = []
    f1_history = []
    precision_history = []
    recall_history = []
    
    best_oa = previous_best_dice
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss = 0.0
        running_dice = 0.0
        processed_samples = 0
        
        running_oa = 0.0
        running_iou = 0.0
        running_f1 = 0.0
        running_precision = 0.0
        running_recall = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for i, (images, masks) in enumerate(progress):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                outputs = net(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)
                metrics = calculate_metrics(outputs, masks)
                
                current_batch_size = images.size(0)
                running_loss += loss.item() * current_batch_size
                running_dice += dice * current_batch_size
                processed_samples += current_batch_size
                
                running_oa += metrics['oa'] * current_batch_size
                running_iou += metrics['iou'] * current_batch_size
                running_f1 += metrics['f1'] * current_batch_size
                running_precision += metrics['precision'] * current_batch_size
                running_recall += metrics['recall'] * current_batch_size
            
            avg_loss = running_loss / processed_samples
            avg_dice = running_dice / processed_samples
            progress.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "dice": f"{avg_dice:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        epoch_loss = running_loss / processed_samples
        epoch_dice = running_dice / processed_samples
        epoch_oa = running_oa / processed_samples
        epoch_iou = running_iou / processed_samples
        epoch_f1 = running_f1 / processed_samples
        epoch_precision = running_precision / processed_samples
        epoch_recall = running_recall / processed_samples
        
        # 修改：在验证阶段计算所有指标
        # 修改：在验证阶段计算所有指标
        net.eval()
        val_metrics_sum = {
            'oa': 0.0,
            'dice': 0.0,
            'iou': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        val_samples = 0
        
        with torch.no_grad():
            for v_img, v_msk in tqdm(val_loader, desc=" validating", leave=False):
                v_img, v_msk = v_img.to(device, non_blocking=True), v_msk.to(device, non_blocking=True)
                out = net(v_img)
                met = calculate_metrics(out, v_msk)
                # 新增：单独计算验证集dice
                dice_val = dice_coefficient(out, v_msk)
                bsz = v_img.size(0)
                
                # 累加所有验证指标
                for key in val_metrics_sum:
                    # 新增：对dice单独处理
                    if key == 'dice':
                        val_metrics_sum[key] += dice_val * bsz
                    else:
                        val_metrics_sum[key] += met[key] * bsz
                val_samples += bsz
        
        # 计算平均验证指标
        val_metrics = {k: v / val_samples for k, v in val_metrics_sum.items()}
        
        net.train()
        
        print(f"\nEpoch {epoch + 1} 总结:")
        print(f" - 平均损失: {epoch_loss:.4f}")
        print(f" - 平均Dice: {epoch_dice:.4f}")
        print(f" - 训练集OA: {epoch_oa:.4f}")
        print(f" - 验证集OA: {val_metrics['oa']:.4f}")
        print(f" - 验证集Precision: {val_metrics['precision']:.4f}")
        print(f" - 验证集Recall: {val_metrics['recall']:.4f}")
        print(f" - 交并比 (IoU): {epoch_iou:.4f}")
        print(f" - F1分数: {epoch_f1:.4f}")
        print(f" - 精确率: {epoch_precision:.4f}")
        print(f" - 召回率: {epoch_recall:.4f}")
        print(f" - 学习率: {scheduler.get_last_lr()[0]:.2e}")
        
        dice_history.append(epoch_dice)
        loss_history.append(epoch_loss)
        oa_history.append(epoch_oa)
        iou_history.append(epoch_iou)
        f1_history.append(epoch_f1)
        precision_history.append(epoch_precision)
        recall_history.append(epoch_recall)
        
        latest_path = os.path.join(experiment_dir, "latest.pth")
        torch.save(net.state_dict(), latest_path)
        
        # 修改：使用验证集OA作为早停标准，并保存完整验证指标
        if val_metrics['oa'] > best_oa + min_delta:
            best_oa = val_metrics['oa']
            # 更新验证集最佳指标
            best_val_metrics = val_metrics.copy()
            epochs_without_improvement = 0
            
            temp_path = os.path.join(experiment_dir, "best_temp.pth")
            final_path = os.path.join(experiment_dir, "best.pth")
            
            state = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_dice': best_oa,
                'best_oa': best_oa,
                'best_val_metrics': best_val_metrics,  # 新增：保存验证集最佳指标
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': {
                    'use_cbam': config.USE_CBAM,
                    'use_res': config.USE_RES,
                    'use_bilinear': config.USE_BILINEAR,
                    'use_ds': config.USE_DS,
                    'use_ghost': config.USE_GHOST,
                    'reduce_channels': config.REDUCE_CHANNELS
                }
            }
            
            torch.save(state, temp_path)
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_path, final_path)
            
            print(f"✅ 更新最佳模型 (OA={best_oa:.4f}, Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"📉 无改进: {epochs_without_improvement}/{patience} epochs")
        
        if epochs_without_improvement >= patience:
            print(f"🛑 早停触发! 连续 {patience} epochs 没有改进")
            print(f"🏆 最终最佳 OA: {best_oa:.4f}")
            break
        
        scheduler.step()
    
    history = {
        'dice_history': dice_history,
        'loss_history': loss_history,
        'oa_history': oa_history,
        'iou_history': iou_history,
        'f1_history': f1_history,
        'precision_history': precision_history,
        'recall_history': recall_history,
        'best_oa': best_oa,
        'total_epochs': len(dice_history)
    }
    
    history_path = os.path.join(experiment_dir, "history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, cls=NpEncoder)
    
    plt.figure(figsize=(10, 5))
    plt.plot(dice_history)
    plt.title(f'Dice Coefficient over Epochs - {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.grid(True)
    plt.savefig(os.path.join(experiment_dir, 'dice_coefficient.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='red')
    plt.title(f'Training Loss over Epochs - {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(experiment_dir, 'loss.png'))
    plt.close()
    
    best_epoch_idx = np.argmax(dice_history)
    
    summary = {
        'experiment_name': experiment_name,
        'model_config': {
            'use_cbam': config.USE_CBAM,
            'use_res': config.USE_RES,
            'use_bilinear': config.USE_BILINEAR,
            'use_ds': config.USE_DS,
            'use_ghost': config.USE_GHOST,
            'reduce_channels': config.REDUCE_CHANNELS
        },
        'model_complexity': {
            'parameters_M': params_m,
            'flops_G': flops_g
        },
        'training_results': {
            # 验证集最佳指标
            'best_val_oa': best_val_metrics['oa'],
            'best_val_precision': best_val_metrics['precision'],
            'best_val_recall': best_val_metrics['recall'],
            'best_val_dice': best_val_metrics['dice'],
            'best_val_iou': best_val_metrics['iou'],
            'best_val_f1': best_val_metrics['f1'],
            # 训练集最佳指标（保持原有）
            'best_oa': oa_history[best_epoch_idx],
            'best_dice': dice_history[best_epoch_idx],
            'best_iou': iou_history[best_epoch_idx],
            'best_f1': f1_history[best_epoch_idx],
            'best_precision': precision_history[best_epoch_idx],
            'best_recall': recall_history[best_epoch_idx],
            'best_epoch': best_epoch_idx + 1,
            'total_epochs': len(dice_history),
            # 最终指标
            'final_oa': oa_history[-1],
            'final_dice': dice_history[-1],
            'final_iou': iou_history[-1],
            'final_f1': f1_history[-1],
            'final_precision': precision_history[-1],
            'final_recall': recall_history[-1]
        }
    }
    
    summary_path = os.path.join(experiment_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    
    print(f"\n{'='*80}")
    print(f"📊 实验 {experiment_name} 完成")
    print(f"{'='*80}")
    print(f"最佳验证OA: {best_val_metrics['oa']:.4f}")
    print(f"最佳验证Precision: {best_val_metrics['precision']:.4f}")
    print(f"最佳验证Recall: {best_val_metrics['recall']:.4f}")
    print(f"最佳验证Dice: {best_val_metrics['dice']:.4f}")
    print(f"最佳验证IoU: {best_val_metrics['iou']:.4f}")
    print(f"最佳验证F1: {best_val_metrics['f1']:.4f}")
    print(f"训练集最佳Dice: {dice_history[best_epoch_idx]:.4f}")
    print(f"模型参数量: {params_m:.2f}M")
    print(f"模型FLOPs: {flops_g:.2f}G")
    print(f"{'='*80}\n")
    
    return summary

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练优化后的UNet模型')
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='实验名称')
    parser.add_argument('--save-dir', type=str, default='./ablation_experiments/',
                        help='实验保存目录（默认当前目录下创建）')
    # 新增：数据根目录参数，替代硬编码的 /root/ST/
    parser.add_argument('--data-root', type=str, required=True,
                        help='数据集根目录路径，应包含 train_images, train_masks, inputs/val 等子目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')

    parser.add_argument('--use-cbam', action='store_true',
                        help='使用CBAM注意力模块')
    parser.add_argument('--use-bilinear', action='store_true',
                        help='使用双线性插值上采样')
    parser.add_argument('--use-ds', action='store_true',
                        help='使用深度可分离卷积')
    parser.add_argument('--use-ghost', action='store_true',
                        help='使用Ghost模块')
    parser.add_argument('--reduce-channels', action='store_true',
                        help='缩减通道数')

    args = parser.parse_args()

    # 新增：构建数据路径（替代原来的 base_dir = r"/root/ST/"）
    base_dir = args.data_root
    img_dir = os.path.join(base_dir, "train_images")
    mask_dir = os.path.join(base_dir, "train_masks")

    # 新增：构建验证集路径（替代原来的硬编码）
    val_img_dir = os.path.join(base_dir, "inputs", "val", "images")
    val_mask_dir = os.path.join(base_dir, "inputs", "val", "masks", "0")

    config = ModelConfig(
        use_cbam=args.use_cbam,
        use_res=True,
        use_bilinear=args.use_bilinear,
        use_ds=args.use_ds,
        use_ghost=args.use_ghost,
        reduce_channels=args.reduce_channels
    )

    # 修改：将路径参数传递给 train_model
    train_model(config, args.experiment_name, args.save_dir, args.resume,
                img_dir, mask_dir, val_img_dir, val_mask_dir)