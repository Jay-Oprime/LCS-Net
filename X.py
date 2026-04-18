#!/usr/bin/env python3
"""
全自动消融实验控制程序
执行9个不同的实验配置，自动训练和评估
"""

import os
import sys
import json
import subprocess
import argparse
import shutil
from datetime import datetime
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==================== 实验配置 ====================
EXPERIMENTS = [
    {
        "name": "Exp01_UNet_Baseline",
        "description": "实验一：基础UNet",
        "config": {
            "use_cbam": False,
            "use_bilinear": False,
            "use_ds": False,
            "use_ghost": False,
            "reduce_channels": False
        }
    },
    {
        "name": "Exp02_UNet_CBAM",
        "description": "实验二：UNet+CBAM",
        "config": {
            "use_cbam": True,
            "use_bilinear": False,
            "use_ds": False,
            "use_ghost": False,
            "reduce_channels": False
        }
    },
    {
        "name": "Exp03_UNet_Bilinear",
        "description": "实验三：UNet+双线性插值与卷积细化",
        "config": {
            "use_cbam": False,
            "use_bilinear": True,
            "use_ds": False,
            "use_ghost": False,
            "reduce_channels": False
        }
    },
    {
        "name": "Exp04_UNet_Bilinear_CBAM",
        "description": "实验四：UNet+双线性插值与卷积细化+CBAM",
        "config": {
            "use_cbam": True,
            "use_bilinear": True,
            "use_ds": False,
            "use_ghost": False,
            "reduce_channels": False
        }
    },
    {
        "name": "Exp05_UNet_ReducedChannels",
        "description": "实验五：UNet+通道数缩减",
        "config": {
            "use_cbam": False,
            "use_bilinear": False,
            "use_ds": False,
            "use_ghost": False,
            "reduce_channels": True
        }
    },
    {
        "name": "Exp06_UNet_Ghost_ReducedChannels",
        "description": "实验六：UNet+Ghost+通道数缩减",
        "config": {
            "use_cbam": False,
            "use_bilinear": False,
            "use_ds": False,
            "use_ghost": True,
            "reduce_channels": True
        }
    },
    {
        "name": "Exp07_UNet_Ghost_DSC_ReducedChannels",
        "description": "实验七：UNet+Ghost+DSC+通道数缩减",
        "config": {
            "use_cbam": False,
            "use_bilinear": False,
            "use_ds": True,
            "use_ghost": True,
            "reduce_channels": True
        }
    },
    {
        "name": "Exp08_UNet_Ghost_DSC_ReducedChannels_CBAM",
        "description": "实验八：UNet+Ghost+DSC+通道数缩减+CBAM",
        "config": {
            "use_cbam": True,
            "use_bilinear": False,
            "use_ds": True,
            "use_ghost": True,
            "reduce_channels": True
        }
    },
    {
        "name": "Exp09_UNet_All_Optimizations",
        "description": "实验九：UNet+Ghost+DSC+通道数缩减+CBAM+双线性插值与卷积细化",
        "config": {
            "use_cbam": True,
            "use_bilinear": True,
            "use_ds": True,
            "use_ghost": True,
            "reduce_channels": True
        }
    }
]

# ==================== 实验管理器 ====================
class AblationExperimentManager:
    def __init__(self, base_dir="./ablation_experiments", data_root=None):
        self.base_dir = base_dir
        self.data_root = data_root  # 新增：存储数据根目录
        self.results = []
        self.start_time = datetime.now()
        self.experiment_times = {}

        # 检查数据目录是否提供
        if data_root is None:
            raise ValueError("必须提供数据根目录 --data-root，例如：/root/ST 或 ./dataset")

        os.makedirs(base_dir, exist_ok=True)
        self.summary_dir = os.path.join(base_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)

        print(f"🔬 消融实验管理器初始化")
        print(f"📁 实验保存目录: {base_dir}")
        print(f"📂 数据来源目录: {data_root}")  # 新增打印
        print(f"📊 汇总目录: {self.summary_dir}")
        print(f"⏰ 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def run_experiment(self, experiment, resume=False):
        """运行单个实验"""
        print(f"\n{'='*80}")
        print(f"🧪 开始实验: {experiment['name']}")
        print(f"📝 描述: {experiment['description']}")
        print(f"{'='*80}\n")
        
        experiment_dir = os.path.join(self.base_dir, experiment['name'])
        
        # 检查是否已经存在结果
        summary_file = os.path.join(experiment_dir, "summary.json")
        if os.path.exists(summary_file) and not resume:
            print(f"✅ 实验 {experiment['name']} 已完成，跳过")
            with open(summary_file, 'r') as f:
                result = json.load(f)
            self.results.append(result)
            return result
        
        # 记录实验开始时间
        exp_start_time = datetime.now()
        
        # 构建训练命令
        # 构建训练命令
        cmd = [
            "python", "train_optimized.py",
            "--experiment-name", experiment['name'],
            "--save-dir", self.base_dir,
            "--data-root", self.data_root  # 新增：传递数据根目录给训练脚本
        ]
        
        # 添加配置参数
        config = experiment['config']
        if config['use_cbam']:
            cmd.append("--use-cbam")
        if config['use_bilinear']:
            cmd.append("--use-bilinear")
        if config['use_ds']:
            cmd.append("--use-ds")
        if config['use_ghost']:
            cmd.append("--use-ghost")
        if config['reduce_channels']:
            cmd.append("--reduce-channels")
        
        # 如果有检查点，恢复训练
        best_model = os.path.join(experiment_dir, "best.pth")
        if resume and os.path.exists(best_model):
            cmd.extend(["--resume", best_model])
            print(f"🔄 从检查点恢复训练: {best_model}")
        
        print(f"🚀 执行命令: {' '.join(cmd)}")
        
        # 执行训练
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"✅ 实验 {experiment['name']} 训练完成")
            
            # 记录实验结束时间
            exp_end_time = datetime.now()
            exp_duration = exp_end_time - exp_start_time
            self.experiment_times[experiment['name']] = {
                'start_time': exp_start_time.isoformat(),
                'end_time': exp_end_time.isoformat(),
                'duration': str(exp_duration),
                'duration_seconds': exp_duration.total_seconds()
            }
            
            # 加载结果
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    result = json.load(f)
                self.results.append(result)
                return result
            else:
                print(f"⚠️ 未找到结果文件: {summary_file}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {experiment['name']} 失败")
            print(f"错误信息: {e}")
            # 记录失败的实验时间
            exp_end_time = datetime.now()
            exp_duration = exp_end_time - exp_start_time
            self.experiment_times[experiment['name']] = {
                'start_time': exp_start_time.isoformat(),
                'end_time': exp_end_time.isoformat(),
                'duration': str(exp_duration),
                'duration_seconds': exp_duration.total_seconds(),
                'failed': True
            }
            return None
    
    def evaluate_experiment(self, experiment):
        """评估单个实验"""
        print(f"\n{'='*80}")
        print(f"📊 开始评估: {experiment['name']}")
        print(f"{'='*80}\n")
        
        experiment_dir = os.path.join(self.base_dir, experiment['name'])
        best_model = os.path.join(experiment_dir, "best.pth")
        config_file = os.path.join(experiment_dir, "config.yml")
        
        if not os.path.exists(best_model):
            print(f"⚠️ 未找到模型文件: {best_model}")
            return None
        
        # 创建评估输出目录
        eval_dir = os.path.join(experiment_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        # 构建测试集路径（基于 data_root）
        test_img_dir = os.path.join(self.data_root, "inputs", "test", "images")
        test_mask_dir = os.path.join(self.data_root, "inputs", "test", "masks", "0")

        # 构建评估命令
        cmd = [
            "python", "val_optimized.py",
            "--model", best_model,
            "--images", test_img_dir,
            "--masks", test_mask_dir,
            "--output", eval_dir,
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ]
        
        # 添加配置文件
        if os.path.exists(config_file):
            cmd.extend(["--config", config_file])
        
        print(f"🚀 执行命令: {' '.join(cmd)}")
        
        # 执行评估
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"✅ 实验 {experiment['name']} 评估完成")
            
            # 检查评估结果
            eval_summary = os.path.join(eval_dir, "evaluation_summary.json")
            if os.path.exists(eval_summary):
                with open(eval_summary, 'r') as f:
                    eval_result = json.load(f)
                print(f"📈 评估结果:")
                print(f"   - 平均IOU: {eval_result['average_metrics']['iou']:.4f}")
                print(f"   - 平均Dice: {eval_result['average_metrics']['dice']:.4f}")
                print(f"   - 平均F1: {eval_result['average_metrics']['f1_score']:.4f}")
                return eval_result
            else:
                print(f"⚠️ 未找到评估结果文件: {eval_summary}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验 {experiment['name']} 评估失败")
            print(f"错误信息: {e}")
            return None
    
    def generate_summary_report(self):
        """生成汇总报告"""
        print(f"\n{'='*80}")
        print(f"📊 生成汇总报告")
        print(f"{'='*80}\n")
        
        if not self.results:
            print("⚠️ 没有实验结果可汇总")
            return
        
        # 汇总所有结果
        summary = {
            "experiment_info": {
                "total_experiments": len(self.results),
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": str(datetime.now() - self.start_time)
            },
            "experiments": []
        }
        
        # 添加每个实验的结果
        for result in self.results:
            exp_summary = {
                "name": result["experiment_name"],
                "config": result["model_config"],
                "complexity": result["model_complexity"],
                "results": result["training_results"]
            }
            summary["experiments"].append(exp_summary)
        
        # 按最佳OA排序
        summary["experiments"].sort(key=lambda x: x["results"]["best_val_oa"], reverse=True)
        
        # 保存汇总报告
        summary_file = os.path.join(self.summary_dir, "ablation_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report(summary)
        
        # 生成CSV报告
        self.generate_csv_report(summary)
        
        # 生成可视化图表
        self.generate_visualization_plots()
        
        # 生成实验时间表
        self.generate_experiment_time_table()
        
        print(f"✅ 汇总报告已生成: {summary_file}")
        
        # 打印排名（修改：显示验证集指标）
        print(f"\n{'='*80}")
        print(f"🏆 实验结果排名（按验证集OA）")
        print(f"{'='*80}")
        for i, exp in enumerate(summary["experiments"], 1):
            print(f"{i}. {exp['name']:<50} "
                  f"Val OA: {exp['results']['best_val_oa']:.4f}  "
                  f"Val Prec: {exp['results']['best_val_precision']:.4f}  "
                  f"Val Rec: {exp['results']['best_val_recall']:.4f}  "
                  f"Dice: {exp['results']['best_dice']:.4f}  "
                  f"Params: {exp['complexity']['parameters_M']:.2f}M")
        print(f"{'='*80}\n")
    
    def generate_markdown_report(self, summary):
        """生成Markdown格式报告"""
        md_file = os.path.join(self.summary_dir, "ablation_report.md")
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 消融实验报告\n\n")
            f.write(f"**实验时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**总实验数**: {len(self.results)}\n")
            f.write(f"**总耗时**: {datetime.now() - self.start_time}\n\n")
            
            f.write("## 实验配置\n\n")
            # 修改：添加验证集Precision和Recall列
            f.write("| 实验名称 | CBAM | 双线性插值 | DSC | Ghost | 通道缩减 | 参数量(M) | FLOPs(G) | Val OA | Val Prec | Val Rec | Train Dice |")
            f.write("\n|----------|------|------------|-----|-------|----------|-----------|----------|--------|----------|---------|------------|\n")
            
            for exp in summary["experiments"]:
                cfg = exp["config"]
                res = exp["results"]
                comp = exp["complexity"]
                
                f.write(f"| {exp['name']} | {cfg['use_cbam']} | {cfg['use_bilinear']} | "
                        f"{cfg['use_ds']} | {cfg['use_ghost']} | {cfg['reduce_channels']} | "
                        f"{comp['parameters_M']:.2f} | {comp['flops_G']:.2f} | "
                        f"{res['best_val_oa']:.4f} | {res['best_val_precision']:.4f} | "
                        f"{res['best_val_recall']:.4f} | {res['best_dice']:.4f} |\n")
            
            f.write("\n## 详细结果\n\n")
            for exp in summary["experiments"]:
                f.write(f"### {exp['name']}\n\n")
                f.write(f"**描述**: {next(e['description'] for e in EXPERIMENTS if e['name'] == exp['name'])}\n\n")
                f.write("**配置**:\n")
                f.write(f"- CBAM注意力: {exp['config']['use_cbam']}\n")
                f.write(f"- 双线性插值: {exp['config']['use_bilinear']}\n")
                f.write(f"- 深度可分离卷积: {exp['config']['use_ds']}\n")
                f.write(f"- Ghost模块: {exp['config']['use_ghost']}\n")
                f.write(f"- 通道数缩减: {exp['config']['reduce_channels']}\n\n")
                
                f.write("**验证集最佳结果**:\n")
                f.write(f"- 最佳验证OA: {exp['results']['best_val_oa']:.4f}\n")
                f.write(f"- 最佳验证Precision: {exp['results']['best_val_precision']:.4f}\n")
                f.write(f"- 最佳验证Recall: {exp['results']['best_val_recall']:.4f}\n")
                f.write(f"- 最佳验证Dice: {exp['results']['best_val_dice']:.4f}\n")
                f.write(f"- 最佳验证IoU: {exp['results']['best_val_iou']:.4f}\n")
                f.write(f"- 最佳验证F1: {exp['results']['best_val_f1']:.4f}\n\n")
                
                f.write("**训练集最佳结果**:\n")
                f.write(f"- 最佳训练Dice: {exp['results']['best_dice']:.4f}\n")
                f.write(f"- 最佳训练IoU: {exp['results']['best_iou']:.4f}\n")
                f.write(f"- 最佳训练F1: {exp['results']['best_f1']:.4f}\n\n")
                
                f.write(f"**模型复杂度**:\n")
                f.write(f"- 模型参数量: {exp['complexity']['parameters_M']:.2f}M\n")
                f.write(f"- 模型FLOPs: {exp['complexity']['flops_G']:.2f}G\n\n")
        
        print(f"✅ Markdown报告已生成: {md_file}")

    def generate_csv_report(self, summary):
        """生成CSV格式报告"""
        import csv
        
        csv_file = os.path.join(self.summary_dir, "ablation_results.csv")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 修改：添加验证集Precision和Recall列
            writer.writerow([
                "实验名称", "CBAM", "双线性插值", "DSC", "Ghost", "通道缩减",
                "参数量(M)", "FLOPs(G)", "Val OA", "Val Precision", "Val Recall", 
                "Val Dice", "Val IoU", "Val F1", "Train Dice", "Train IoU", "Train F1"
            ])
            
            # 写入数据
            for exp in summary["experiments"]:
                cfg = exp["config"]
                res = exp["results"]
                comp = exp["complexity"]
                
                writer.writerow([
                    exp['name'],
                    cfg['use_cbam'],
                    cfg['use_bilinear'],
                    cfg['use_ds'],
                    cfg['use_ghost'],
                    cfg['reduce_channels'],
                    f"{comp['parameters_M']:.2f}",
                    f"{comp['flops_G']:.2f}",
                    f"{res['best_val_oa']:.4f}",
                    f"{res['best_val_precision']:.4f}",
                    f"{res['best_val_recall']:.4f}",
                    f"{res['best_val_dice']:.4f}",
                    f"{res['best_val_iou']:.4f}",
                    f"{res['best_val_f1']:.4f}",
                    f"{res['best_dice']:.4f}",
                    f"{res['best_iou']:.4f}",
                    f"{res['best_f1']:.4f}"
                ])
        
        print(f"✅ CSV报告已生成: {csv_file}")
    
    def generate_visualization_plots(self):
        """生成可视化图表"""
        print(f"\n{'='*80}")
        print(f"📈 生成训练曲线可视化图表")
        print(f"{'='*80}\n")
        
        # 定义颜色映射 - 使用9种不同的颜色
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'
        ]
        
        # 收集所有实验的历史数据
        all_histories = {}
        for experiment in EXPERIMENTS:
            exp_name = experiment['name']
            history_file = os.path.join(self.base_dir, exp_name, "history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    all_histories[exp_name] = history
            else:
                print(f"⚠️ 未找到历史数据文件: {history_file}")
        
        if not all_histories:
            print("⚠️ 没有历史数据可生成图表")
            return
        
        # 创建图表
        metrics_to_plot = [
            ('iou_history', 'mIoU', 'mIoU over Epochs', 'mIoU'),
            ('dice_history', 'Dice Coefficient', 'Dice Coefficient over Epochs', 'Dice'),
            ('f1_history', 'F1 Score', 'F1 Score over Epochs', 'F1'),
            ('oa_history', 'Overall Accuracy (OA)', 'OA over Epochs', 'OA'),
            ('precision_history', 'Precision', 'Precision over Epochs', 'Precision')
        ]
        
        for metric_key, ylabel, title, filename_suffix in metrics_to_plot:
            plt.figure(figsize=(14, 8))
            
            for i, (exp_name, history) in enumerate(all_histories.items()):
                if metric_key in history:
                    epochs = range(1, len(history[metric_key]) + 1)
                    plt.plot(epochs, history[metric_key], 
                           color=colors[i % len(colors)], 
                           linewidth=2, 
                           label=exp_name.replace('Exp', ''), 
                           marker='o', 
                           markersize=3,
                           markevery=max(1, len(epochs)//20))  # 每20个点显示一个marker
            
            plt.xlabel('Epoch', fontsize=14, fontweight='bold')
            plt.ylabel(ylabel, fontsize=14, fontweight='bold')
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(loc='best', fontsize=10, ncol=2, framealpha=0.9)
            
            # 设置背景
            ax = plt.gca()
            ax.set_facecolor('#fafafa')
            
            # 保存图表
            output_path = os.path.join(self.summary_dir, f'all_experiments_{filename_suffix.lower()}.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ 生成图表: {output_path}")
        
        print(f"\n✅ 所有可视化图表已生成到: {self.summary_dir}")
    
    def generate_experiment_time_table(self):
        """生成实验时间表"""
        print(f"\n{'='*80}")
        print(f"⏱️ 生成实验时间表")
        print(f"{'='*80}\n")
        
        if not self.experiment_times:
            print("⚠️ 没有实验时间数据")
            return
        
        # 创建表格数据
        time_data = []
        for exp_name, time_info in self.experiment_times.items():
            time_data.append({
                '实验名称': exp_name,
                '开始时间': time_info['start_time'],
                '结束时间': time_info['end_time'],
                '耗时': time_info['duration'],
                '耗时(秒)': f"{time_info['duration_seconds']:.2f}",
                '状态': '失败' if time_info.get('failed', False) else '成功'
            })
        
        # 创建DataFrame
        df = pd.DataFrame(time_data)
        
        # 保存为CSV
        csv_path = os.path.join(self.summary_dir, 'experiment_times.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 实验时间表已保存: {csv_path}")
        
        # 保存为Excel
        try:
            excel_path = os.path.join(self.summary_dir, 'experiment_times.xlsx')
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"✅ 实验时间表已保存: {excel_path}")
        except ImportError:
            print("⚠️ 未安装openpyxl，无法生成Excel文件")
        
        # 打印表格
        print(f"\n{'='*80}")
        print(f"📋 实验时间汇总")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"{'='*80}\n")
        
        # 计算总时间
        total_seconds = sum(float(row['耗时(秒)']) for row in time_data)
        total_time = pd.to_timedelta(total_seconds, unit='s')
        print(f"总实验时间: {total_time}")
        print(f"平均实验时间: {pd.to_timedelta(total_seconds/len(time_data), unit='s')}")
    
    def run_all_experiments(self, start_from=0, evaluate_only=False):
        """运行所有实验"""
        print(f"\n{'='*80}")
        print(f"🚀 开始执行所有 {len(EXPERIMENTS)} 个实验")
        print(f"{'='*80}\n")
        
        for i, experiment in enumerate(EXPERIMENTS):
            if i < start_from:
                print(f"⏭️ 跳过实验 {i+1}/{len(EXPERIMENTS)}: {experiment['name']}")
                continue
            
            print(f"\n📋 实验进度: {i+1}/{len(EXPERIMENTS)}")
            
            if not evaluate_only:
                # 运行训练
                result = self.run_experiment(experiment, resume=True)
            else:
                # 只评估
                result = None
            
            # 评估
            eval_result = self.evaluate_experiment(experiment)
            
            print(f"\n{'='*80}")
            print(f"✅ 实验 {i+1}/{len(EXPERIMENTS)} 完成: {experiment['name']}")
            print(f"{'='*80}\n")
        
        # 生成汇总报告
        self.generate_summary_report()
        
        print(f"\n🎉 所有实验完成！")
        print(f"📊 请查看汇总报告: {self.summary_dir}")

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description='全自动消融实验控制程序')
    parser.add_argument('--base-dir', type=str, default='./ablation_experiments',  # 建议改为相对路径
                        help='实验保存的基础目录（默认当前目录下ablation_experiments）')
    # 新增：数据根目录参数（必需）
    parser.add_argument('--data-root', type=str, required=True,
                        help='数据集根目录路径，必须包含train_images, train_masks, inputs/val等子目录')
    parser.add_argument('--start-from', type=int, default=0,
                        help='从第几个实验开始（0表示从第一个开始）')
    parser.add_argument('--experiment-ids', type=int, nargs='+',
                        help='指定要运行的实验ID（1-9），不指定则运行所有')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='只进行评估，不训练')
    parser.add_argument('--single-experiment', type=str, default=None,
                        help='只运行单个实验（指定实验名称）')

    args = parser.parse_args()

    # 修改：传递data_root参数
    manager = AblationExperimentManager(args.base_dir, args.data_root)
    
    if args.single_experiment:
        # 运行单个实验
        experiment = next((e for e in EXPERIMENTS if e['name'] == args.single_experiment), None)
        if experiment:
            print(f"\n🧪 运行单个实验: {experiment['name']}\n")
            if not args.evaluate_only:
                manager.run_experiment(experiment)
            manager.evaluate_experiment(experiment)
        else:
            print(f"❌ 未找到实验: {args.single_experiment}")
            print("可用实验:")
            for exp in EXPERIMENTS:
                print(f"  - {exp['name']}")
    elif args.experiment_ids:
        # 运行指定的实验
        valid_ids = [i for i in args.experiment_ids if 1 <= i <= len(EXPERIMENTS)]
        if not valid_ids:
            print("❌ 无效的实验ID")
            return
        
        print(f"\n🧪 运行指定实验: {valid_ids}\n")
        experiments_to_run = [EXPERIMENTS[i-1] for i in valid_ids]
        
        for experiment in experiments_to_run:
            print(f"\n{'='*80}")
            print(f"开始实验: {experiment['name']}")
            print(f"{'='*80}\n")
            
            if not args.evaluate_only:
                manager.run_experiment(experiment)
            manager.evaluate_experiment(experiment)
        
        # 生成汇总报告
        manager.generate_summary_report()
    else:
        # 运行所有实验
        manager.run_all_experiments(start_from=args.start_from, 
                                   evaluate_only=args.evaluate_only)

if __name__ == "__main__":
    main()