
# LCS-Net: Lightweight Convolutional Segmentation Network for Near-Shore Water Mapping

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](LICENSE)

> **Associated Paper**: LCS-Net: A Lightweight U-Net Variant with Ghost-DSC and Channel Compression for Semantic Segmentation of Near-Shore Water Bodies  
> **Journal**: *Frontiers in Earth Science*  
> **Status**: Under Review  
> **Submitted**: April 2026  
> **Preprint**: Please cite this repository and the associated paper (DOI to be updated upon formal publication)

---

## 1. Introduction

This repository provides the official PyTorch implementation of **LCS-Net**, including model definitions, training pipelines, automated ablation experiment frameworks, and evaluation scripts. LCS-Net is built upon the U-Net encoder-decoder architecture, systematically incorporating five lightweight strategies: **Ghost modules**, **Depthwise Separable Convolution (DSC)**, **channel reduction**, **CBAM attention mechanisms**, and **bilinear interpolation upsampling**. The effectiveness of each module and their coupled effects are validated through nine controlled ablation experiments.

**Key Design Features**:
- **Configurable Architecture**: The `ModelConfig` class enables switch-based control of all five optimization strategies, supporting arbitrary combinations for ablation studies.
- **Dual-Platform Validation**: Frame rate and accuracy evaluations provided for both edge CPU (Intel i7-1360P, 6.96 FPS) and desktop GPU (NVIDIA RTX 3060, 58.42 FPS) deployments.
- **Automated Experimentation**: `X.py` supports one-click sequential execution of nine ablation experiments (Exp-01 to Exp-09), automatically generating CSV summaries, Markdown reports, and visualization curves.

---

## 2. Requirements

| Dependency | Recommended Version | Description |
|------------|---------------------|-------------|
| Python | ≥ 3.8 | Base runtime |
| PyTorch | ≥ 2.0 | Deep learning framework |
| torchvision | ≥ 0.15 | Image preprocessing and augmentation |
| thop | ≥ 0.1.1 | Model complexity analysis (parameters and FLOPs) |
| Pillow | ≥ 9.0 | Image I/O processing |
| PyYAML | ≥ 6.0 | Experiment configuration serialization |
| pandas | ≥ 1.3 | Experimental results aggregation |
| matplotlib | ≥ 3.5 | Training curves and visualization |
| numpy | ≥ 1.21 | Numerical computation |
| tqdm | ≥ 4.60 | Training progress monitoring |

### Installation

```bash
# 1. Clone this repository
git clone https://github.com/Jay-Oprime/LCS-Net.git
cd LCS-Net

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Note**: `thop` is an optional dependency used solely for FLOPs statistics; if not installed, the code will automatically degrade and skip this calculation without affecting training or inference.

---

## 3. Dataset Preparation and Availability Statement

This repository supports standard binary semantic segmentation dataset formats. Due to **UAV remote sensing image copyright restrictions and institutional data-sharing policies**, the Panzhihua near-shore water dataset and Chongming Island cross-domain validation dataset used in this study **cannot be publicly provided**. However, this repository provides complete data organization specifications and preprocessing pipelines, enabling researchers to replicate results using their own near-shore water datasets.

### Directory Structure Specification

Please organize your dataset according to the following hierarchy (path specified via `--data-root` parameter):

```
dataset_root/
├── train_images/          # Training images (*.jpg, RGB 3-channel)
├── train_masks/           # Training labels (*.png, single-channel binary, water>0, background=0)
└── inputs/
    ├── val/
    │   ├── images/        # Validation images
    │   └── masks/
    │       └── 0/         # Validation labels (*.png)
    └── test/
        ├── images/        # Test images
        └── masks/
            └── 0/         # Test labels (*.png)
```

**Naming Conventions**:
- Image and label filenames must strictly correspond (e.g., `sample_001.jpg` corresponds to `sample_001.png`).
- Recommended spatial resolution is 256×256 pixels, adjustable at runtime via `transforms.Resize`.

**Data Augmentation Strategy**:
- Built-in training augmentations include: random rotation (90°/180°/270°) and horizontal flipping.
- These are automatically executed via `transforms` in `train_optimized.py` without requiring preprocessing.

---
### 3.1 Demo Dataset (Sample Availability)

Due to UAV remote sensing image copyright restrictions and institutional confidentiality agreements, the **complete dataset (600 images)** used in this study cannot be publicly provided. It is available upon reasonable request subject to: (i) a signed Data Use Agreement (DUA) ensuring non-commercial use, (ii) ethical review approval, and (iii) a formal application letter stating research purposes.

To facilitate immediate testing and format verification, we provide **10 representative image-label pairs** in the `demo/` folder:
- **Contents**: Covering clear water, shadow interference, vegetation boundaries, and fragmented shoreline scenarios
- **Format**: RGB image (`.jpg`) + binary mask (`.png`, 0=non-water, 1=water)
- **Resolution**: 256×256 pixels (consistent with training pipeline)
- **Usage**: Allows testing of pretrained models without accessing the full private dataset

**Data Format Example**:
demo/
├── demo_001.jpg        # Original UAV image
├── demo_001_mask.png   # Binary mask (water=1, background=0)
├── demo_002.jpg
└── demo_002_mask.png

## 4. Quick Start and Usage Examples

### 4.1 Single Experiment Training (Exp-09 Full Configuration Example)

```bash
python train_optimized.py \
  --experiment-name Exp09_LCSNet \
  --data-root ./dataset_root \
  --save-dir ./experiments \
  --use-cbam \
  --use-bilinear \
  --use-ds \
  --use-ghost \
  --reduce-channels
```

**Key Parameter Descriptions**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `--data-root` | String | Root directory of dataset, should contain `train_images`, `train_masks`, and `inputs` subdirectories |
| `--save-dir` | String | Experiment output directory; subfolders will be automatically created for weights, configurations, and logs |
| `--use-cbam` | Boolean flag | Enable CBAM attention module |
| `--use-bilinear` | Boolean flag | Enable bilinear interpolation upsampling (replacing transposed convolution) |
| `--use-ds` | Boolean flag | Enable Depthwise Separable Convolution (DSC) |
| `--use-ghost` | Boolean flag | Enable Ghost module for cheap feature map generation |
| `--reduce-channels` | Boolean flag | Enable channel reduction (64→48, 128→96, ..., 1024→768) |
| `--resume` | String (optional) | Resume training from specified checkpoint (`.pth`) |

**Training Output Structure**:
```
experiments/Exp09_LCSNet/
├── best.pth              # Validation set OA optimal weights
├── latest.pth            # Latest epoch weights
├── config.yml            # Experiment configuration archive (including module switch states)
├── history.json          # Training history (Loss, Dice, IoU, OA, Precision, Recall)
├── summary.json          # Experiment summary report
├── dice_coefficient.png  # Training curve visualization
└── loss.png
```

### 4.2 Model Evaluation and Inference

Use `val_optimized.py` for batch inference on test sets, generating segmentation results and visual comparisons:

```bash
python val_optimized.py \
  --model ./experiments/Exp09_LCSNet/best.pth \
  --images ./dataset_root/inputs/test/images \
  --masks ./dataset_root/inputs/test/masks/0 \
  --output ./results/Exp09 \
  --device cuda
```

**Evaluation Outputs**:
- `*_pred_mask.png`: Binary prediction masks
- `*_prob_map.png`: Probability heatmaps (Sigmoid output)
- `*_visualization.png`: Six-panel visualization (original, ground truth, prediction, overlay, error analysis)
- `evaluation_summary.json`: Overall metrics (IoU, Dice, F1, OA, Precision, Recall)

### 4.3 One-Click Ablation Experiments (9 Configurations)

Execute the following command to automatically run all nine ablation experiments sequentially:

```bash
python X.py \
  --data-root ./dataset_root \
  --base-dir ./ablation_experiments
```

This command automatically generates:
- `ablation_summary.json`: Structured experiment summary
- `ablation_report.md`: Markdown-readable report
- `ablation_results.csv`: Tabular data (easily imported into Excel/LaTeX)
- `all_experiments_iou.png`: Multi-experiment IoU convergence curve comparison
- `experiment_times.csv`: Time statistics for each experiment

**Details of 9 Experimental Configurations**:

| Exp. ID | Configuration Name | CBAM | Bilinear | DSC | Ghost | Channel Reduction | Theoretical Position |
|:-------:|:-------------------|:----:|:--------:|:---:|:-----:|:-----------------:|:--------------------:|
| Exp-01 | UNet Baseline | ✗ | ✗ | ✗ | ✗ | ✗ | Control group |
| Exp-02 | +CBAM | ✓ | ✗ | ✗ | ✗ | ✗ | Attention mechanism verification |
| Exp-03 | +Bilinear | ✗ | ✓ | ✗ | ✗ | ✗ | Upsampling mechanism verification |
| Exp-04 | +Bilinear+CBAM | ✓ | ✓ | ✗ | ✗ | ✗ | Joint optimization verification |
| Exp-05 | +Channel Reduction | ✗ | ✗ | ✗ | ✗ | ✓ | **Precision ceiling exploration** |
| Exp-06 | +Ghost+Channel Reduction | ✗ | ✗ | ✗ | ✓ | ✓ | Lightweight decoupling verification |
| Exp-07 | +Ghost+DSC+Channel Reduction | ✗ | ✗ | ✓ | ✓ | ✓ | Convolution replacement verification |
| Exp-08 | +Ghost+DSC+Channel Reduction+CBAM | ✓ | ✗ | ✓ | ✓ | ✓ | Attention supplementation verification |
| **Exp-09** | **LCS-Net (Full Configuration)** | **✓** | **✓** | **✓** | **✓** | **✓** | **Optimal for engineering deployment** |

**Key Findings** (corresponding to Reviewer Comment 4):
- Exp-05 (channel reduction only) achieves **97.01% IoU** precision ceiling, but with 20.01 M parameters;
- Exp-09 (full LCS-Net) achieves **96.10% IoU** with **8.9× parameter compression** (2.24 M vs. 20.01 M) and **13.5× computational cost reduction** (8.24 G vs. 111.44 G FLOPs);
- In the context of edge deployment, the 96.10% IoU with smaller parameter size possesses greater scenario-adaptation value than the 97.01% IoU with larger parameter size.

---

## 5. Model Architecture Details

LCS-Net is based on the U-Net encoder-decoder architecture. Key modules are defined as follows:

**Conv_Block**: Dual-layer convolution (supporting standard / DSC / Ghost modes) + Batch Normalization / LeakyReLU activation + CBAM attention (optional) + residual connections.

**DownSample**: MaxPool2d downsampling + convolution feature extraction (supporting standard / DSC modes).

**UpSample**: Bilinear interpolation or transposed convolution upsampling, concatenated with encoder features via Skip Connection; channel alignment achieved through 1×1 pointwise convolution.

**Channel Configurations**:
- Standard configuration (Exp-01): 64-128-256-512-1024
- Reduced configuration (Exp-05/06/07/08/09): 48-96-192-384-768

---

## 6. Project Structure

```
LCS-Net/
├── net_optimized.py      # Model definition (UNet + ModelConfig + complexity calculation)
├── train_optimized.py    # Training script (supports mixed precision, early stopping, best validation save)
├── val_optimized.py      # Evaluation and visualization script
├── X.py                  # Automated ablation experiment scheduler
├── requirements.txt      # Python dependency list
├── README.md             # English documentation (this file)
└── LICENSE               # CC BY 4.0 License
```

---

## 7. Citation

If this code is helpful to your research, please cite:

```bibtex
@article{lv2026lcs,
  title={LCS-Net: A Lightweight U-Net Variant with Ghost-DSC and Channel Compression for Semantic Segmentation of Near-Shore Water Bodies},
  author={Lv, M. and Wang, H. and et al.},
  journal={Frontiers in Earth Science},
  year={2026},
  publisher={Frontiers Media SA},
  note={Under Review}
}
```

---

## 8. License and Copyright

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**. You are free to share and adapt this material, provided you give appropriate credit and indicate if changes were made.

**Acknowledgments**: We thank the reviewers for their constructive comments, which prompted us to supplement cross-domain migration experiments (Chongming Island dataset) and failure case analyses.
