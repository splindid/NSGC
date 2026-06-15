# NSGC: Nonlinear Semantic-Geometric Residual Correction for Unbiased Scene Graph Generation

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.2+-orange)](https://pytorch.org/)

This repository contains the official implementation of our paper:

> **Nonlinear Semantic-Geometric Residual Correction for Unbiased Scene Graph Generation**  
> Hao Zhang, Xudong Li, Lizhen Wu*, Anqi Liu  
> *IEEE Signal Processing Letters*, 2026

## Overview

Scene graph generation (SGG) models trained on long-tailed datasets tend to predict head-class relations while neglecting tail-class ones. We propose a **lightweight post-processing residual correction framework** that recalibrates the logits of pretrained SGG models without architectural modifications.

Our method:
- Designs a **nonlinear semantic-geometric interaction module** that fuses subject, object, and geometric features
- Introduces a **residual decomposition** separating corrections into instance-adaptive, category-pair prior, and global components, modulated by class-adaptive scaling
- Achieves **up to 138% mean-recall improvement** while maintaining competitive R@K
- Is **plug-and-play**: applicable to any logit-producing baseline (Motifs, VCTree, Transformer, etc.)

<p align="center">
  <img src="docs/framework.png" width="90%">
</p>

## Main Results

Comparison with similar debiasing methods on Visual Genome (VG150):

### PredCls

| Method | mR@20/50/100 | R@20/50/100 |
|--------|:---:|:---:|
| Motifs | 12.80/17.50/19.17 | 58.46/65.88/67.47 |
| Motifs + **Ours** | 28.00/34.70/40.18 | 50.09/55.90/57.55 |
| VCTree | 14.26/17.90/19.85 | 59.18/66.09/67.96 |
| VCTree + **Ours** | 28.06/35.20/41.30 | 50.63/56.81/58.98 |
| Transformer | 12.80/17.10/17.17 | 58.46/65.88/66.97 |
| Transformer + **Ours** | 25.98/34.51/28.61 | 51.19/56.95/58.51 |

### SGCls

| Method | mR@20/50/100 | R@20/50/100 |
|--------|:---:|:---:|
| Motifs + **Ours** | 10.92/13.63/14.26 | 40.12/42.15/43.83 |
| VCTree + **Ours** | 10.80/14.08/14.78 | 40.61/43.63/44.35 |
| Transformer + **Ours** | 11.47/12.72/14.15 | 40.71/43.06/43.75 |

### SGDet

| Method | mR@20/50/100 | R@20/50/100 |
|--------|:---:|:---:|
| Motifs + **Ours** | 9.40/12.72/15.83 | 26.23/32.60/32.97 |
| VCTree + **Ours** | 9.36/13.39/15.00 | 27.69/33.55/36.08 |
| Transformer + **Ours** | 9.35/12.98/15.78 | 22.61/30.22/32.56 |

## Installation

Follow the steps below to set up the environment:

```bash
# Create conda environment
conda create -n sgg python=3.7
conda activate sgg

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this project
git clone https://github.com/splindid/NSGC.git
cd NSGC
pip install -r requirements.txt
python setup.py build develop
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Dataset

We use the Visual Genome (VG150) dataset. See [DATASET.md](DATASET.md) for download and preprocessing instructions.

## Usage

### Step 1: Prepare Pretrained SGG Model

Train a baseline SGG model (e.g., Motifs) using the standard training script, or download pretrained checkpoints:

```bash
# Train baseline (e.g., Motifs PredCls)
python train.py --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```

### Step 2: Train the Residual Correction Module

```bash
python train_bias_residual_v2.py
```

### Step 3: Evaluate

```bash
python test_bias_residual_v2.py
```

## Project Structure

```
NSGC/
├── maskrcnn_benchmark/
│   └── modeling/roi_heads/relation_head/
│       ├── CorrectionAlgorithm_BiasResidual.py  # Core: Pairwise Bias Residual Corrector
│       ├── my_DataStatistic.py                  # Data statistics utilities
│       ├── model_motifs.py                      # Motifs baseline
│       ├── model_vctree.py                      # VCTree baseline
│       └── model_transformer.py                 # Transformer baseline
├── train_bias_residual_v2.py                    # Training script for correction module
├── test_bias_residual_v2.py                     # Evaluation script
├── configs/                                     # Model configuration files
├── tools/                                       # Utility scripts
└── train.py / test.py                           # Baseline training/testing
```

## Key Files

| File | Description |
|------|-------------|
| `CorrectionAlgorithm_BiasResidual.py` | The nonlinear semantic-geometric residual correction module |
| `my_DataStatistic.py` | Predicate frequency statistics and class-adaptive weight computation |
| `train_bias_residual_v2.py` | Training pipeline for the correction module |
| `test_bias_residual_v2.py` | Evaluation with standard SGG metrics |



## Acknowledgement

This codebase is built upon [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) by Kaihua Tang. We thank the authors for their excellent work.

## License

This project is released under the [MIT License](LICENSE).
