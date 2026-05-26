# Face Anti-Spoofing Portfolio

A comprehensive face liveness detection project covering multiple model architectures — from training from scratch to SOTA fine-tuning — benchmarked on standard FAS datasets.

## Models

| Track | Model | Strategy | Dataset | ACER ↓ | AUC ↑ |
|-------|-------|----------|---------|--------|-------|
| 1 | ResNet-18 | From scratch | CASIA-FASD | TBD | TBD |
| 2 | EfficientNet-B2 | Pretrained (ImageNet) | OULU-NPU | TBD | TBD |
| 3 | ViT-B/16 + DINO | Fine-tune (self-supervised) | CelebA-Spoof | TBD | TBD |
| 4 | ConViT | Fine-tune (cross-domain) | OULU→Replay | TBD | TBD |

## Datasets

| Dataset | Samples | Attack Types | Link |
|---------|---------|-------------|------|
| OULU-NPU | 4,950 videos | Print, Replay | [Download](https://sites.google.com/site/oulunpudatabase/) |
| CelebA-Spoof | 625,537 images | Print, Replay, 3D mask | [GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) |
| CASIA-FASD | 600 videos | Print, Cut, Replay | [Papers with Code](https://paperswithcode.com/dataset/casia-fasd) |
| Replay-Attack | 1,200 videos | Print, Replay | [Idiap](https://www.idiap.ch/en/dataset/replayattack) |

## Metrics

- **ACER** — Average Classification Error Rate (primary benchmark metric)
- **AUC** — Area Under ROC Curve
- **EER** — Equal Error Rate
- **HTER** — Half Total Error Rate (cross-dataset evaluation)

## Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/face-anti-spoofing
cd face-anti-spoofing

# 2. Run setup (Windows + Anaconda + RTX 30xx)
setup.bat

# 3. Verify GPU
python verify_gpu.py
```

## Training

```bash
# Activate environment
conda activate fas

# Train any model by swapping the config
python src/training/train.py --config configs/resnet18.yaml
python src/training/train.py --config configs/efficientnet_b2.yaml
python src/training/train.py --config configs/vit_dino.yaml
```

## Demo

```bash
python demo/app.py --checkpoint results/checkpoints/best.ckpt --config configs/vit_dino.yaml
```

## Tech Stack

- **PyTorch 2.x** + **PyTorch Lightning** — training framework
- **timm** — model architectures (ViT, EfficientNet, ConViT)
- **OpenCV** + **MediaPipe** — face detection and alignment
- **albumentations** — augmentation pipeline
- **Weights & Biases** — experiment tracking
- **Gradio** — interactive demo
- **ONNX Runtime** — optimized inference

## Project Structure

```
├── configs/          # YAML configs per model
├── data/             # Dataset CSVs (not raw images)
├── src/
│   ├── data/         # DataLoader, augmentations
│   ├── models/       # Model factory (all architectures)
│   ├── training/     # Training loop (Lightning)
│   └── evaluation/   # ACER, AUC, EER, HTER metrics
├── demo/             # Gradio app
├── notebooks/        # EDA and result visualization
└── results/          # Checkpoints, logs, plots
```
