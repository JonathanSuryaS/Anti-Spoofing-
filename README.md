# Face Anti-Spoofing — Cross-Dataset Generalization Study

A face anti-spoofing (FAS) system that detects presentation attacks (printed
photos, screen replays, masks) versus genuine live faces.

The project is built around a **cross-dataset evaluation protocol**: the model is
trained on one dataset and tested on entirely separate, unseen datasets. This is
the realistic and rigorous measure of whether a FAS model genuinely generalizes,
rather than memorizing one dataset's cameras, lighting, and compression.

> **Why this framing matters.** FAS models are notorious for overfitting to
> *domain artifacts* and scoring >99% on their own test split while collapsing on
> faces from a different camera. This project measures that gap directly.

---

## Datasets & Roles

| Role  | Dataset | Notes |
|-------|---------|-------|
| Train | **CelebA-Spoof** (sampled subset) | Large, diverse, balanced live/spoof |
| Test  | **OULU-NPU** | Unseen domain: print & replay attacks |
| Test  | **Kaggle anti-spoofing set** | Unseen domain: phone replays, printouts, cut-outs |

Each test dataset lives in its own folder (`test_oulu/`, `test_kaggle/`) so metrics
are reported **per domain**. A dataset used in training cannot honestly measure
generalization, so the test datasets are never trained on.

## Pipeline Overview

```
raw data ─▶ frame extraction ─▶ MTCNN crop (224×224) ─▶ balanced subset
        ─▶ ResNet-50 (ImageNet) training ─▶ cross-dataset evaluation
        ─▶ metrics (APCER/BPCER/ACER/EER/AUC) ─▶ plots + error analysis
```

- **Preprocessing:** MTCNN face detection, 20px margin crop, center-crop fallback,
  resize to 224×224, ImageNet normalization.
- **Model:** ResNet-50, ImageNet-pretrained, final layer → 2-class (live/spoof).
  A standard backbone is intentional — the project's contribution is the
  cross-dataset evaluation and augmentation ablation, not backbone novelty.
- **Metrics:** APCER (attacks accepted), BPCER (live rejected), ACER (their mean),
  EER and AUC (threshold-independent). Accuracy is intentionally avoided — test
  sets are imbalanced and per-class error is what matters.

## Repository Structure

```
src/
├── preprocess/     frame extraction, MTCNN crop, balance diagnostics
│                   (config.py = single source of truth for all paths)
├── dataset/        FASDataset — multi-domain loader, ImageNet norm,
│                   swappable augmentation slot
├── models/         ResNet-50 + 2-class head
├── augmentation/   SPSC physical-artifact augmentation (for the ablation)
├── training/       baseline training loop
├── evaluation/     metrics, cross-dataset eval, plots, error analysis
└── utils/          class-weight helper, etc.

experiments/        per-experiment reports (see below)
results/            generated figures
checkpoints/        trained model weights (gitignored)
```

## How to Run

```bash
# 1. Preprocess (edit paths in src/preprocess/config.py first)
python src/preprocess/process_celeba.py     # -> train/
python src/preprocess/process_oulu.py       # -> test_oulu/
python src/preprocess/process_kaggle.py     # -> test_kaggle/
python src/preprocess/detect_and_crop.py    # crop all splits
python src/preprocess/diagnose_balance.py   # verify balance

# 2. Train
python src/training/train.py

# 3. Evaluate (cross-dataset)
python src/evaluation/evaluate.py
python src/evaluation/plot_results.py
python src/evaluation/error_analysis.py
```

## Experiments

| # | Name | Status | Report |
|---|------|--------|--------|
| 1 | ResNet-50 baseline, no augmentation | ✅ Done | [experiments/01_baseline.md](experiments/01_baseline.md) |
| 2 | SPSC augmentation ablation | ⬜ Planned | — |
| 3 | Grad-CAM interpretability | ⬜ Planned | — |

## Roadmap

1. **Threshold analysis** — report ACER at the EER threshold; the fixed 0.5 cutoff
   is the main weakness of the baseline.
2. **SPSC augmentation ablation** — does simulating physical attack artifacts
   improve cross-dataset generalization? (Core experiment.)
3. **Grad-CAM** — verify the model attends to real artifact regions, not shortcuts.
4. **Scale training** — larger CelebA subset on stronger hardware (config change).
5. **Multi-task backbone** — optional, match the CelebA-Spoof paper's AENet design.

---

*Cross-dataset results are reported honestly rather than replaced with flattering
in-domain figures. The gap between in-domain (~0.995 acc) and cross-dataset
(AUC ~0.83) performance is itself a central finding — see the experiment reports.*
