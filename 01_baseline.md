# Experiment 1 — ResNet-50 Baseline (No Augmentation)

**Status:** Complete
**Goal:** Establish an honest cross-dataset baseline before adding any
augmentation, so later experiments have a fixed reference point.

---

## Setup

| Component | Choice |
|-----------|--------|
| Train data | CelebA-Spoof subset, balanced (~2,150 live / 2,205 spoof) |
| Test data | OULU-NPU, Kaggle anti-spoofing (both unseen) |
| Backbone | ResNet-50, ImageNet-pretrained (`IMAGENET1K_V2`) |
| Head | FC → 2 classes (live / spoof) |
| Loss | CrossEntropyLoss (no class weights — train set is balanced) |
| Optimizer | Adam, lr 1e-4 |
| Epochs | 10 (converged early) |
| Augmentation | **None** (this is the baseline) |
| Monitoring | 10% in-domain CelebA validation slice |

## Training Outcome

In-domain validation accuracy reached **0.995** within a couple of epochs.

This is **not** the headline result. The validation slice is held out of CelebA —
same domain as training — so high accuracy only confirms the model learned
CelebA. It says nothing about generalization. The real test is cross-dataset.

## Cross-Dataset Results (threshold = 0.5)

| Test domain | APCER | BPCER | ACER | EER | AUC | n (live/spoof) |
|-------------|-------|-------|------|-----|-----|----------------|
| OULU-NPU | 71.43% | 0.87% | 36.15% | 24.51% | **0.833** | 343 / 1358 |
| Kaggle | 4.44% | 64.81% | 34.63% | 29.63% | **0.816** | 54 / 135 |

## Interpretation

### 1. The model genuinely generalizes
AUC ≈ 0.82–0.83 on *unseen* datasets is well above random (0.5). The model
learned real liveness cues that transfer across domains — not just CelebA-specific
shortcuts. This is the core positive result.

### 2. The high ACER is a threshold artifact, not a discrimination failure
The error pattern is **inverted** between the two domains:

- **OULU:** biased toward *live* → misses attacks (APCER 71%, BPCER ~1%).
- **Kaggle:** biased toward *spoof* → rejects real faces (BPCER 65%, APCER ~4%).

Same model, same 0.5 threshold, opposite failures. This is a classic **domain-shift**
signature: each domain's score distribution sits in a different range, so a single
fixed threshold is mis-placed for both.

The threshold-independent metrics confirm it: **EER (24–30%) is much better than the
0.5-threshold ACER (~35%)**. The ranking of live vs spoof is solid; only the decision
cutoff is wrong. This is why AUC looks healthy while fixed-threshold ACER looks poor.

### 3. Caveat
The Kaggle test set is small (54 live / 135 spoof), so its metrics are noisier than
OULU's and its error analysis should not be over-generalized.

## Figures

| File | Shows |
|------|-------|
| `results/roc.png` | ROC per domain — threshold-free discrimination (AUC) |
| `results/score_dist.png` | Per-domain score histograms — visualizes the score shift driving the inverted errors |
| `results/error_rates.png` | APCER/BPCER/ACER bars — the inverted asymmetry |
| `results/errors/*_false_accept.png` | Spoofs wrongly accepted (security failures) |
| `results/errors/*_false_reject.png` | Live faces wrongly rejected (usability failures) |

## Conclusions & Implications for Next Experiments

- A working, honest baseline is established: **cross-dataset AUC ≈ 0.83**.
- The main weakness is **threshold calibration**, not the model's ability to
  separate live from spoof → the immediate next step is reporting at the EER
  threshold rather than a fixed 0.5.
- This baseline is the reference the **SPSC augmentation ablation (Experiment 2)**
  will be compared against: the question is whether simulating physical attack
  artifacts during training narrows the cross-dataset gap (raises AUC / lowers EER).

---

*Reproduce:* `python src/training/train.py` then `python src/evaluation/evaluate.py`
