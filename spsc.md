# Experiment 2 — SPSC augmentation ablation

**Status:** complete · **Result:** negative (augmentation did not improve cross-domain performance)

## Hypothesis

The baseline generalizes poorly across datasets (in-domain AUC 0.995 → cross-domain
~0.82). If the model is starved of spoof diversity, synthesizing additional spoof
examples should help.

SPSC (Simulated Physical Spoofing Clues) stamps synthetic artifacts — moiré, glare,
print colour-shift — onto **live** faces and flips the label to spoof, on the logic
that such artifacts cannot occur on a genuine capture. Free extra spoof data.

## Setup

Identical to the baseline in every respect except the augmentation: ResNet-50
(ImageNet1K_V2), balanced CelebA-Spoof subset (~5k images), same schedule, same
seed. Cross-tested on OULU-NPU and a Kaggle FAS set, neither seen in training.

## Results

| Model | OULU AUC | OULU EER% | Kaggle AUC | Kaggle EER% |
|---|---|---|---|---|
| baseline | **0.8329** | **24.72** | **0.8156** | 29.63 |
| + SPSC | 0.8066 | 26.63 | 0.7917 | 28.33 |

SPSC is worse on AUC in both target domains (−2.6 and −2.4 points).

## Interpretation

The augmentation taught a spoof cue that lives in the *simulator*, not in the target
domains. Real screen-replay moiré and real print texture have different frequency and
colour statistics than a parametric simulation, so the model partly learned
"spoof = carries the artifact my generator produces." At test time OULU and Kaggle
spoofs carry their own real artifacts, that learned cue does not fire, and the model
has been nudged toward a synthetic domain neither test set occupies.

Two factors compound this. The baseline already generalized off genuinely transferable
cues, so SPSC's strong, specific prior **competed with** those cues rather than adding
to them. And with only ~5k training images, there is not enough real spoof diversity to
dilute a synthetic shortcut once the model latches onto it.

## Honest caveats

- Single seed, and the Kaggle test set is 189 images. A 2–3 point AUC swing is close to
  run-to-run noise.
- The defensible claim is therefore **"SPSC did not help here, and plausibly hurt
  slightly, in this small-data single-source regime"** — not "SPSC hurts."
- Establishing a real effect would need multi-seed runs.

## Takeaway

The *instinct* — synthesize the spoof cues you lack — is sound; the execution failed on
the simulation-to-real gap. This motivated Experiment 3: rather than guess at why the
model generalizes badly, look at where it is actually attending.