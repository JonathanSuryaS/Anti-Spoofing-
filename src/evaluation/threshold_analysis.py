"""
Threshold analysis
==================
The baseline reports ACER at a fixed 0.5 threshold, which is mis-calibrated
under domain shift (hence OULU's 71% APCER despite AUC 0.83). This script
reports the HONEST picture with two threshold strategies per test domain:

  1. EER-optimal (oracle)  — the best threshold FOR THAT DOMAIN. Shows the
     model's true discrimination ceiling. Standard in FAS papers, but it
     "peeks" at the test set, so it's a best-case, not a deployable, number.

  2. CelebA-calibrated     — threshold chosen on a held-out CelebA (in-domain)
     validation slice, then applied UNCHANGED to OULU/Kaggle. This simulates
     real deployment: you fix the threshold before seeing test data. The gap
     between (1) and (2) quantifies how much you lose to threshold
     miscalibration under domain shift.

Run:
    !cd /content/Anti-Spoofing- && python src/evaluation/threshold_analysis.py
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset.fas_dataset import FASDataset, TRAIN_SPLIT, TEST_SPLITS
from models.fas_model import build_model
from evaluation.metrics import apcer_bpcer_acer, compute_eer, compute_auc

DATA_ROOT = "/content/Anti-Spoofing-/data/processed_cropped"
CKPT_PATH = "/content/Anti-Spoofing-/checkpoints/resnet50_baseline.pth"
VAL_FRAC  = 0.1
SEED      = 42
SPOOF_IDX = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def scores_labels(model, ds):
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    ss, ll = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, SPOOF_IDX]
        ss.append(p.cpu().numpy()); ll.append(y.numpy())
    return np.concatenate(ss), np.concatenate(ll)


def best_acer_threshold(scores, labels):
    """Threshold that minimizes ACER on this set (used for CelebA calibration)."""
    best_t, best_acer = 0.5, 1.0
    for t in np.unique(scores):
        _, _, acer = apcer_bpcer_acer(scores, labels, t)
        if acer < best_acer:
            best_acer, best_t = acer, t
    return best_t


def row(name, scores, labels, t):
    apcer, bpcer, acer = apcer_bpcer_acer(scores, labels, t)
    return (f"  {name:24s} t={t:.3f} | "
            f"APCER {100*apcer:5.2f}%  BPCER {100*bpcer:5.2f}%  "
            f"ACER {100*acer:5.2f}%")


def main():
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # --- derive a CelebA-calibrated threshold from a held-out train slice ---
    full = FASDataset(DATA_ROOT, TRAIN_SPLIT, augment=None)
    n_val = int(len(full) * VAL_FRAC)
    g = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(full, [len(full) - n_val, n_val], generator=g)
    val_scores, val_labels = scores_labels(model, val_ds)
    celeba_t = best_acer_threshold(val_scores, val_labels)
    print(f"CelebA-calibrated threshold (from in-domain val): {celeba_t:.3f}\n")

    # --- per test domain: fixed 0.5, EER-optimal, CelebA-calibrated ---
    for split in TEST_SPLITS:
        try:
            ds = FASDataset(DATA_ROOT, split, augment=None)
        except Exception as e:
            print(f"[skip] {split}: {e}"); continue
        s, l = scores_labels(model, ds)
        auc = compute_auc(s, l)
        eer, eer_t = compute_eer(s, l)

        print(f"=== {split}  (AUC {auc:.4f}, EER {100*eer:.2f}%) ===")
        print(row("fixed 0.5", s, l, 0.5))
        print(row("EER-optimal (oracle)", s, l, eer_t))
        print(row("CelebA-calibrated", s, l, celeba_t))
        print()

    print("Reading guide:")
    print("  fixed 0.5         -> the naive baseline number")
    print("  EER-optimal       -> best case if threshold perfectly placed per domain")
    print("  CelebA-calibrated -> realistic deployment (threshold fixed on train domain)")
    print("  gap(EER vs CelebA) = performance lost purely to domain miscalibration")


if __name__ == "__main__":
    main()