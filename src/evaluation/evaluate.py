"""
Cross-dataset evaluation
========================
Loads the trained checkpoint and evaluates on EACH unseen test domain
(OULU, Kaggle) separately, reporting APCER/BPCER/ACER/EER/AUC per domain.

This is the project's headline result: how well a CelebA-trained model
generalizes to datasets it never saw. Expect these numbers to be WORSE
than in-domain CelebA val — that gap is the honest cost of domain shift,
and reporting it is the point.

Run in Colab:
    !cd /content/Anti-Spoofing- && python src/evaluation/evaluate.py
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]      # .../src
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from dataset.fas_dataset import FASDataset, TEST_SPLITS
from models.fas_model import build_model
from evaluation.metrics import summary, format_summary

# ── CONFIG ──────────────────────────────────────────────
DATA_ROOT = "/content/Anti-Spoofing-/data/processed_cropped"
CKPT_PATH = "/content/Anti-Spoofing-/checkpoints/resnet50_baseline.pth"
BATCH_SIZE = 64
THRESHOLD = 0.5         # decision threshold for APCER/BPCER/ACER
SPOOF_IDX = 1
# ────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def get_scores(model, loader):
    """Return (scores, labels): spoof-probability and true label per sample."""
    all_scores, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)[:, SPOOF_IDX]
        all_scores.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def main():
    from torch.utils.data import DataLoader

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"loaded checkpoint: {CKPT_PATH}\n")

    print("=== Cross-dataset evaluation (threshold = %.2f) ===" % THRESHOLD)
    results = {}
    for split in TEST_SPLITS:
        try:
            ds = FASDataset(DATA_ROOT, split, augment=None)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[skip] {split}: {e}")
            continue
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
        scores, labels = get_scores(model, loader)
        s = summary(scores, labels, threshold=THRESHOLD)
        results[split] = s
        print(format_summary(split, s))

    if len(results) > 1:
        mean_acer = np.mean([s["ACER"] for s in results.values()])
        print(f"\nmean cross-dataset ACER: {100*mean_acer:.2f}%")

    print("\nReminder: these are CROSS-DATASET numbers (train=CelebA, "
          "test=unseen). They are meant to be lower than in-domain val.")


if __name__ == "__main__":
    main()