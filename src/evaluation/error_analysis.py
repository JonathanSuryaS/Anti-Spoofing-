"""
Error analysis — look at the images the model gets WRONG.
========================================================
For each test domain, builds two contact sheets of the model's most
CONFIDENT mistakes (most diagnostic):

  false_accept : SPOOF predicted LIVE   (security failures -> APCER)
  false_reject : LIVE  predicted SPOOF  (usability failures -> BPCER)

"Most confident" = ranked by how far the score was on the wrong side, so
you see the model's worst, most certain errors first — the ones whose
shared pattern (attack type, lighting, crop quality) is the real lesson.

Run in Colab:
    !cd /content/Anti-Spoofing- && python src/evaluation/error_analysis.py
Saves grids to /content/Anti-Spoofing-/results/errors/
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from dataset.fas_dataset import FASDataset, TEST_SPLITS
from models.fas_model import build_model

DATA_ROOT = "/content/Anti-Spoofing-/data/processed_cropped"
CKPT_PATH = "/content/Anti-Spoofing-/checkpoints/resnet50_baseline.pth"
OUT_DIR   = Path("/content/Anti-Spoofing-/results/errors")
THRESHOLD = 0.5
SPOOF_IDX = 1
N_SHOW    = 12          # images per contact sheet
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def score_all(model, ds):
    """Return per-sample (path, true_label, spoof_score)."""
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    scores, labels = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, SPOOF_IDX]
        scores.append(p.cpu().numpy()); labels.append(y.numpy())
    scores = np.concatenate(scores); labels = np.concatenate(labels)
    paths = [p for p, _ in ds.samples]
    return paths, labels, scores


def contact_sheet(items, title, out_path):
    """items: list of (path, score). Save a grid."""
    if not items:
        print(f"  (none) {title}"); return
    items = items[:N_SHOW]
    cols = 4
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis("off")
    for ax, (path, score) in zip(axes, items):
        try:
            ax.imshow(Image.open(path).convert("RGB"))
        except Exception:
            continue
        ax.set_title(f"score={score:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    for split in TEST_SPLITS:
        try:
            ds = FASDataset(DATA_ROOT, split, augment=None)
        except Exception as e:
            print(f"[skip] {split}: {e}"); continue
        print(f"\n{split}:")
        paths, labels, scores = score_all(model, ds)

        # false accepts: true spoof (1), scored LOW (predicted live).
        # most confident = lowest score.
        fa = [(paths[i], scores[i]) for i in range(len(paths))
              if labels[i] == 1 and scores[i] < THRESHOLD]
        fa.sort(key=lambda t: t[1])                     # lowest score first

        # false rejects: true live (0), scored HIGH (predicted spoof).
        # most confident = highest score.
        fr = [(paths[i], scores[i]) for i in range(len(paths))
              if labels[i] == 0 and scores[i] >= THRESHOLD]
        fr.sort(key=lambda t: -t[1])                    # highest score first

        print(f"  false accepts (spoof->live): {len(fa)}")
        print(f"  false rejects (live->spoof): {len(fr)}")
        contact_sheet(fa, f"{split}: false ACCEPTS (spoof predicted live)",
                      OUT_DIR / f"{split}_false_accept.png")
        contact_sheet(fr, f"{split}: false REJECTS (live predicted spoof)",
                      OUT_DIR / f"{split}_false_reject.png")


if __name__ == "__main__":
    main()
