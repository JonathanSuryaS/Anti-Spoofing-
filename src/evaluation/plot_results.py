"""
Visualize cross-dataset evaluation results.
Produces three figures per the project's story:
  1. ROC curve per domain          -> threshold-free discrimination (AUC)
  2. Score distributions per domain -> shows the domain-shift / threshold issue
  3. APCER/BPCER/ACER bar chart     -> makes the inverted asymmetry obvious

Run in Colab:
    !cd /content/Anti-Spoofing- && python src/evaluation/plot_results.py
Figures save to /content/Anti-Spoofing-/results/
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset.fas_dataset import FASDataset, TEST_SPLITS
from models.fas_model import build_model
from evaluation.metrics import summary, apcer_bpcer_acer, compute_auc

DATA_ROOT = "/content/Anti-Spoofing-/data/processed_cropped"
CKPT_PATH = "/content/Anti-Spoofing-/checkpoints/resnet50_baseline.pth"
OUT_DIR   = Path("/content/Anti-Spoofing-/results")
SPOOF_IDX = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def get_scores(model, loader):
    ss, ll = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, SPOOF_IDX]
        ss.append(p.cpu().numpy()); ll.append(y.numpy())
    return np.concatenate(ss), np.concatenate(ll)


def roc_points(scores, labels):
    """Return (fpr, tpr) sweeping threshold. tpr=attack detection, fpr=live misrejected."""
    ts = np.linspace(0, 1, 200)
    tpr, fpr = [], []
    pos = labels == 1; neg = labels == 0
    for t in ts:
        pred = scores >= t
        tpr.append(pred[pos].mean() if pos.sum() else 0)
        fpr.append(pred[neg].mean() if neg.sum() else 0)
    return np.array(fpr), np.array(tpr)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    data = {}
    for split in TEST_SPLITS:
        try:
            ds = FASDataset(DATA_ROOT, split, augment=None)
        except Exception as e:
            print(f"[skip] {split}: {e}"); continue
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        data[split] = get_scores(model, loader)

    # ── Figure 1: ROC per domain ──
    plt.figure(figsize=(6, 6))
    for split, (s, l) in data.items():
        fpr, tpr = roc_points(s, l)
        auc = compute_auc(s, l)
        plt.plot(fpr, tpr, label=f"{split} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    plt.xlabel("FPR (live wrongly flagged)")
    plt.ylabel("TPR (attacks detected)")
    plt.title("Cross-dataset ROC")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / "roc.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── Figure 2: score distributions ──
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1: axes = [axes]
    for ax, (split, (s, l)) in zip(axes, data.items()):
        ax.hist(s[l == 0], bins=30, alpha=0.6, label="live", color="green", density=True)
        ax.hist(s[l == 1], bins=30, alpha=0.6, label="spoof", color="red", density=True)
        ax.axvline(0.5, color="black", linestyle="--", label="t=0.5")
        ax.set_title(f"{split} score distribution")
        ax.set_xlabel("spoof score"); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "score_dist.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── Figure 3: APCER/BPCER/ACER bars ──
    splits = list(data.keys())
    metrics = {"APCER": [], "BPCER": [], "ACER": []}
    for split in splits:
        a, b, c = apcer_bpcer_acer(*data[split], threshold=0.5)
        metrics["APCER"].append(100*a); metrics["BPCER"].append(100*b); metrics["ACER"].append(100*c)
    x = np.arange(len(splits)); w = 0.25
    plt.figure(figsize=(7, 5))
    for i, (name, vals) in enumerate(metrics.items()):
        plt.bar(x + (i-1)*w, vals, w, label=name)
    plt.xticks(x, splits); plt.ylabel("%")
    plt.title("Error rates per domain (threshold=0.5)")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.savefig(OUT_DIR / "error_rates.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved 3 figures to {OUT_DIR}/")
    print("  roc.png, score_dist.png, error_rates.png")


if __name__ == "__main__":
    main()
