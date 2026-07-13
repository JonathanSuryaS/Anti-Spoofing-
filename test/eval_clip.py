#!/usr/bin/env python3
"""
eval_clip.py — cross-domain evaluation of the frozen-CLIP model
===============================================================

Same metrics, same conventions as compare_checkpoints.py, so the numbers drop
straight into your existing results table.

Reads:  results/checkpoints/clip_head.pth
Tests:  data/processed_cropped/test_oulu   (unmasked — see note in train_clip.py)
        data/processed_cropped/test_kaggle

Compare against your recorded baseline:
        OULU   AUC 0.8329   EER 24.72%   BPCER@APCER1% 83.97%
        Kaggle AUC 0.8156   EER 29.63%   BPCER@APCER1% 77.78%
"""

import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import open_clip
except ImportError:
    raise SystemExit("pip install open_clip_torch")


DATA_ROOT = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed_cropped"
CKPT      = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\results\checkpoints\clip_head.pth"
DOMAINS   = ["test_oulu", "test_kaggle"]

BATCH_SIZE   = 64
TARGET_APCER = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

BASELINE = {   # for side-by-side printing
    "test_oulu":   dict(auc=0.8329, eer=24.72, sec=83.97),
    "test_kaggle": dict(auc=0.8156, eer=29.63, sec=77.78),
}


class DomainDataset(Dataset):
    def __init__(self, domain_dir, preprocess):
        self.samples = []
        for label, sub in [(0, "live"), (1, "spoof")]:
            d = os.path.join(domain_dir, sub)
            files = []
            for ext in IMG_EXTS:
                files += glob.glob(os.path.join(d, ext))
            self.samples += [(f, label) for f in sorted(files)]
        if not self.samples:
            raise FileNotFoundError(f"No images under {domain_dir}")
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.preprocess(Image.open(path).convert("RGB")), label


def compute_eer(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def bpcer_at_apcer(scores, labels, target=TARGET_APCER):
    """Largest threshold whose APCER stays within budget; report the BPCER there."""
    spoof = np.sort(scores[labels == 1])
    if spoof.size == 0:
        return float("nan"), float("nan")
    best_tau = None
    for tau in np.unique(scores):
        if float(np.mean(spoof < tau)) <= target:
            best_tau = float(tau)
        else:
            break
    if best_tau is None:
        return float("nan"), float("nan")
    return float(np.mean(scores[labels == 0] >= best_tau)), best_tau


@torch.no_grad()
def main():
    print(f"Device: {DEVICE}\n")

    ck = torch.load(CKPT, map_location="cpu")
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        ck["clip_model"], pretrained=ck["clip_pretrained"]
    )
    backbone = backbone.to(DEVICE).eval()

    head = nn.Linear(ck["feat_dim"], 2).to(DEVICE)
    head.load_state_dict(ck["head"])
    head.eval()
    print(f"Loaded frozen {ck['clip_model']} + trained head\n")

    for domain in DOMAINS:
        ddir = os.path.join(DATA_ROOT, domain)
        if not os.path.isdir(ddir):
            print(f"[skip] {ddir} not found")
            continue

        ds = DomainDataset(ddir, preprocess)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        scores, labels = [], []
        for x, y in dl:
            f = F.normalize(backbone.encode_image(x.to(DEVICE)).float(), dim=-1)
            p = F.softmax(head(f), dim=1)[:, 1]
            scores.append(p.cpu().numpy())
            labels.append(np.asarray(y))
        s = np.concatenate(scores)
        yy = np.concatenate(labels)

        auc = roc_auc_score(yy, s)
        eer, _ = compute_eer(s, yy)
        sec, _ = bpcer_at_apcer(s, yy)

        b = BASELINE[domain]
        d_auc = auc - b["auc"]
        arrow = "UP  " if d_auc > 0 else "DOWN"

        print(f"=== {domain}  (live={int((yy==0).sum())}, spoof={int((yy==1).sum())}) ===")
        print(f"  baseline   AUC {b['auc']:.4f}   EER {b['eer']:5.2f}%   BPCER@APCER1% {b['sec']:5.2f}%")
        print(f"  CLIP       AUC {auc:.4f}   EER {eer*100:5.2f}%   BPCER@APCER1% {sec*100:5.2f}%")
        print(f"  --> AUC {arrow} {abs(d_auc):.4f}\n")

    print("Read the AUC delta. A rise means the ImageNet initialization was")
    print("carrying the domain bias — hypothesis confirmed. Flat or down means")
    print("the problem is deeper than the representation, and the next lever is")
    print("multi-source training (SSDG), not a different backbone.")


if __name__ == "__main__":
    main()