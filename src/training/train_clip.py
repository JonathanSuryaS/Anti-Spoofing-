#!/usr/bin/env python3
"""
train_clip.py — Experiment 5: swap the backbone, change nothing else
====================================================================

Single-variable ablation. Everything matches the baseline except the frozen
feature extractor:

    baseline : ImageNet ResNet-50   (fine-tuned end-to-end)
    this     : frozen CLIP ViT-B/16 + linear head

The hypothesis: your cross-domain failure is inherited from the ImageNet
INITIALIZATION, not from the architecture, the loss, or the augmentation
(experiments 2-4 ruled those out). CLIP was trained to match images to captions
across 400M diverse web pairs, which forces it to ignore camera / lighting /
compression — exactly the nuisance variation destroying you cross-domain.

Why FROZEN
----------
Two reasons, and the second is the important one.
  1. You have ~5k images. Fine-tuning 86M ViT params on that overfits instantly.
  2. Freezing keeps the experiment CLEAN. If we fine-tuned, a null result would
     be ambiguous — did CLIP not help, or did we just destroy its pretrained
     invariances? Frozen means you are testing the REPRESENTATION, full stop.

Runs on an RTX 3070 comfortably: no gradients through the backbone, so memory is
small and each epoch is fast. Features are cached after the first epoch, making
subsequent epochs nearly instant.

Install:
    pip install open_clip_torch

Outputs:
    results/checkpoints/clip_head.pth
"""

import os
import glob
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score

try:
    import open_clip
except ImportError:
    raise SystemExit("pip install open_clip_torch")


# ------------------------------ CONFIG ------------------------------
# NOTE: unmasked crops. Experiment 4 showed masking destroys OULU (bezel cues
# are real signal), so we go back to the original crops for this test.
DATA_ROOT   = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed_cropped"
OUT_CKPT    = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\results\checkpoints\clip_head.pth"

TRAIN_SPLIT = "train"
VAL_SPLIT   = "val"

CLIP_MODEL  = "ViT-B-16-quickgelu"
CLIP_PRETRAIN = "openai"

EPOCHS      = 20          # cheap: only a linear head is training
BATCH_SIZE  = 64
LR          = 1e-3        # higher than baseline: training a head, not a backbone
WEIGHT_DECAY = 1e-4
SEED        = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
# --------------------------------------------------------------------


torch.manual_seed(SEED)
np.random.seed(SEED)


class FASDataset(Dataset):
    """<root>/<split>/{live,spoof}/*  ->  (tensor, label). live=0, spoof=1."""

    def __init__(self, root, split, preprocess):
        self.samples = []
        for label, sub in [(0, "live"), (1, "spoof")]:
            d = os.path.join(root, split, sub)
            files = []
            for ext in IMG_EXTS:
                files += glob.glob(os.path.join(d, ext))
            self.samples += [(f, label) for f in sorted(files)]
        if not self.samples:
            raise FileNotFoundError(f"No images under {root}/{split}/(live|spoof)")
        self.preprocess = preprocess   # CLIP's own transform — do not substitute

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), label


@torch.no_grad()
def extract_features(backbone, loader, desc):
    """
    Run the frozen backbone once and cache the embeddings.

    This is the whole reason the experiment is cheap: the backbone never needs a
    backward pass, so we compute each image's 512-d embedding ONCE and then train
    the head on those vectors for 20 epochs in seconds.
    """
    backbone.eval()
    feats, labels = [], []
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        f = backbone.encode_image(x.to(DEVICE))
        f = F.normalize(f.float(), dim=-1)     # CLIP embeddings are used L2-normalized
        feats.append(f.cpu())
        labels.append(y)
        if i % 10 == 0:
            print(f"  {desc}: batch {i}/{len(loader)}", end="\r")
    print(f"  {desc}: {len(loader)} batches in {time.time()-t0:.1f}s" + " " * 20)
    return torch.cat(feats), torch.cat(labels)


def main():
    print(f"Device: {DEVICE}")

    # --- frozen CLIP backbone ---
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN
    )
    backbone = backbone.to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad = False

    feat_dim = backbone.visual.output_dim
    print(f"CLIP {CLIP_MODEL}/{CLIP_PRETRAIN}, frozen. Feature dim = {feat_dim}\n")

    # --- no val/ directory exists, so split train/ internally (as the baseline did) ---
    from torch.utils.data import random_split

    full_ds = FASDataset(DATA_ROOT, TRAIN_SPLIT, preprocess)
    n_val   = int(0.2 * len(full_ds))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"train: {len(train_ds)} images | val: {len(val_ds)} images "
          f"(split 80/20 from {TRAIN_SPLIT}/, seed {SEED})\n")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Extracting CLIP features (one pass, then cached):")
    Xtr, ytr = extract_features(backbone, train_dl, "train")
    Xva, yva = extract_features(backbone, val_dl,   "val")
    print()

    # --- the only trainable part ---
    head = nn.Linear(feat_dim, 2).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    Xtr, ytr = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xva, yva = Xva.to(DEVICE), yva.to(DEVICE)
    n = Xtr.size(0)

    best_auc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = crit(head(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.size(0)

        head.eval()
        with torch.no_grad():
            probs = F.softmax(head(Xva), dim=1)[:, 1]
            acc = ((probs >= 0.5).long() == yva).float().mean().item()
            auc = roc_auc_score(yva.cpu().numpy(), probs.cpu().numpy())

        print(f"epoch {epoch:2d}/{EPOCHS}  train_loss={total/n:.4f}  "
              f"val_acc={acc:.4f}  val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save({
        "head": best_state,
        "clip_model": CLIP_MODEL,
        "clip_pretrained": CLIP_PRETRAIN,
        "feat_dim": feat_dim,
    }, OUT_CKPT)

    print(f"\nDone. best in-domain val AUC = {best_auc:.4f}")
    print(f"checkpoint: {OUT_CKPT}")
    print("\nIn-domain will look similar to the baseline (~0.99) — that is NOT the")
    print("result. The experiment is CROSS-DOMAIN: run eval_clip.py on OULU/Kaggle")
    print("and compare against baseline OULU 0.8329 / Kaggle 0.8156.")


if __name__ == "__main__":
    main()