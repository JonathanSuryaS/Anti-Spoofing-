"""
Experiment 2 — SPSC augmentation ablation
==========================================
IDENTICAL to the baseline training in every way EXCEPT augmentation:
same backbone, epochs, lr, batch size, seed, and val split. This is what
makes it a valid ablation — any change in cross-dataset metrics is
attributable to SPSC, not to some other moving part.

Augmentation policy: SPSC applied to LIVE images only (paper-faithful).
SPSC simulates physical spoofing artifacts on genuine faces; applying it
to images that are already spoofs is incoherent, so live-only.

Saves to a SEPARATE checkpoint so the baseline is preserved for comparison.

Run:
    !cd /content/Anti-Spoofing- && python src/training/train_spsc.py
Then evaluate with the same scripts pointed at this checkpoint.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset.fas_dataset import FASDataset, TRAIN_SPLIT
from models.fas_model import build_model
from augmentation.spsc_augmentation import SPSC      # the enhanced SPSC class

# ── CONFIG — identical to baseline except CKPT + augmentation ──
DATA_ROOT  = "/content/Anti-Spoofing-/data/processed_cropped"
CKPT_PATH  = "/content/Anti-Spoofing-/checkpoints/resnet50_spsc.pth"   # separate!
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-4
VAL_FRAC   = 0.1
NUM_WORKERS = 2
SEED       = 42
SPSC_P     = 0.5      # global augmentation strength
LIVE_LABEL = 0
# ───────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"


class LiveOnlyAugment:
    """
    Adapter bridging SPSC to the Dataset's augment(img, label) interface.
    Applies SPSC to LIVE images only; spoof images pass through untouched.
    Labels are NOT flipped (paper-faithful baseline variant).
    """
    def __init__(self, p=0.5):
        self.spsc = SPSC(p=p)         # SPSC.__call__(img) -> img

    def __call__(self, img, label):
        if label == LIVE_LABEL:
            out = self.spsc(img)
            # SPSC may return img or (img, became_spoof); take the image
            img = out[0] if isinstance(out, tuple) else out
        return img, label


def make_loaders(augment):
    full = FASDataset(DATA_ROOT, TRAIN_SPLIT, augment=augment)
    n_val = int(len(full) * VAL_FRAC)
    g = torch.Generator().manual_seed(SEED)   # SAME split as baseline
    train_ds, val_ds = random_split(full, [len(full) - n_val, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"train={len(train_ds)} val={len(val_ds)}  (SPSC live-only, p={SPSC_P})")
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


def main():
    torch.manual_seed(SEED)
    augment = LiveOnlyAugment(p=SPSC_P)
    train_loader, val_loader = make_loaders(augment)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    Path(CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)
    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader)
        print(f"epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  saved best -> {CKPT_PATH}  (val_acc={best_val:.4f})")

    print(f"\nDone. best val_acc={best_val:.4f}")
    print(f"checkpoint: {CKPT_PATH}")
    print("Next: run threshold_analysis.py / evaluate.py pointed at this checkpoint,")
    print("      then compare cross-dataset EER/AUC against the baseline.")


if __name__ == "__main__":
    main()