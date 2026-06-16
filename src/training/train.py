"""
Training loop — baseline (no augmentation yet)
==============================================
Trains the ResNet50 FAS model on CelebA (the 'train' split) and saves the
best checkpoint. Evaluation on OULU/Kaggle is a SEPARATE step (next module)
so training stays clean and reusable.

Monitoring: we carve a small VALIDATION slice out of the CelebA train set
(in-domain) purely to watch for overfitting during training. This val set
is NOT OULU/Kaggle — those stay untouched for the real cross-dataset eval.

Run in Colab:
    !python src/training/train.py
"""
import sys
from pathlib import Path

# make sibling packages importable regardless of CWD
REPO = Path(__file__).resolve().parents[1]          # .../src
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset.fas_dataset import FASDataset, TRAIN_SPLIT
from models.fas_model import build_model

# ── CONFIG ──────────────────────────────────────────────
DATA_ROOT  = "/content/Anti-Spoofing-/data/processed_cropped"
CKPT_PATH  = "/content/Anti-Spoofing-/checkpoints/resnet50_baseline.pth"
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-4
VAL_FRAC   = 0.1          # in-domain val slice for monitoring only
NUM_WORKERS = 2
SEED       = 42
# ────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_loaders():
    full = FASDataset(DATA_ROOT, TRAIN_SPLIT, augment=None)
    n_val = int(len(full) * VAL_FRAC)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"train={n_train}  val={n_val}  (in-domain val for monitoring)")
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


def main():
    torch.manual_seed(SEED)
    train_loader, val_loader = make_loaders()

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()          # train set is balanced
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
        print(f"epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  saved best -> {CKPT_PATH}  (val_acc={best_val:.4f})")

    print(f"\nDone. best in-domain val_acc={best_val:.4f}")
    print(f"checkpoint: {CKPT_PATH}")
    print("Next: run cross-dataset evaluation (OULU/Kaggle) with this checkpoint.")


if __name__ == "__main__":
    main()