"""
Imbalance handling at the TRAINING level.

Even after capping Axon spoof frames, you'll have a residual imbalance
(~4-5 : 1 spoof:live). This computes inverse-frequency class weights to
pass into the loss so the rarer class (live) is weighted up.

Usage in training:
    from utils.class_weights import compute_class_weights
    import torch.nn as nn

    weights = compute_class_weights(CROPPED / "train")   # tensor([w_live, w_spoof])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

How it works: weight[c] = total / (n_classes * count[c]). The class with
fewer samples gets a larger weight, so each live mistake costs more and
gradients stop being dominated by spoof.
"""
from pathlib import Path
import torch

LIVE, SPOOF = 0, 1

def count_classes(train_dir):
    train_dir = Path(train_dir)
    n_live  = len(list((train_dir / "live").glob("*.jpg")))
    n_spoof = len(list((train_dir / "spoof").glob("*.jpg")))
    return n_live, n_spoof

def compute_class_weights(train_dir, n_classes=2):
    n_live, n_spoof = count_classes(train_dir)
    counts = [max(1, n_live), max(1, n_spoof)]   # avoid div-by-zero
    total = sum(counts)
    weights = [total / (n_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)

def make_sampler_weights(train_dir):
    """
    Alternative to weighted loss: per-sample weights for a
    WeightedRandomSampler (balances each batch by oversampling live).
    Returns a list aligned with ImageFolder-style ordering (live then spoof).
    """
    n_live, n_spoof = count_classes(train_dir)
    w_live  = 1.0 / max(1, n_live)
    w_spoof = 1.0 / max(1, n_spoof)
    return [w_live]*n_live + [w_spoof]*n_spoof

if __name__ == "__main__":
    import sys
    train_dir = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed_cropped\train"
    n_live, n_spoof = count_classes(train_dir)
    w = compute_class_weights(train_dir)
    print(f"live={n_live}  spoof={n_spoof}  ratio={n_spoof/max(1,n_live):.1f}:1")
    print(f"class weights -> live={w[0]:.3f}  spoof={w[1]:.3f}")