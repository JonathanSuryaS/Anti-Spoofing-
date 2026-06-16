"""
FAS Dataset (multi-domain)
==========================
Loads cropped faces for the anti-spoofing pipeline.

Layout (from detect_and_crop.py):
    processed_cropped/
      train/{live,spoof}/         <- CelebA  (the only training data)
      test_oulu/{live,spoof}/     <- OULU    (unseen test domain 1)
      test_kaggle/{live,spoof}/   <- Kaggle  (unseen test domain 2)

This class loads ONE split by name, so each test domain is a separate
Dataset -> separate DataLoader -> separate per-domain metrics.

Label convention:  live = 0,  spoof = 1
Normalization   :  ImageNet stats (backbone will be pretrained)

Augmentation: optional `augment` callable, applied to TRAIN only, before
the tensor conversion. OFF by default (baseline first). When added it may
return either `img` or `(img, new_label)`; the callable owns any live-only
or label-flip logic. Test splits can NEVER be augmented (forced off).
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LIVE, SPOOF = 0, 1
CLASS_DIRS = {"live": LIVE, "spoof": SPOOF}

TRAIN_SPLIT = "train"
TEST_SPLITS = ["test_oulu", "test_kaggle"]


class FASDataset(Dataset):
    def __init__(self, root, split="train", img_size=224, augment=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        # augmentation only ever applies to the train split
        self.augment = augment if split == TRAIN_SPLIT else None

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        self.samples = []
        for cls_name, label in CLASS_DIRS.items():
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((p, label))

        if not self.samples:
            raise RuntimeError(f"No images under {split_dir}")

        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def class_counts(self):
        live = sum(1 for _, y in self.samples if y == LIVE)
        spoof = sum(1 for _, y in self.samples if y == SPOOF)
        return {"live": live, "spoof": spoof}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.augment is not None:
            result = self.augment(img, label)
            if isinstance(result, tuple):
                img, label = result
            else:
                img = result

        return self.to_tensor(img), label


def build_dataloaders(root, batch_size=32, img_size=224, augment=None,
                      num_workers=2):
    """
    Returns:
        train_loader,
        test_loaders : dict {domain_name: DataLoader}  e.g.
                       {"test_oulu": ..., "test_kaggle": ...}
    """
    train_ds = FASDataset(root, TRAIN_SPLIT, img_size, augment=augment)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_loaders = {}
    for split in TEST_SPLITS:
        try:
            ds = FASDataset(root, split, img_size, augment=None)
        except (FileNotFoundError, RuntimeError):
            continue
        test_loaders[split] = DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers,
                                         pin_memory=True)
    return train_loader, test_loaders


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "/content/Anti-Spoofing-/data/processed_cropped"

    for split in [TRAIN_SPLIT] + TEST_SPLITS:
        try:
            ds = FASDataset(root, split)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[{split}] {e}")
            continue
        c = ds.class_counts()
        x, y = ds[0]
        print(f"[{split}] {len(ds)} imgs  live={c['live']} spoof={c['spoof']}  "
              f"| tensor {tuple(x.shape)} label={y}")