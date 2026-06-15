"""
FAS Dataset
===========
Loads cropped face images for the face anti-spoofing pipeline.
 
Folder layout it expects (produced by detect_and_crop.py):
    processed_cropped/
      train/{live,spoof}/*.jpg      <- OULU + AxonData
      test/{live,spoof}/*.jpg       <- Kaggle (cross-domain)
 
The train/test split is ALREADY baked into the folders, so this class
does NO splitting. It just reads whichever split you ask for.
 
Label convention:  live = 0,  spoof = 1   (consistent across the project)
 
Transforms:
  - TEST  : resize + normalize ONLY. Never augmented.
  - TRAIN : resize + normalize, with an optional `augment` transform slot.
            Augmentation (SPSC) is OFF by default so the first baseline is
            clean. It drops in later as a swappable callable -- see
            build_datasets(augment=...). When added, it should apply to
            LIVE images only (that decision lives in the augment callable,
            not here), matching the SPSC live-only design.
 
This file is deliberately augmentation-agnostic: the model/training code
decides whether to pass an augment transform, so we can A/B baseline vs
SPSC vs other strategies with a one-line change.
"""


from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ImageNet stats -- required because the backbone will be pretrained
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.299, 0.224, 0.225]

LIVE, SPOOF = 0, 1
CLASS_DIRS = {"live": LIVE, "spoof": SPOOF}


class FASDataset(Dataset):
    """
    Args
    ----
    root : str | Path
        Path to processed_cropped (the folder containing train/ and test/).
    split : str
        'train' or 'test'.
    img_size : int
        Output square size (default 224, matches the crop step + ImageNet).
    augment : callable | None
        Optional augmentation applied to a PIL image BEFORE the tensor
        conversion. Off by default. When provided it is applied only during
        the 'train' split. The callable itself is responsible for any
        live-only logic (it receives the label so it can decide).
    """
    
    def __init__(self, root, split="train", img_size=224, augment=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment if split == "train" else None

        split_dir = self.root / split
        if not split_dir.exists():
          raise FileNotFoundError(f"Split folder not found: {split_dir}")

        # Gather (path, label) pairs from live / and spoof /
        self.samples = []
        for cls_name, label in CLASS_DIRS.items():
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((img_path, label))
                
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {split_dir}")
        
        
        # Deterministic tensor pipeline (applied to very image, both splits)
        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    
    
    # Stats helper (useful for weighted loss / sanity check) --
    def class_count(self):
        live = sum(1 for _, y in self.samples if y == LIVE)
        spoof = sum(1 for _, y in self.samples if y == SPOOF)
        return {"live": live, "spoof": spoof}
    
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Optional augmentation (train only). The callable may use the label
        # to apply effects to live images only and / or flip the label
        if self.augment is not None:
            result = self.augment(img, label)
            # support both
            if isinstance(result, tuple):
                img, label = result
            else:
                img = result
        
        tensor = self.to_tensor(img)
        return tensor, label


def build_dataset(root, img_size=224, augment=None):
    """Convenience: build train + test datasets in one call."""
    train_ds = FASDataset(root, "train", img_size, augment=augment)
    test_ds = FASDataset(root, "test", img_size, augment=augment)
    return train_ds, test_ds


def build_dataloader(root, batch_size=32, img_size=224, augment=None, num_workers=2):
    """Convenience: build train + test datasets in one call."""
    train_ds, test_ds = build_dataset(root, img_size, augment)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ── quick self-test ──
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed_cropped"
 
    for split in ["train", "test"]:
        try:
            ds = FASDataset(root, split)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[{split}] {e}")
            continue
        counts = ds.class_count()
        x, y = ds[0]
        print(f"[{split}] {len(ds)} images  "
            f"live={counts['live']} spoof={counts['spoof']}  "
            f"| sample tensor {tuple(x.shape)} label={y}")