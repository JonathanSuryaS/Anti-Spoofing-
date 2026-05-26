"""
src/data/dataset.py
Unified DataLoader for OULU-NPU, CelebA-Spoof, CASIA-FASD, Replay-Attack
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.4),
        # Critical for FAS: simulate print/replay artifacts
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# Base Dataset
# ─────────────────────────────────────────────

class FASDataset(Dataset):
    """
    Generic face anti-spoofing dataset.
    Expects a CSV with columns: [image_path, label]
    label: 1 = real (live), 0 = spoof (fake)
    """

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        assert "image_path" in self.df.columns, "CSV must have 'image_path' column"
        assert "label" in self.df.columns, "CSV must have 'label' column"

        print(f"Loaded {len(self.df)} samples | "
              f"Real: {(self.df.label == 1).sum()} | "
              f"Spoof: {(self.df.label == 0).sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label


# ─────────────────────────────────────────────
# DataModule (PyTorch Lightning)
# ─────────────────────────────────────────────

class FASDataModule:
    """
    Handles train/val/test splits for any FAS dataset.
    Usage:
        dm = FASDataModule(cfg)
        train_loader = dm.train_dataloader()
        val_loader   = dm.val_dataloader()
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = Path(cfg.data.data_dir)
        self.image_size = cfg.data.image_size
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    def train_dataloader(self):
        dataset = FASDataset(
            csv_path=self.data_dir / "train.csv",
            transform=get_train_transforms(self.image_size)
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        dataset = FASDataset(
            csv_path=self.data_dir / "val.csv",
            transform=get_val_transforms(self.image_size)
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        dataset = FASDataset(
            csv_path=self.data_dir / "test.csv",
            transform=get_val_transforms(self.image_size)
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
