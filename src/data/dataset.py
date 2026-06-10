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
from torchvision import transform


# ImageNet stats -- required because the backbone will be pretrained
