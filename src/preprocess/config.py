"""
Shared configuration — SINGLE source of truth for all preprocessing.

PLAN: CelebA-Spoof trains; OULU + Kaggle are unseen cross-dataset TEST
domains, each in its own folder so we can report per-domain metrics.

Layout produced:
    processed/
      train/{live,spoof}/        <- CelebA (balanced subset)
      test_oulu/{live,spoof}/    <- OULU
      test_kaggle/{live,spoof}/  <- Kaggle
    processed_cropped/  (same structure, after detect_and_crop.py)

EDIT THE RAW-DATA PATHS below to match your machine. Leave outputs as-is.
"""
from pathlib import Path

# ── RAW DATA (edit these) ───────────────────────────────
CELEBA_ROOT = Path(r"C:\Users\user\Downloads\CelebA_Spoof")
OULU_TRUE   = Path(r"C:\Users\user\Downloads\archive\Oulu-NPU\true")
OULU_FALSE  = Path(r"C:\Users\user\Downloads\archive\Oulu-NPU\false")
KAGGLE_ROOT = Path(r"C:\Users\user\Downloads\kaggle")

# ── OUTPUT (leave as-is) ────────────────────────────────
PROCESSED = Path(r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed")
CROPPED   = Path(r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed_cropped")

# split folder names
TRAIN_DIR       = "train"
TEST_OULU_DIR   = "test_oulu"
TEST_KAGGLE_DIR = "test_kaggle"

# ── CelebA training subset (balanced, broad) ────────────
# Debug subset target: ~2k images. Flip these up for the 5090 run.
CELEBA_LIVE_PER_SUBJECT  = 5
CELEBA_SPOOF_PER_SUBJECT = 5
CELEBA_MAX_SUBJECTS      = 500     # ~500 subj * (2+2) = ~2000 imgs
CELEBA_SEED              = 42

# ── Kaggle test sampling ────────────────────────────────
KAGGLE_FRAMES_PER_VIDEO = 5
KAGGLE_LABEL_MAP = {
    "live_selfie": "live", "live_video": "live",
    "cut-out printouts": "spoof", "cut-out_printouts": "spoof",
    "printouts": "spoof", "replay": "spoof",
}

# ── crop settings ───────────────────────────────────────
CROP_SIZE = 224
CROP_MARGIN = 20

# all splits the cropper should walk
ALL_SPLITS = [TRAIN_DIR, TEST_OULU_DIR, TEST_KAGGLE_DIR]