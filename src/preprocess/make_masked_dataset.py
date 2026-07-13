#!/usr/bin/env python3
"""
make_masked_dataset.py — background suppression ablation (step 1)
=================================================================

Creates a masked copy of every split: the face region is kept, everything
outside a fixed centred ellipse is zeroed out.

    data/processed_cropped/  ->  data/processed_masked/
        train/{live,spoof}          train/{live,spoof}
        val/{live,spoof}            val/{live,spoof}
        test_oulu/{live,spoof}      test_oulu/{live,spoof}
        test_kaggle/{live,spoof}    test_kaggle/{live,spoof}

Why a FIXED ellipse (and not per-image landmarks)?
--------------------------------------------------
The mask must be IDENTICAL for every image — same shape, same position, on
live and spoof, on CelebA and OULU and Kaggle. That makes the mask boundary
carry zero discriminative signal: the model cannot learn anything from the
artifact we are introducing, because the artifact is a constant. The ONLY
variable that changes between the baseline run and this run is whether
background pixels exist. That is what makes it a clean ablation.

A per-image landmark mask would hug each face more tightly, but its shape
would then vary with the face — reintroducing a confound.

The edge is feathered (Gaussian blur on the mask) so we don't hand the CNN a
razor-sharp high-frequency ring to latch onto.

After running this, RETRAIN from scratch on data/processed_masked/train, then
evaluate on data/processed_masked/test_oulu and test_kaggle. Train and test
must both be masked — mismatched preprocessing would invalidate the result.
"""

import os
import glob
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ------------------------------ CONFIG ------------------------------
SRC_ROOT = "data/processed_cropped"
DST_ROOT = "data/processed_masked"

SPLITS  = ["train", "val", "test_oulu", "test_kaggle"]
CLASSES = ["live", "spoof"]

# Ellipse geometry as a fraction of image size. The MTCNN crops centre the
# face, so a centred ellipse covering ~86% width / ~96% height keeps the face
# (incl. chin and forehead) and drops the corners, which is where the CAMs
# showed the model cheating.
ELLIPSE_W   = 0.86
ELLIPSE_H   = 0.96
FEATHER_PX  = 6       # Gaussian blur radius on the mask edge; 0 = hard cut

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
# --------------------------------------------------------------------


def build_mask(size):
    """Fixed centred ellipse, feathered. Identical for every image."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    ew, eh = int(w * ELLIPSE_W), int(h * ELLIPSE_H)
    x0, y0 = (w - ew) // 2, (h - eh) // 2
    draw.ellipse([x0, y0, x0 + ew, y0 + eh], fill=255)

    if FEATHER_PX > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(FEATHER_PX))
    return mask


def apply_mask(img, mask):
    """Keep face pixels, push background to black."""
    arr = np.asarray(img, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    return Image.fromarray((arr * m).astype(np.uint8))


def main():
    if not os.path.isdir(SRC_ROOT):
        raise SystemExit(f"[error] source not found: {SRC_ROOT}")

    if os.path.isdir(DST_ROOT):
        ans = input(f"{DST_ROOT} exists. Delete and rebuild? [y/N] ").strip().lower()
        if ans != "y":
            raise SystemExit("aborted")
        shutil.rmtree(DST_ROOT)

    mask_cache = {}
    total = 0

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = os.path.join(SRC_ROOT, split, cls)
            if not os.path.isdir(src_dir):
                print(f"[skip] {src_dir} not found")
                continue

            dst_dir = os.path.join(DST_ROOT, split, cls)
            os.makedirs(dst_dir, exist_ok=True)

            files = []
            for ext in IMG_EXTS:
                files += glob.glob(os.path.join(src_dir, ext))

            for fp in sorted(files):
                img = Image.open(fp).convert("RGB")

                # one mask per image size (all crops should be the same size,
                # but this is cheap insurance)
                if img.size not in mask_cache:
                    mask_cache[img.size] = build_mask(img.size)

                out = apply_mask(img, mask_cache[img.size])
                out.save(os.path.join(dst_dir, os.path.basename(fp)))
                total += 1

            print(f"  {split}/{cls:5s}  {len(files):5d} images -> {dst_dir}")

    print(f"\nDone. {total} masked images written to {DST_ROOT}/")
    print("\nNext:")
    print("  1. Retrain from scratch with the train dir pointed at "
          f"{DST_ROOT}/train")
    print("  2. Save as checkpoints/resnet50_masked.pth")
    print(f"  3. Evaluate on {DST_ROOT}/test_oulu and {DST_ROOT}/test_kaggle")
    print("     (masked train + masked test — do NOT mix with unmasked)")


if __name__ == "__main__":
    main()