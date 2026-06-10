"""
Step 4: Face Detection + Crop Preprocessing
Uses MTCNN (facenet-pytorch) to detect and crop faces.

Pipeline per image:
  1. Detect face with MTCNN
  2. If found     -> crop with 20px margin
  3. If not found -> center crop fallback
  4. Resize to 224x224
  5. Save to processed_cropped/

Input:  data/processed/{train,test}/{live,spoof}/*.jpg
Output: data/processed_cropped/{train,test}/{live,spoof}/*.jpg

Handles every split/class folder that exists, including the Kaggle
cross-domain test set (which lives in test/). Reports detection stats
PER split/class so you can spot a domain where MTCNN struggles
(e.g. Kaggle phone videos) before it skews evaluation.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN

# ── CONFIG ──────────────────────────────────────────────
INPUT_ROOT  = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed"
OUTPUT_ROOT = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed_cropped"
MARGIN      = 20          # pixels to expand bounding box
OUT_SIZE    = 224         # final size (fits RTX 3070, matches ImageNet)
SPLITS      = ["train", "test"]
CLASSES     = ["live", "spoof"]
# ────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

mtcnn = MTCNN(keep_all=False, device=device, post_process=False)


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Fallback: square center crop then resize."""
    h, w = img.shape[:2]
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    cropped = img[top:top + side, left:left + side]
    return cv2.resize(cropped, (size, size))


def crop_with_margin(img: np.ndarray, box: np.ndarray, margin: int, size: int) -> np.ndarray:
    """Crop detected face box with margin, then resize."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return center_crop(img, size)
    return cv2.resize(face, (size, size))


def process_folder():
    # per (split, class) stats, plus a grand total
    per_group = {}
    grand = {"detected": 0, "fallback": 0, "total": 0}

    for split in SPLITS:
        for label_name in CLASSES:
            in_dir  = Path(INPUT_ROOT) / split / label_name
            out_dir = Path(OUTPUT_ROOT) / split / label_name

            if not in_dir.exists():
                print(f"\n[skip] {in_dir} does not exist")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            images = sorted(in_dir.glob("*.jpg"))
            print(f"\n{split}/{label_name}: {len(images)} images")

            s = {"detected": 0, "fallback": 0, "total": 0}

            for img_path in tqdm(images, desc=f"Cropping {split}/{label_name}"):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                boxes, _ = mtcnn.detect(Image.fromarray(img_rgb))

                if boxes is not None and len(boxes) > 0:
                    out = crop_with_margin(img_rgb, boxes[0], MARGIN, OUT_SIZE)
                    s["detected"] += 1
                else:
                    out = center_crop(img_rgb, OUT_SIZE)
                    s["fallback"] += 1
                s["total"] += 1

                out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / img_path.name), out_bgr)

            per_group[(split, label_name)] = s
            for k in grand:
                grand[k] += s[k]

    # ── per-group summary ──
    print("\n=== Detection rate per split/class ===")
    for (split, label_name), s in per_group.items():
        if s["total"]:
            rate = 100 * s["detected"] / s["total"]
            print(f"  {split:5s}/{label_name:5s}: "
                  f"{s['detected']:5d} det / {s['fallback']:4d} fallback "
                  f"({rate:5.1f}% detected)")

    print("\n=== Grand total ===")
    print(f"  Total processed : {grand['total']}")
    print(f"  Face detected   : {grand['detected']}")
    print(f"  Center fallback : {grand['fallback']}")
    if grand["total"]:
        rate = 100 * grand["detected"] / grand["total"]
        print(f"  Detection rate  : {rate:.1f}%")


if __name__ == "__main__":
    process_folder()