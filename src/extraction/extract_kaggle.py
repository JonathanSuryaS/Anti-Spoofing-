"""
Step 3: Process the Kaggle anti-spoofing set into the TEST split
================================================================
This set is used ONLY for cross-domain evaluation. The model never
trains on it. Labels come from the folder names (the CSV is just a
subject-alignment index and is not needed here):

  live_selfie        -> live   (images)
  live_video         -> live   (video)
  cut-out printouts  -> spoof  (video)
  printouts          -> spoof  (video)
  replay             -> spoof  (video)

Because it's a TEST set we sample only a FEW frames per video (coverage,
not volume). Handles .mp4/.mov/.jpg/.png, sanitizes messy filenames
(spaces, non-ASCII device names), and writes to processed/test/.
"""

import re
import shutil
import unicodedata
from pathlib import Path

import cv2
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────
KAGGLE_ROOT = r"C:\Users\user\Downloads\kaggle"  
OUTPUT_ROOT = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed"
FRAMES_PER_VIDEO = 5          # sparse sampling for a test set
SEED_NOTE = "test set: sampled evenly across each video"
# ────────────────────────────────────────────────────────

# folder name -> class label
LABEL_MAP = {
    "live_selfie":       "live",
    "live_video":        "live",
    "cut-out printouts": "spoof",
    "cut-out_printouts": "spoof",   # in case of underscore variant
    "printouts":         "spoof",
    "replay":            "spoof",
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def sanitize(name: str) -> str:
    """Make a filesystem-safe ASCII filename fragment."""
    # drop non-ASCII (handles the corrupted Cyrillic device names)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("_") or "file"


def sample_frame_indices(total: int, k: int):
    """Pick k frame indices spread evenly across the video."""
    if total <= 0:
        return []
    if total <= k:
        return list(range(total))
    step = total / float(k)
    return [int(i * step) for i in range(k)]


def extract_video(src: Path, out_dir: Path, prefix: str, k: int) -> int:
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(sample_frame_indices(total, k))

    saved, idx = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            cv2.imwrite(str(out_dir / f"{prefix}_f{saved:02d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def copy_image(src: Path, out_dir: Path, prefix: str) -> int:
    shutil.copy2(str(src), str(out_dir / f"{prefix}.jpg"))
    return 1


def main():
    root = Path(KAGGLE_ROOT)
    if not root.exists():
        print(f"ERROR: KAGGLE_ROOT not found: {root}")
        return

    counts = {"live": 0, "spoof": 0}

    for folder_name, label in LABEL_MAP.items():
        folder = root / folder_name
        if not folder.exists():
            continue

        out_dir = Path(OUTPUT_ROOT) / "test" / label
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in folder.rglob("*") if f.is_file()]
        print(f"\n{folder_name} -> {label}: {len(files)} files")

        for f in tqdm(files, desc=f"  {folder_name}"):
            ext = f.suffix.lower()
            prefix = f"kaggle_{sanitize(folder_name)}_{sanitize(f.stem)}"

            if ext in VIDEO_EXTS:
                counts[label] += extract_video(f, out_dir, prefix, FRAMES_PER_VIDEO)
            elif ext in IMAGE_EXTS:
                counts[label] += copy_image(f, out_dir, prefix)
            # silently ignore other file types

    print("\n=== Kaggle test set ===")
    print(f"  live  frames: {counts['live']}")
    print(f"  spoof frames: {counts['spoof']}")
    print(f"  total       : {counts['live'] + counts['spoof']}")


if __name__ == "__main__":
    main()