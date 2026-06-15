"""
Kaggle -> processed/test_kaggle/{live,spoof}  (UNSEEN test domain 2)
Labels from folder names. Sparse sampling. Handles mp4/mov/jpg, sanitizes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re, shutil, unicodedata
import cv2
from tqdm import tqdm
import config as C

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def sanitize(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "file"

def sample_indices(total, k):
    if total <= 0: return set()
    if total <= k: return set(range(total))
    step = total / float(k)
    return {int(i*step) for i in range(k)}

def extract_video(src, out_dir, prefix, k):
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sample_indices(total, k)
    idx = saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx in wanted:
            cv2.imwrite(str(out_dir / f"{prefix}_f{saved:02d}.jpg"), frame); saved += 1
        idx += 1
    cap.release(); return saved

def main():
    root = Path(C.KAGGLE_ROOT)
    if not root.exists():
        print(f"ERROR: KAGGLE_ROOT not found: {root}"); return
    counts = {"live": 0, "spoof": 0}
    for folder_name, label in C.KAGGLE_LABEL_MAP.items():
        folder = root / folder_name
        if not folder.exists(): continue
        out_dir = C.PROCESSED / C.TEST_KAGGLE_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in folder.rglob("*") if f.is_file()]
        print(f"\n{folder_name} -> {label}: {len(files)} files")
        for f in tqdm(files, desc=f"  {folder_name}"):
            ext = f.suffix.lower()
            prefix = f"kaggle_{sanitize(folder_name)}_{sanitize(f.stem)}"
            if ext in VIDEO_EXTS:
                counts[label] += extract_video(f, out_dir, prefix, C.KAGGLE_FRAMES_PER_VIDEO)
            elif ext in IMAGE_EXTS:
                shutil.copy2(str(f), str(out_dir / f"{prefix}.jpg")); counts[label] += 1
    print(f"\nKaggle -> test_kaggle  live: {counts['live']}, spoof: {counts['spoof']}")

if __name__ == "__main__":
    main()