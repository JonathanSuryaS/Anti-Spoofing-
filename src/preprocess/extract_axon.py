"""
Step 1: Axon -> processed/train  (REBALANCED)
LIVE  : dense (every Nth frame / copy all images)  -- scarce, keep lots
SPOOF : capped at SPOOF_FRAMES_PER_VIDEO per video -- abundant, limit it
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shutil
import cv2
from tqdm import tqdm
import config as C


def is_live(src: Path) -> bool:
    return src.relative_to(C.AXON_ROOT).parts[0] in C.LIVE_FOLDER

def out_dir_for(src: Path) -> Path:
    cls = "live" if is_live(src) else "spoof"
    d = C.PROCESSED / "train" / cls
    d.mkdir(parents=True, exist_ok=True)
    return d

def safe_name(src: Path) -> str:
    parts = src.relative_to(C.AXON_ROOT).parts
    return "axon_" + "_".join(parts).replace(" ", "_").rsplit(".", 1)[0]

def sample_indices(total, k):
    if total <= 0:
        return set()
    if total <= k:
        return set(range(total))
    step = total / float(k)
    return {int(i * step) for i in range(k)}

def extract_live_video(src):
    out_dir, name = out_dir_for(src), safe_name(src)
    cap = cv2.VideoCapture(str(src)); idx = saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx % C.LIVE_EVERY_N_FRAME == 0:
            cv2.imwrite(str(out_dir / f"{name}_f{saved:04d}.jpg"), frame); saved += 1
        idx += 1
    cap.release(); return saved

def extract_spoof_video(src):
    out_dir, name = out_dir_for(src), safe_name(src)
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sample_indices(total, C.SPOOF_FRAMES_PER_VIDEO)
    idx = saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx in wanted:
            cv2.imwrite(str(out_dir / f"{name}_f{saved:02d}.jpg"), frame); saved += 1
        idx += 1
    cap.release(); return saved

def copy_image(src):
    out_dir, name = out_dir_for(src), safe_name(src)
    shutil.copy2(str(src), str(out_dir / f"{name}.jpg")); return 1

def main():
    videos = list(Path(C.AXON_ROOT).rglob("*.mp4"))
    images = list(Path(C.AXON_ROOT).rglob("*.jpg"))
    counts = {"live": 0, "spoof": 0}
    for v in tqdm(videos, desc="Axon videos"):
        if is_live(v): counts["live"] += extract_live_video(v)
        else:          counts["spoof"] += extract_spoof_video(v)
    for im in tqdm(images, desc="Axon images"):
        counts["live" if is_live(im) else "spoof"] += copy_image(im)
    print(f"\nAxon -> train  live: {counts['live']}, spoof: {counts['spoof']}")

if __name__ == "__main__":
    main()