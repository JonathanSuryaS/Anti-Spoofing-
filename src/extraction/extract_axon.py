"""
Step 1: Extract frames from AxonData MP4 videos
Output goes to Anti-Spoofing-\data\processed
 
AxonData is train-only (too few subjects for a meaningful test split)
"""

import cv2
import os
from pathlib import Path
import shutil
from tqdm import tqdm

# CONFIG
AXON_ROOT = r"C:\Users\user\Downloads\AxonLab"
OUTPUT_ROOT = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed"
EVERY_N_FRAME = 3
LIVE_FOLDER = ["Selfies"]


def extract_axon_frames():
    live_out = Path(OUTPUT_ROOT) / "live"
    spoof_out = Path(OUTPUT_ROOT) / "spoof"

    live_out.mkdir(parents=True, exist_ok=True)
    spoof_out.mkdir(parents=True, exist_ok=True)

    mp4_files = list(Path(AXON_ROOT).rglob("*.mp4"))
    print(f"Found {len(mp4_files)} MP4 Files in AxonData")

    total_saved = 0

    for video_path in tqdm(mp4_files, desc="Extracting"):
        parts = video_path.relative_to(AXON_ROOT).parts
        top_folder = parts[0]
        is_live = top_folder in LIVE_FOLDER
        out_dir = live_out if is_live else spoof_out

        # Build unique filename prefix from path
        safe_name = "_".join(parts).replace(" ", "_").replace(".mp4", "")

        cap = cv2.VideoCapture(str(video_path))
        frame_idx, saved = 0, 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_idx % EVERY_N_FRAME == 0:
                fname = f"axon_{safe_name}_f{saved:04d}.jpg"
                cv2.imwrite(str(out_dir / fname), frame)
                saved += 1
            
            frame_idx += 1
        
        cap.release()
        total_saved += saved
    
    # Handles JPG files
    jpg_files = list(Path(AXON_ROOT).rglob("*.jpg"))
    print(f"Found {len(jpg_files)} JPG files")
    
    for img_path in tqdm(jpg_files, desc="Copying images"):
        parts      = img_path.relative_to(AXON_ROOT).parts
        top_folder = parts[0]
        is_live    = top_folder in LIVE_FOLDER
        out_dir    = live_out if is_live else spoof_out
        safe_name  = "_".join(parts).replace(" ", "_")
        dest       = out_dir / f"axon_{safe_name}"
        shutil.copy2(str(img_path), str(dest))
        total_saved += 1

    print(f"\nDone. {total_saved} frames from {len(mp4_files)} videos")
    print(f"  Live  → {live_out}")
    print(f"  Spoof → {spoof_out}")
 
 
if __name__ == "__main__":
    extract_axon_frames()
            