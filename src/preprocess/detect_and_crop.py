"""
Step 4: Face Detection + Crop Preprocessing
Uses MTCNN (facenet-pytorch) to detect and crop faces.
 
Pipeline per image:
  1. Detect face with MTCNN
  2. If found    → crop with 20px margin
  3. If not found → center crop fallback
  4. Resize to 224x224
  5. Save to processed_cropped/
 
Input:  data/processed/{live,spoof}/*.jpg
Output: data/processed_cropped/{live,spoof}/*.jpg
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN

# CONFIG
INPUT_ROOT  = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed"
OUTPUT_ROOT = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed_cropped"
MARGIN      = 20          # pixels to expand bounding box
OUT_SIZE    = 224         # final image size (fits RTX 3070, matches ImageNet)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# keep_all = False -> return only the most preminent face
# post_process = False -> we handle cropping / resizing ourselved
mtcnn = MTCNN(keep_all = False, device=device, post_process = False)


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Fallback: square center crop then resize"""
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = img[top:top + side, left: left+side]
    return cv2.resize(cropped, (size, size))


def crop_with_margin(img: np.ndarray, box: np.ndarray, margin: int, size: int)->np.ndarray:
    """Crop detected face box with margin, then resize."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    
    # Expand box by margin, clamp to image bounds
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return center_crop(img, size)
    return cv2.resize(face, (size, size))


def process_folder():
    stats = {"detected": 0, "fallback": 0, "total": 0}
    
    for label_name in ["live", "spoof"]:
        in_dir = Path(INPUT_ROOT) / label_name
        out_dir = Path(OUTPUT_ROOT) / label_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        images = sorted(in_dir.glob("*.jpg"))
        print(f"\n{label_name}: {len(images)} images")
        
        for img_path in tqdm(images, desc=f"Cropping {label_name}"):
            # Read as RGB (MTCNN expects RGB)
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect Face
            boxes, _ = mtcnn.detect(Image.fromarray(img_rgb))
            
            if boxes is not None and len(boxes) > 0:
                out = crop_with_margin(img_rgb, boxes[0], MARGIN, OUT_SIZE)
                stats["detected"] += 1
            else:
                out = center_crop(img_rgb, OUT_SIZE)
                stats["fallback"] += 1

            stats["total"] += 1
            
            # Save back as BGR for cv2
            out_bgr = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(out_dir / img_path.name), out_bgr)
            
    print("\n=== Preprocessing Summary ===")
    print(f"  Total processed : {stats['total']}")
    print(f"  Face detected   : {stats['detected']}")
    print(f"  Center fallback : {stats['fallback']}")
    if stats["total"] > 0:
        rate = 100 * stats["detected"] / stats["total"]
        print(f"  Detection rate  : {rate:.1f}%")

if __name__ == "__main__":
    process_folder()