"""
Face detection + crop for ALL splits (train, test_oulu, test_kaggle).
MTCNN detect -> margin crop -> center-crop fallback -> resize -> save.
Reports detection rate per split/class.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import config as C

CLASSES = ["live", "spoof"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

def center_crop(img, size):
    h, w = img.shape[:2]; side = min(h, w)
    t, l = (h-side)//2, (w-side)//2
    return cv2.resize(img[t:t+side, l:l+side], (size, size))

def crop_margin(img, box, m, size):
    h, w = img.shape[:2]; x1,y1,x2,y2 = box.astype(int)
    x1,y1 = max(0,x1-m), max(0,y1-m)
    x2,y2 = min(w,x2+m), min(h,y2+m)
    face = img[y1:y2, x1:x2]
    return center_crop(img, size) if face.size == 0 else cv2.resize(face, (size, size))

def main():
    per, grand = {}, {"detected":0,"fallback":0,"total":0}
    for split in C.ALL_SPLITS:
        for cls in CLASSES:
            in_dir, out_dir = C.PROCESSED/split/cls, C.CROPPED/split/cls
            if not in_dir.exists():
                print(f"[skip] {in_dir} missing"); continue
            out_dir.mkdir(parents=True, exist_ok=True)
            imgs = sorted(in_dir.glob("*.jpg"))
            print(f"\n{split}/{cls}: {len(imgs)} images")
            s = {"detected":0,"fallback":0,"total":0}
            for p in tqdm(imgs, desc=f"{split}/{cls}"):
                bgr = cv2.imread(str(p))
                if bgr is None: continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(Image.fromarray(rgb))
                if boxes is not None and len(boxes) > 0:
                    out = crop_margin(rgb, boxes[0], C.CROP_MARGIN, C.CROP_SIZE); s["detected"]+=1
                else:
                    out = center_crop(rgb, C.CROP_SIZE); s["fallback"]+=1
                s["total"]+=1
                cv2.imwrite(str(out_dir/p.name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            per[(split,cls)] = s
            for k in grand: grand[k]+=s[k]
    print("\n=== Detection per split/class ===")
    for (split,cls), s in per.items():
        if s["total"]:
            print(f"  {split:12s}/{cls:5s}: {s['detected']:5d} det / {s['fallback']:4d} fb "
                  f"({100*s['detected']/s['total']:5.1f}%)")
    if grand["total"]:
        print(f"\nGrand total: {grand['total']}  "
              f"detected {100*grand['detected']/grand['total']:.1f}%")

if __name__ == "__main__":
    main()