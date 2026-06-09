"""
Step 2: Copy OULU-NPU images into processed/live and processed/spoof
Split into train/test happens later in the Dataset class
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# Config
OULU_TRUE = r"C:\Users\user\Downloads\archive\Oulu-NPU\true"
OULU_FALSE = r"C:\Users\user\Downloads\archive\Oulu-NPU\false"
OUTPUT_TRUE = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed"

def process_oulu():
    for label_name, folder_path in [("live", OULU_TRUE), ("spoof", OULU_FALSE)]:
        images = sorted(Path(folder_path).glob("*.jpg"))
        out_dir = Path(OUTPUT_TRUE) / label_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"OULU {label_name}: {len(images)} images found")
        if len(images) == 0:
            print(f"WARNING: No images found in {folder_path}")
            continue
        
        for img_path in tqdm(images, desc=f"Copying {label_name}"):
            dest = out_dir / f"oulu{img_path.name}"
            shutil.copy2(str(img_path), str(dest))
        
        print(f"-> {out_dir}")
    
    print("Done")
    

if __name__ == "__main__":
    process_oulu()