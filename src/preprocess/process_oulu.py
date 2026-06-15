"""
OULU -> processed/test_oulu/{live,spoof}   (UNSEEN test domain 1)
Model never trains on this. true->live, false->spoof.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shutil
from tqdm import tqdm
import config as C


def process():
    for cls, folder in [("live", C.OULU_TRUE), ("spoof", C.OULU_FALSE)]:
        images = sorted(Path(folder).glob("*.jpg"))
        out_dir = C.PROCESSED / C.TEST_OULU_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"OULU {cls}: {len(images)} images")
        for img in tqdm(images, desc=f"  {cls}"):
            shutil.copy2(str(img), str(out_dir / f"oulu_{img.name}"))
    print("OULU -> test_oulu done.")


if __name__ == "__main__":
    process()