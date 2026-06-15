"""
Diagnostic: class balance + source breakdown of the CROPPED train set.
Run after cropping to confirm the rebalance worked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import Counter
import config as C

def source_of(name):
    for p in ("axon_", "oulu_", "kaggle_"):
        if name.startswith(p): return p.rstrip("_")
    return "other"

def main():
    for split in ["train", "test"]:
        print(f"\n===== {split} =====")
        for cls in ["live", "spoof"]:
            d = C.CROPPED/split/cls
            if not d.exists(): print(f"[missing] {d}"); continue
            c = Counter(source_of(f.name) for f in d.glob("*.jpg"))
            tot = sum(c.values())
            print(f"{cls}: {tot}")
            for src, n in c.most_common():
                print(f"  {src:7s}: {n:6d} ({100*n/tot:.1f}%)" if tot else "")
        live = len(list((C.CROPPED/split/'live').glob('*.jpg')))
        spoof = len(list((C.CROPPED/split/'spoof').glob('*.jpg')))
        if live: print(f"ratio spoof:live = {spoof/live:.1f} : 1")

if __name__ == "__main__":
    main()