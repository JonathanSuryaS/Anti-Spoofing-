"""
Quick diagnostic: where does the train imbalance come from?
Reads the existing processed/train folders (no re-extraction needed).
Breaks counts down by source (axon_ vs oulu_) and class.
"""
from pathlib import Path
from collections import Counter

TRAIN_ROOT = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed\train"

def source_of(name: str) -> str:
    if name.startswith("axon_"):
        return "axon"
    if name.startswith("oulu_"):
        return "oulu"
    return "other"

def main():
    for cls in ["live", "spoof"]:
        d = Path(TRAIN_ROOT) / cls
        if not d.exists():
            print(f"[missing] {d}")
            continue
        counter = Counter()
        for f in d.glob("*.jpg"):
            counter[source_of(f.name)] += 1
        total = sum(counter.values())
        print(f"\n{cls}: {total} total")
        for src, n in counter.most_common():
            print(f"  {src:6s}: {n:6d}  ({100*n/total:.1f}%)")

if __name__ == "__main__":
    main()