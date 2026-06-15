"""
Step 1 (TRAIN): CelebA-Spoof -> processed/train/{live,spoof}
Balanced, broad subject sampling. This is the ONLY training data.

Diversity over volume: a few LIVE and a few SPOOF images from many
distinct subjects. ~CELEBA_MAX_SUBJECTS subjects * (live+spoof per subj).
Ignores *_BB.txt (globs *.jpg); MTCNN crops uniformly later.

To scale up for the 5090: raise CELEBA_MAX_SUBJECTS / per-subject caps
in config.py. Nothing else changes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import random, shutil
from tqdm import tqdm
import config as C


def find_subject_dirs(root: Path):
    """Any dir containing a 'live' subfolder is a subject dir."""
    return [p.parent for p in root.rglob("live") if p.is_dir()]


def take(img_dir: Path, k: int):
    imgs = sorted(img_dir.glob("*.jpg"))   # skips *_BB.txt
    random.shuffle(imgs)
    return imgs[:k]


def main():
    root = Path(C.CELEBA_ROOT)
    if not root.exists():
        print(f"ERROR: CELEBA_ROOT not found: {root}"); return

    live_out  = C.PROCESSED / C.TRAIN_DIR / "live"
    spoof_out = C.PROCESSED / C.TRAIN_DIR / "spoof"
    live_out.mkdir(parents=True, exist_ok=True)
    spoof_out.mkdir(parents=True, exist_ok=True)

    subjects = find_subject_dirs(root)
    print(f"Found {len(subjects)} subjects")
    if not subjects:
        print("Check CELEBA_ROOT — no subject folders found."); return

    random.seed(C.CELEBA_SEED)
    random.shuffle(subjects)
    subjects = subjects[:C.CELEBA_MAX_SUBJECTS]

    counts = {"live": 0, "spoof": 0}
    for subj in tqdm(subjects, desc="CelebA subjects"):
        for img in take(subj / "live", C.CELEBA_LIVE_PER_SUBJECT):
            shutil.copy2(str(img), str(live_out / f"celeba_{subj.name}_{img.name}"))
            counts["live"] += 1
        spoof_dir = subj / "spoof"
        if spoof_dir.exists():
            for img in take(spoof_dir, C.CELEBA_SPOOF_PER_SUBJECT):
                shutil.copy2(str(img), str(spoof_out / f"celeba_{subj.name}_{img.name}"))
                counts["spoof"] += 1

    print(f"\nCelebA -> train  live: {counts['live']}, spoof: {counts['spoof']}")


if __name__ == "__main__":
    main()