"""
Unzip all CelebA-Spoof zip parts into a single output folder.
Usage:
    python unzip_celeba.py

Edit ZIP_DIR and OUT_DIR below before running.
"""

import zipfile
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────
ZIP_DIR = r"C:\Users\user\Downloads"          # folder containing all the zips
OUT_DIR = r"C:\Users\user\Downloads\CelebA-Spoof"
# ─────────────────────────────────────────────────────────────────

def unzip_all():
    zip_dir = Path(ZIP_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(zip_dir.glob("CelebA-Sp*.zip"))

    if not zips:
        print(f"No CelebA-Spoof zip files found in: {zip_dir}")
        print("Make sure ZIP_DIR points to your Downloads folder.")
        return

    print(f"Found {len(zips)} zip files → extracting to {out_dir}\n")

    failed = []

    for zip_path in tqdm(zips, desc="Extracting"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
        except zipfile.BadZipFile:
            print(f"\n  [SKIP] Bad zip: {zip_path.name}")
            failed.append(zip_path.name)
        except Exception as e:
            print(f"\n  [ERROR] {zip_path.name}: {e}")
            failed.append(zip_path.name)

    print("\n=== Done ===")
    print(f"  Extracted : {len(zips) - len(failed)} / {len(zips)}")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    else:
        print("  All zips extracted successfully.")

if __name__ == "__main__":
    unzip_all()