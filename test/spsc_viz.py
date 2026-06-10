"""
SPSC Augmentation Visualizer (Enhanced)
Shows each physical artifact INDIVIDUALLY plus the full composed SPSC,
so you can see and tune what every effect does to your real face crops.

Usage:
  python augmentation/visualize_spsc.py
  python augmentation/visualize_spsc.py path/to/an/image.jpg

Output: spsc_visualization.jpg
  Rows    = source images
  Columns = original + each artifact at fixed strength + full SPSC
"""

from pathlib import Path

import matplotlib.pyplot as plt
import sys
from PIL import Image

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from spsc_augmentation import (
    print_color_shift, color_jitter, moire, glare,
    motion_blur, low_resolution, jpeg_artifacts, SPSC,
)

# ── CONFIG ──────────────────────────────────────────────
SAMPLE_DIR = r"C:\Users\user\Documents\GitHub\Anti-Spoofing-\data\processed\live"
N_SAMPLES  = 3
OUT_PATH   = "spsc_visualization.jpg"
FIXED_STRENGTH = 0.7    # strength used for the per-artifact columns
# ────────────────────────────────────────────────────────

# (column label, function or None for original / "spsc" for full pipeline)
COLUMNS = [
    ("Original",      None),
    ("PrintColor",    print_color_shift),
    ("ColorJitter",   color_jitter),
    ("Moire",         moire),
    ("Glare",         glare),
    ("MotionBlur",    motion_blur),
    ("LowRes",        low_resolution),
    ("JPEG",          jpeg_artifacts),
    ("Full SPSC",     "spsc"),
]


def main():
    paths = sorted(Path(SAMPLE_DIR).glob("*.jpg"))[:N_SAMPLES]
    if not paths:
        print(f"No images in {SAMPLE_DIR} - update SAMPLE_DIR")
        return

    full_spsc = SPSC(p=1.0)
    ncols = len(COLUMNS)
    fig, axes = plt.subplots(len(paths), ncols,
                             figsize=(2.2 * ncols, 2.6 * len(paths)))
    if len(paths) == 1:
        axes = axes.reshape(1, -1)

    for r, p in enumerate(paths):
        base = Image.open(p).convert("RGB")
        for c, (label, fn) in enumerate(COLUMNS):
            if fn is None:
                out = base
            elif fn == "spsc":
                out = full_spsc(base)
            else:
                out = fn(base, FIXED_STRENGTH)
            axes[r, c].imshow(out)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(label, fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=100, bbox_inches="tight")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()