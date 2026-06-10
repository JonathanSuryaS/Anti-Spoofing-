"""
SPSC - Simulated Physical Spoofing Clues (Enhanced)
====================================================
Reimplemented and extended from:
  He et al., CVPR 2024, "Joint Physical-Digital Facial Attack Detection
  Via Simulating Spoofing Clues"
 
The paper's SPSC bundles two effects (ColorJitter + moire). This version
keeps that core but decomposes physical spoofing into SEVERAL distinct,
individually-controllable artifacts, each mapped to a real-world phenomenon.
Inspired by the granularity of RizhaoCai/FAS-Aug, but implemented with NO
external assets (no ICC profiles, no texture image files) so it stays
dependency-free.
 
Artifacts and what they physically simulate
-------------------------------------------
  print_color_shift  : printer CMYK gamut compression (PRINT attack)
  color_jitter       : brightness/contrast/saturation/hue drift (PRINT)
  moire              : screen<->sensor grid interference fringes (REPLAY)
  glare              : specular highlight off glossy paper / screen (BOTH)
  motion_blur        : hand-held camera shake when re-capturing (BOTH)
  low_resolution     : detail loss from re-capturing an image (BOTH)
  jpeg_artifacts     : compression blocking from re-encoding (BOTH)
 
The `change_label` idea (from FAS-Aug)
--------------------------------------
Some artifacts ONLY appear in spoofs (moire, glare, print color shift) ->
applying them to a live image conceptually turns it into a spoof. Others
(motion blur, low-res, jpeg) can appear in live captures too -> they do
NOT change the label. We expose this via `return_label_change=True` so the
Dataset can optionally flip a live label to spoof when a spoof-defining
artifact was applied.
 
Usage
-----
    from augmentation.spsc import SPSC
 
    spsc = SPSC(p=0.5)              # transform for PIL images
    aug_img = spsc(pil_img)
 
    # If you want label-flipping behaviour:
    spsc = SPSC(p=0.5, return_label_change=True)
    aug_img, became_spoof = spsc(pil_img)
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ----------------------------------------------------------------------
# Individual artifact functions. Each takes a PIL.Image (RGB) + a single
# `strength` float in [0, 1] and returns a PIL.Image (RGB). A uniform
# signature makes them trivial to compose and to visualize.
# ----------------------------------------------------------------------

def print_color_shift(img: Image.Image, strength: float):
    """
    Simulate the color distortion of a printed photo.
 
    A real printer maps screen RGB into the smaller CMYK ink gamut, which
    compresses saturated colors and shifts hues. We approximate this by
    round-tripping RGB -> CMYK -> RGB (PIL's built-in, profile-free
    conversion) and blending the result back in proportion to `strength`.
    No ICC profile files required.
    """
    cmyk = img.convert("CMYK").convert("RGB")
    return Image.blend(img, cmyk, alpha=0.3 + 0.7 * strength)


def color_jitter(img: Image.Image, strength: float) -> Image.Image:
    """
    Random brightness / contrast / saturation / hue drift.
 
    Implemented with PIL ImageEnhance so magnitude scales cleanly with
    `strength` and there is no torch dependency here.
    """
    
    s = strength
    b = 1.0 + random.uniform(-0.5, 0.5) * s
    c = 1.0 + random.uniform(-0.5, 0.5) * s
    sat = random.uniform(-0.5, 0.5) * s
    
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(sat)
    
    # crude hue drift: roll the hue channel in HSV space
    hsv = np.array(img.convert("HSV")).astype(np.int16)
    hue_shift = int(random.uniform(-15, 15) * s)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    img = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
    return img


def moire(img: Image.Image, strength: float)->Image.Image:
    """
    Additive moire interference fringes (REPLAY attack signature).
 
    IMPORTANT: this is an ADDITIVE LUMINANCE OVERLAY, not a pixel warp.
    Real moire arises from beating between the display pixel grid and the
    camera sensor grid, showing up as faint periodic light/dark bands. We
    build a 2D sinusoid at a random orientation and frequency and blend it
    onto the image, which looks far more like a real replay than a warp.
    """
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    
    angle = random.uniform(0, np.pi)
    freq = random.uniform(0.10, 0.35)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = xx * np.cos(angle) + yy * np.sin(angle)
    fringe = np.sin(2 * np.pi * freq * proj) # in [-1, 1]
    
    amp = 30.0 * strength # keep subtle; face intact
    overlay = fringe[..., None] * amp
    out = np.clip(arr + overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def glare(img: Image.Image, strenght: float)-> Image.Image:
    """
    Specular highlight / reflection off glossy paper or a screen.
    A soft radial bright spot at a random location, blended additively.
    """
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    
    cx = random.uniform(0.2, 0.8) * w
    cy = random.uniform(0.2, 0.8) * h
    radius = random.uniform(0.2, 0.45) * min(h, w)
    
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = np.exp(-dist2 / (2 * radius ** 2)) # Gaussian fallof [0, 1]
    
    boost = 120.0 * strenght
    out = np.clip(arr + mask[..., None] * boost, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# Takes PIL image and a strength (Control how severe the blur is)
def motion_blur(img: Image.Image, strength: float)->Image.Image:
    """
    Directional blur from hand-shake while re-capturing (hand trembling).
    Kernel size grows with strength; direction is random.
    """
    arr = np.asarray(img)
    ksize = max(3, int(2 + 18 * strength))
    if ksize % 2 == 0:
        ksize += 1
    
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    direction = random.choice(["H", "V", "D", "U"])
    mid = ksize // 2
    if direction == "H": # Horizontal line through center row
        kernel[mid, :] = 1.0
    elif direction == "V": # Vertical line through center column
        kernel[:, mid] = 1.0
    elif direction == "D": # top - left -> bottom - right
        np.fill_diagonal(kernel, 1.0)
    else: # top - right -> bottom - left
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    kernel /= kernel.sum()
    
    out = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(out)


def low_resolution(img: Image.Image, strength: float)->Image.Image:
    """
    Detail loss from re-capturing an already-displayed/printed image.
    Downscale then upscale back; harsher downscale at higher strength.
    """
    w, h = img.size
    factor = 1.0 - 0.8 * strength
    sw, sh = max(8, int(w * factor)), max(8, int(h * factor))
    
    small = img.resize((sw, sh), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


def jpeg_artifacts(img: Image.Image, strength: float)->Image.Image:
    """
    Blocky JPEG compression artifacts from re-encoding a recaptured image.
    Lower quality at higher strength.
    """
    quality = int(90 -80 * strength)
    quality = max(5, min(95, quality))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------
# Registry: (function, prob, strength range, defines_spoof?)
# defines_spoof == True -> artifact only occurs in real attacks, so
# applying it to a live image flips the label.
# ----------------------------------------------------------------------


@dataclass
class _Artifact:
    fn: callable
    p: float
    smin: float
    smax: float
    defines_spoof: bool
    

DEFAULT_ARTIFACTS = {
    "print_color_shift": _Artifact(print_color_shift, 0.4, 0.3, 0.9, True),
    "color_jitter":      _Artifact(color_jitter,      0.5, 0.3, 1.0, False),
    "moire":             _Artifact(moire,             0.4, 0.3, 0.8, True),
    "glare":             _Artifact(glare,             0.3, 0.3, 0.8, True),
    "motion_blur":       _Artifact(motion_blur,       0.3, 0.2, 0.7, False),
    "low_resolution":    _Artifact(low_resolution,    0.3, 0.2, 0.7, False),
    "jpeg_artifacts":    _Artifact(jpeg_artifacts,    0.3, 0.3, 0.9, False),
}
    

class SPSC:
    """
    Simulated Physical Spoofing Clues augmentation.
 
    On each call it walks the artifact registry and applies each artifact
    independently with its own probability (gated by the global `p`),
    composing several physical effects into one augmented image.
 
    Args
    ----
    p : float
        Global multiplier on every artifact's individual probability.
        p=0 disables all augmentation; p=1 uses each artifact's own prob.
    artifacts : dict | None
        Override the default registry (e.g. print-only or replay-only).
    return_label_change : bool
        If True, __call__ returns (img, became_spoof).
    seed : int | None
        Optional seed for reproducible visualization.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        artifact: dict | None = None,
        return_label_change: bool = False,
        seed: int | None = None
    ):
        self.p = float(p)
        self.artifact = artifact if artifact is not None else DEFAULT_ARTIFACTS
        self.return_label_change = return_label_change
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
    
    def __call__(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        become_spoof = False
        for name, art in self.artifact.items():
            if random.random() < self.p * art.p :
                strength = random.uniform(art.smin, art.smax)
                img = art.fn(img, strength)
                if art.defines_spoof:
                    become_spoof = True
        
        if self.return_label_change:
            return img, become_spoof
        return img
    
    
    def __repr__(self):
        names = ", ".join(self.artifact.keys())
        return f"SPSC(p={self.p}, artifacts=[{names}])"

# Convenience presets ---------------------------------------------------
 
def print_only_spsc(p: float = 0.5, **kw) -> SPSC:
    """Restrict to PRINT-attack artifacts."""
    keys = ["print_color_shift", "color_jitter", "motion_blur",
            "low_resolution", "jpeg_artifacts"]
    return SPSC(p=p, artifacts={k: DEFAULT_ARTIFACTS[k] for k in keys}, **kw)
 
 
def replay_only_spsc(p: float = 0.5, **kw) -> SPSC:
    """Restrict to REPLAY-attack artifacts."""
    keys = ["moire", "glare", "color_jitter", "motion_blur",
            "low_resolution", "jpeg_artifacts"]
    return SPSC(p=p, artifacts={k: DEFAULT_ARTIFACTS[k] for k in keys}, **kw)
 
 
# Quick self-test -------------------------------------------------------
 
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) > 1:
        src = Image.open(sys.argv[1]).convert("RGB")
        print(f"Loaded {sys.argv[1]}  size={src.size}")
    else:
        src = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        print("Using random dummy image (224x224)")
 
    spsc = SPSC(p=1.0, return_label_change=True)  # force everything on
    out, flipped = spsc(src)
    out.save("spsc_test_output.jpg")
    print(f"{spsc}")
    print(f"became_spoof = {flipped}")
    print("Saved -> spsc_test_output.jpg")