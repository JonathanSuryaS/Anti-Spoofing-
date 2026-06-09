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