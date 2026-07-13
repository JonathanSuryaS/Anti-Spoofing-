#!/usr/bin/env python3
"""
gradcam_compare.py — where do the baseline and SPSC models look?
================================================================

Generates Grad-CAM heatmaps for BOTH checkpoints on the SAME images, side by
side, so you can see whether they attend to the same regions.

Self-contained (no repo imports, no pytorch-grad-cam dependency) — Grad-CAM is
implemented directly with forward/backward hooks so the mechanism is visible.

Reads:
    checkpoints/resnet50_baseline.pth
    checkpoints/resnet50_spsc.pth
    data/processed_cropped/test_oulu/{live,spoof}
    data/processed_cropped/test_kaggle/{live,spoof}

Writes into results/gradcam/:
    gradcam_<domain>_<class>.png    grid: original | baseline CAM | SPSC CAM

Reading the output
------------------
Heat = "this region pushed the model toward SPOOF".
We always backprop the SPOOF logit (class 1), never the predicted class, so
every panel answers the same question and images stay comparable.

  heat on background / borders / whole frame  -> domain shortcut (bad)
  heat on face + plausible artifact regions   -> real spoof cues (good)
  baseline and SPSC attending differently     -> the sim-to-real gap, visualized

Each panel's title shows P(spoof) so you can tie attention back to the score.
"""

import os
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ CONFIG ------------------------------
DATA_ROOT    = "data/processed_masked"
TEST_DOMAINS = ["test_oulu", "test_kaggle"]
CLASSES      = ["live", "spoof"]
CHECKPOINTS  = {
    "baseline": "results/checkpoints/resnet50_baseline.pth",
    "spsc":     "results/checkpoints/resnet50_spsc.pth",
    "masked": "results/checkpoints/resnet50_masked.pth",
}
OUT_DIR      = "results/gradcam"

N_SAMPLES    = 4       # images per (domain, class) grid
SEED         = 42      # fixed so the same images are used every run
IMG_SIZE     = 224
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

SPOOF_CLASS  = 1       # backprop THIS logit, always (see docstring)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
IMG_EXTS      = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
# --------------------------------------------------------------------


eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
])


def build_model():
    m = resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    ckpt = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    return model


class GradCAM:
    """
    Grad-CAM on a target conv layer (here: ResNet-50's layer4).

    forward hook  -> stashes the layer's output activations   A  [1, C, 7, 7]
    backward hook -> stashes the gradient of the target logit dY/dA  [1, C, 7, 7]

    weights = global-average-pool of the gradients            [C]
              ("how much does turning this map up raise the spoof logit?")
    cam     = ReLU( sum_c  weights[c] * A[c] )                [7, 7]
              (weighted blend of the maps, keeping only positive evidence)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, target_class=SPOOF_CLASS):
        self.model.zero_grad()
        logits = self.model(x)                       # [1, 2]
        prob_spoof = F.softmax(logits, dim=1)[0, 1].item()

        logits[0, target_class].backward()           # populates the hooks

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1,1,7,7]
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        # normalize to [0,1] for display; guard the all-zero case (ReLU killed
        # everything = no positive evidence for spoof anywhere)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, prob_spoof


def denormalize(tensor):
    """Undo ImageNet normalization so we can show the actual image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def sample_images(domain, cls, n):
    d = os.path.join(DATA_ROOT, domain, cls)
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(d, ext))
    if not files:
        return []
    files = sorted(files)
    random.seed(SEED)
    return random.sample(files, min(n, len(files)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    cams = {}
    for name, path in CHECKPOINTS.items():
        if not os.path.isfile(path):
            raise SystemExit(f"[error] checkpoint not found: {path}")
        model = load_checkpoint(build_model(), path).to(DEVICE).eval()
        # layer4 = last conv block: richest semantics, still spatially located
        cams[name] = GradCAM(model, model.layer4)
        print(f"Loaded {name:8s} <- {path}")

    model_names = list(CHECKPOINTS.keys())

    for domain in TEST_DOMAINS:
        for cls in CLASSES:
            files = sample_images(domain, cls, N_SAMPLES)
            if not files:
                print(f"[skip] no images in {domain}/{cls}")
                continue

            n = len(files)
            ncol = 1 + len(model_names)          # original + one col per model
            fig, axes = plt.subplots(n, ncol, figsize=(4 * ncol, 4 * n))
            if n == 1:
                axes = np.array([axes])

            print(f"\n=== {domain} / {cls} ===")
            for r, fp in enumerate(files):
                img = Image.open(fp).convert("RGB")
                x = eval_tf(img).unsqueeze(0).to(DEVICE)
                x.requires_grad_(True)

                axes[r, 0].imshow(denormalize(x[0].detach()))
                axes[r, 0].set_title(f"original ({cls})", fontsize=11)
                axes[r, 0].axis("off")

                line = f"  {os.path.basename(fp)[:28]:30s}"
                for c, name in enumerate(model_names, start=1):
                    cam, p_spoof = cams[name](x, SPOOF_CLASS)
                    axes[r, c].imshow(denormalize(x[0].detach()))
                    axes[r, c].imshow(cam, cmap="jet", alpha=0.45)
                    axes[r, c].set_title(f"{name}  P(spoof)={p_spoof:.3f}", fontsize=11)
                    axes[r, c].axis("off")
                    line += f"  {name}={p_spoof:.3f}"
                print(line)

            fig.suptitle(f"Grad-CAM (spoof logit) — {domain} / {cls}", fontsize=13)
            fig.tight_layout()
            out = os.path.join(OUT_DIR, f"gradcam_{domain}_{cls}.png")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> {out}")

    print(f"\nDone. 4 grids written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()