"""
demo/app.py
Live webcam / image upload demo using your best trained model.

Usage:
    python demo/app.py --checkpoint results/checkpoints/best.ckpt --config configs/vit_dino.yaml
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
from omegaconf import OmegaConf
from PIL import Image

from src.models.model_factory import build_model
from src.data.dataset import get_val_transforms


def load_model(checkpoint_path: str, cfg):
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Strip Lightning prefix if needed
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.eval()
    return model


def predict(image: np.ndarray, model, transform):
    """Run inference on a single image (H x W x C, RGB uint8)."""
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0)   # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    real_prob  = float(probs[1])
    spoof_prob = float(probs[0])

    label = "REAL ✅" if real_prob > 0.5 else "SPOOF ❌"
    confidence = max(real_prob, spoof_prob) * 100

    return {
        "prediction": label,
        "confidence": f"{confidence:.1f}%",
        "real_prob":  f"{real_prob * 100:.1f}%",
        "spoof_prob": f"{spoof_prob * 100:.1f}%",
    }


def build_demo(model, transform):
    def run(image):
        if image is None:
            return "No image provided."
        image_rgb = np.array(image.convert("RGB"))
        result = predict(image_rgb, model, transform)
        return (
            f"**{result['prediction']}** ({result['confidence']} confidence)\n\n"
            f"Real probability:  {result['real_prob']}\n"
            f"Spoof probability: {result['spoof_prob']}"
        )

    return gr.Interface(
        fn=run,
        inputs=gr.Image(type="pil", label="Upload a face image"),
        outputs=gr.Markdown(label="Result"),
        title="Face Anti-Spoofing Demo",
        description="Upload a face photo to detect if it's real or a spoof (print/replay attack).",
        examples=[],
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",     type=str, required=True)
    args = parser.parse_args()

    cfg       = OmegaConf.load(args.config)
    transform = get_val_transforms(cfg.data.image_size)
    model     = load_model(args.checkpoint, cfg)

    demo = build_demo(model, transform)
    demo.launch(share=True)     # share=True gives a public link
