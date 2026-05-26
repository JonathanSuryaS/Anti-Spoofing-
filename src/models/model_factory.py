"""
src/models/model_factory.py
Unified model factory — swap any architecture via config.
Supports: resnet18, efficientnet_b2, vit_base_patch16_224_dino, convit_base
"""

import torch
import torch.nn as nn
import timm


# ─────────────────────────────────────────────
# Model name → timm identifier mapping
# ─────────────────────────────────────────────

MODEL_MAP = {
    # Track 1: From scratch baseline
    "resnet18":                    "resnet18",

    # Track 2: Pretrained fine-tune
    "efficientnet_b2":             "efficientnet_b2",

    # Track 3: SOTA — ViT with DINO weights
    "vit_base_patch16_224_dino":   "vit_base_patch16_224.dino",

    # Track 4: Cross-domain generalization
    "convit_base":                 "convit_base",
}


# ─────────────────────────────────────────────
# FAS Classifier wrapper
# ─────────────────────────────────────────────

class FASModel(nn.Module):
    """
    Wraps any timm backbone into a binary FAS classifier.
    Replaces the classification head with:
        Dropout → Linear(num_features, 2)
    """

    def __init__(self, model_name: str, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        timm_name = MODEL_MAP.get(model_name)
        if timm_name is None:
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Choose from: {list(MODEL_MAP.keys())}"
            )

        # Load backbone without classification head
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,       # removes the head — we add our own
        )

        num_features = self.backbone.num_features

        # Custom FAS head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 2)
        )

        print(f"Model: {model_name} | Pretrained: {pretrained} | Features: {num_features}")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self):
        """Freeze backbone weights — used during warmup epochs for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen.")

    def unfreeze_backbone(self):
        """Unfreeze all weights for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen.")


# ─────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────

def build_model(cfg) -> FASModel:
    model = FASModel(
        model_name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.get("dropout", 0.3),
    )
    return model
