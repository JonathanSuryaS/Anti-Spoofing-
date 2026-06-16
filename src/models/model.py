"""
FAS Model
=========
Pretrained ResNet50 backbone + a 2-class head (live vs spoof).

Why ResNet50 + ImageNet pretrain:
  - Standard, well-understood FAS baseline -> credible, reproducible.
  - Pretrained features transfer well and train fast on a small set.
  - Keeps the project's focus on the interesting parts (cross-dataset
    evaluation + augmentation ablation), not backbone novelty.

The ImageNet normalization in the Dataset matches this pretrained backbone.

Output: raw logits of shape (B, 2). Use CrossEntropyLoss (which applies
softmax internally). For a spoof "score" at eval time, take
softmax(logits)[:, SPOOF].
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Args
    ----
    num_classes : 2 (live, spoof)
    pretrained  : load ImageNet weights (recommended True)
    freeze_backbone : if True, train only the new head (faster, weaker).
                      Default False -> fine-tune the whole network.
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # swap the final FC layer for our 2-class head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # (new fc always has requires_grad=True even if backbone frozen)

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = build_model(pretrained=True).to(device)

    # sanity: one forward pass on a fake batch
    x = torch.randn(4, 3, 224, 224, device=device)
    out = model(x)
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(out.shape)}  (expect (4, 2))")

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params : {n_params/1e6:.1f}M total, {n_train/1e6:.1f}M trainable")