# =============================================================
# model.py  —  Build EfficientNetB0 for fish classification
# =============================================================
# What this file does:
#   1. Loads EfficientNetB0 pretrained on ImageNet
#   2. Freezes all layers (keep what it already learned)
#   3. Replaces the last layer for our 31 fish species
#   4. Only the last layer trains initially
# =============================================================

import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=31, pretrained=False):
    """
    Loads EfficientNetB0 and adapts it for fish classification.
    
    Args:
        num_classes: number of fish species (31 in our case)
        pretrained: whether to download ImageNet weights (True for training from scratch, False for inference/checkpoints)
    
    Returns:
        model ready for training or inference
    """

    # Step 1: Load EfficientNetB0 (pretrained on ImageNet if training from scratch)
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Step 2: Freeze backbone layers if training transfer learning
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # Step 3: Replace the classifier (last layer)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ── Quick test ─────────────────────────────────────────────────
# Run: python src/model.py
if __name__ == "__main__":
    model = build_model(num_classes=31)
    print(model.classifier)  # should show Linear(1280 → 31)

    # Test with a dummy image batch
    dummy = torch.randn(2, 3, 224, 224)  # 2 fake images
    output = model(dummy)
    print(f"Output shape: {output.shape}")  # should be [2, 31]