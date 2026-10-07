"""Model definition: ImageNet-pretrained EfficientNet-B0 with a 2-class head."""
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from ml.config import CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, len(CLASSES)))
    return model
