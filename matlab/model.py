import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    efficientnet_b0, efficientnet_b4,
    EfficientNet_B0_Weights, EfficientNet_B4_Weights,
    ResNet18_Weights, ResNet50_Weights,
    DenseNet121_Weights
)

# === Backbone Wrappers ===

class AGClassifierResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class AGClassifierResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class AGClassifierDenseNet121(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b4(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# === SE Block for DenseNet + Attention ===

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class AGClassifierDenseNetSE(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        self.seblock = SEBlock(self.backbone.features[-1].num_features)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.seblock(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        logits = self.backbone.classifier(out)
        return logits

# === Model Selection ===

def get_model(name: str, num_classes: int = 5, pretrained: bool = True):
    name = name.lower()
    if name == "resnet18":
        return AGClassifierResNet18(num_classes, pretrained)
    elif name == "resnet50":
        return AGClassifierResNet50(num_classes, pretrained)
    elif name == "densenet121":
        return AGClassifierDenseNet121(num_classes, pretrained)
    elif name == "efficientnetb0":
        return AGClassifierEfficientNetB0(num_classes, pretrained)
    elif name == "efficientnetb4":
        return AGClassifierEfficientNetB4(num_classes, pretrained)
    elif name == "densenetse":
        return AGClassifierDenseNetSE(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {name}")
