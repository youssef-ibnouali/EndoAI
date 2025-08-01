import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b7, efficientnet_b5, efficientnet_b6, swin_t, swin_s, swin_b, vit_b_16, vit_b_32
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet34_Weights, ResNet101_Weights, EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights, EfficientNet_B4_Weights, DenseNet121_Weights, DenseNet169_Weights, Swin_T_Weights, Swin_S_Weights, Swin_B_Weights, ViT_B_16_Weights
from torchvision.models import (
    efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l,
    EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
)
import torch.nn.functional as F
import timm


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

class AGClassifierResNet34(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class AGClassifierResNet101(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet101(weights=weights)
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
    
class AGClassifierDenseNet169(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = DenseNet169_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet169(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)


# DenseNet + Attention (Squeeze-and-Excitation Block)
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
        # 1) Backbone DenseNet121
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        # 2) adding a SEBlock before classification
        self.seblock = SEBlock(self.backbone.features[-1].num_features)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
    def forward(self, x):
        features = self.backbone.features(x) 
        features = self.seblock(features)     # apply Attention
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1,1)).view(features.size(0), -1)
        logits = self.backbone.classifier(out)
        return logits

class AGClassifierEfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)
    
class AGClassifierEfficientNetB1(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b1(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetB2(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b2(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetB3(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        #self.backbone.classifier = nn.Sequential(
        #    nn.Dropout(p=0.15), 
        #    nn.Linear(in_features, num_classes)
        #)
    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b4(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        #self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)
    
class AGClassifierEfficientNetB5(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = EfficientNet_B5_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b5(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)
    
class AGClassifierEfficientNetB6(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = EfficientNet_B6_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b6(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetB7(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B7_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b7(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetV2S(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetV2M(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_m(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)
    
class AGClassifierEfficientNetV2L(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_l(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)


class AGClassifierSwinT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin_t(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)
    

class AGClassifierSwinS(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = Swin_S_Weights.DEFAULT if pretrained else None
        self.backbone = swin_s(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)
    

class AGClassifierSwinB(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = Swin_B_Weights.DEFAULT if pretrained else None
        self.backbone = swin_b(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class AGClassifierViTB16(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)


def get_model(name: str, num_classes: int = 5, pretrained: bool = True):
    name = name.lower()
    if name == "resnet18":
        return AGClassifierResNet18(num_classes, pretrained)
    elif name == "resnet50":
        return AGClassifierResNet50(num_classes, pretrained)
    elif name == "resnet34":
        return AGClassifierResNet34(num_classes, pretrained)
    elif name == "resnet101":
        return AGClassifierResNet101(num_classes, pretrained)
    elif name == "densenet121":
        return AGClassifierDenseNet121(num_classes, pretrained)
    elif name == "densenet169":
        return AGClassifierDenseNet169(num_classes, pretrained)
    elif name == "efficientnetb0":
        return AGClassifierEfficientNetB0(num_classes, pretrained)
    elif name == "efficientnetb1":
        return AGClassifierEfficientNetB1(num_classes, pretrained)
    elif name == "efficientnetb2":
        return AGClassifierEfficientNetB2(num_classes, pretrained)
    elif name == "efficientnetb3":
        return AGClassifierEfficientNetB3(num_classes, pretrained)
    elif name == "efficientnetb4":
        return AGClassifierEfficientNetB4(num_classes, pretrained)
    elif name == "efficientnetb5":
        return AGClassifierEfficientNetB5(num_classes, pretrained)
    elif name == "efficientnetb7":
        return AGClassifierEfficientNetB7(num_classes, pretrained)
    elif name == "efficientnetb6":
        return AGClassifierEfficientNetB6(num_classes, pretrained)
    elif name == "efficientnetv2_s":
        return AGClassifierEfficientNetV2S(num_classes, pretrained)
    elif name == "efficientnetv2_m":
        return AGClassifierEfficientNetV2M(num_classes, pretrained)
    elif name == "efficientnetv2_l":
        return AGClassifierEfficientNetV2L(num_classes, pretrained)
    elif name == "densenetse":
        return AGClassifierDenseNetSE(num_classes, pretrained)
    elif name == "swin_t":
        return AGClassifierSwinT(num_classes, pretrained)
    elif name == "swin_s":
        return AGClassifierSwinS(num_classes, pretrained)
    elif name == "swin_b":
        return AGClassifierSwinB(num_classes, pretrained)
    elif name == "vit_b_16":
        return AGClassifierViTB16(num_classes, pretrained)
    elif name in timm.list_models():
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")