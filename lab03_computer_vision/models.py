import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.models as models
from lab02_cnn.model import SimpleCNN


def get_transfer_model(num_classes=3, freeze_backbone=True):
    """
    Завантажує попередньо навчену модель MobileNetV3-Small з вагами ImageNet
    та адаптує класифікаційну голову під num_classes цільового датасету.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    
    # Заморожування ваг екстрактора ознак (feature extraction)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
    # Заміна фінального повнозв'язного шару класифікатора
    # В MobileNetV3-Small: classifier[0] = Linear(576, 1024), classifier[1] = Hardswish, classifier[2] = Dropout, classifier[3] = Linear(1024, 1000)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features=in_features, out_features=num_classes)
    
    return model


def get_base_model(checkpoint_path="models/best_cnn_model.pt", num_classes=3, device="cpu"):
    """
    Завантажує базову модель SimpleCNN з ЛР №2.
    """
    model = SimpleCNN(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    """Повертає загальну та навчальну кількість параметрів."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    mb_model = get_transfer_model(num_classes=3, freeze_backbone=True)
    total_p, train_p = count_parameters(mb_model)
    print(f"MobileNetV3-Small (Transfer Learning):")
    print(f"  Всього параметрів: {total_p:,}")
    print(f"  Навчальних параметрів: {train_p:,}")
