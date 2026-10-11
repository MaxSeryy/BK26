import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Згорткова нейронна мережа для класифікації зображень з 3 згортковими блоками,
    пакетною нормалізацією (BatchNorm), підвибіркою (MaxPool) та Dropout для регуляризації.
    """
    def __init__(self, num_classes=3, in_channels=3):
        super(SimpleCNN, self).__init__()
        
        # Блок 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Блок 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Блок 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Адаптивний пулінг та класифікатор
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SimpleCNN(num_classes=3)
    dummy_input = torch.randn(4, 3, 64, 64)
    out = model(dummy_input)
    print(f"Model output shape: {out.shape}")
    print(f"Total trainable parameters: {model.count_parameters():,}")
