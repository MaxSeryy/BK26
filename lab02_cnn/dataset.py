import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class VisionDataset(Dataset):
    """
    Клас Dataset для завантаження зображень та міток за маніфестом dataset_split.json
    з підтримкою аугментації та нормалізації PyTorch.
    """
    def __init__(self, manifest_path="data/dataset_split.json", split="train", transform=None):
        self.manifest_path = manifest_path
        self.split = split
        self.transform = transform
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Маніфест не знайдено: {manifest_path}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        self.classes = manifest["classes"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        if split not in manifest["splits"]:
            raise ValueError(f"Підвибірка '{split}' відсутня у маніфесті.")
            
        self.samples = manifest["splits"][split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item["path"]
        label = self.class_to_idx[item["class"]]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, img_path


def get_data_loaders(manifest_path="data/dataset_split.json", batch_size=16, img_size=(64, 64)):
    """
    Створює DataLoaders для train, val та test вибірок.
    """
    # ImageNet статистика нормалізації
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = VisionDataset(manifest_path, split="train", transform=train_transform)
    val_dataset = VisionDataset(manifest_path, split="val", transform=eval_transform)
    test_dataset = VisionDataset(manifest_path, split="test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes
