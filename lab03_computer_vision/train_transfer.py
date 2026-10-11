import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from lab02_cnn.dataset import get_data_loaders
from lab03_computer_vision.models import get_transfer_model


def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_transfer_model(epochs=10, batch_size=16, lr=0.0005, manifest_path="data/dataset_split.json"):
    set_seed(13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій для навчання: {device}")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("lab03_computer_vision/figures", exist_ok=True)
    
    # Використовуємо стандартний для MobileNet розмір 224x224
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        img_size=(224, 224)
    )
    
    model = get_transfer_model(num_classes=len(classes), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_checkpoint_path = "models/best_transfer_mobilenet.pt"
    
    print("\n--- Початок Transfer Learning (MobileNetV3-Small) ---")
    for epoch in range(1, epochs + 1):
        # 1. Тренування
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # 2. Валідація
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Епоха [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")
        
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "classes": classes
            }, best_checkpoint_path)
            
    print(f"\nНавчання завершено. Найкраща Val Acc: {best_val_acc * 100:.2f}%")
    
    # 3. Графіки кривих
    plot_curves(history, "lab03_computer_vision/figures/transfer_learning_curves.png")
    
    # 4. Оцінювання на Test
    evaluate_transfer_test(best_checkpoint_path, test_loader, classes, device)


def plot_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(11, 4.8))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=1.8)
    plt.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=1.8)
    plt.title("Функція втрат Transfer Learning", fontsize=12, fontweight="bold")
    plt.xlabel("Епоха")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy", linewidth=1.8)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "r--s", label="Val Accuracy", linewidth=1.8)
    plt.title("Точність Transfer Learning (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Епоха")
    plt.ylabel("Точність (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Графік збережено: {save_path}")


def evaluate_transfer_test(checkpoint_path, test_loader, classes, device):
    model = get_transfer_model(num_classes=len(classes), freeze_backbone=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    print("\nЗвіт класифікації MobileNetV3-Small (Test):\n",
          classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
    metrics = {
        "model": "MobileNetV3-Small Pretrained",
        "best_epoch": checkpoint["epoch"],
        "val_acc": float(checkpoint["val_acc"]),
        "test_accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {cls: report[cls] for cls in classes}
    }
    
    with open("lab03_computer_vision/transfer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train_transfer_model(epochs=10, batch_size=16)
