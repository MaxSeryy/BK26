import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

from lab02_cnn.model import SimpleCNN
from lab02_cnn.dataset import get_data_loaders, VisionDataset
import torchvision.transforms as transforms


def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(epochs=20, batch_size=16, lr=0.001, manifest_path="data/dataset_split.json"):
    set_seed(13)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовується пристрій: {device}")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("lab02_cnn/figures", exist_ok=True)
    
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        img_size=(64, 64)
    )
    
    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    best_model_path = "models/best_cnn_model.pt"
    
    print("\n--- Початок навчання SimpleCNN ---")
    for epoch in range(1, epochs + 1):
        # 1. Режим навчання
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # 2. Режим валідації
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Епоха [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")
        
        # Збереження найкращої моделі за валідаційною точністю
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "classes": classes
            }, best_model_path)
            print(f" -> Збережено нову найкращу модель з Val Acc: {best_val_acc * 100:.2f}%")
            
    print(f"\nНавчання завершено. Найкраща валідаційна точність: {best_val_acc * 100:.2f}%")
    
    # 3. Побудова графіків навчання
    plot_training_curves(history, "lab02_cnn/figures/training_curves.png")
    
    # 4. Оцінювання на test вибірці
    evaluate_test_set(best_model_path, test_loader, classes, device)


def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Графік втрат
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=1.8, markersize=4)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=1.8, markersize=4)
    plt.title("Динаміка функції втрат (Loss)", fontsize=13, fontweight="bold")
    plt.xlabel("Епоха", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    
    # Графік точності
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy", linewidth=1.8, markersize=4)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "r--s", label="Validation Accuracy", linewidth=1.8, markersize=4)
    plt.title("Динаміка точності (Accuracy)", fontsize=13, fontweight="bold")
    plt.xlabel("Епоха", fontsize=11)
    plt.ylabel("Точність (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Графік навчання збережено: {save_path}")


def evaluate_test_set(checkpoint_path, test_loader, classes, device):
    print("\n--- Оцінювання найкращої моделі на Test вибірці ---")
    model = SimpleCNN(num_classes=len(classes)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Classification Report
    report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\nЗвіт класифікації (Classification Report):\n", report_text)
    
    # Матриця помилок
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes, "lab02_cnn/figures/confusion_matrix.png")
    
    # Збереження числових метрик
    metrics_summary = {
        "best_epoch": checkpoint["epoch"],
        "val_acc": float(checkpoint["val_acc"]),
        "test_accuracy": float(report_dict["accuracy"]),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "per_class": {cls: report_dict[cls] for cls in classes},
        "confusion_matrix": cm.tolist()
    }
    
    with open("lab02_cnn/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
    # Візуалізація прогнозів (5 правильних і 5 помилкових)
    visualize_predictions(model, checkpoint_path, test_loader, classes, device, all_preds, all_labels, all_probs, all_paths)


def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Матриця невідповідностей (Confusion Matrix)", fontsize=12, fontweight="bold")
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{val}",
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=12,
                     fontweight="bold",
                     color="white" if val > thresh else "black")
                     
    plt.ylabel("Істинний клас (Ground Truth)", fontsize=11)
    plt.xlabel("Передбачений клас (Predicted)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Матрицю помилок збережено: {save_path}")


def visualize_predictions(model, checkpoint_path, test_loader, classes, device, all_preds, all_labels, all_probs, all_paths):
    correct_indices = np.where(all_preds == all_labels)[0]
    incorrect_indices = np.where(all_preds != all_labels)[0]
    
    print(f"\nКількість правильних прогнозів на чистій тестовій вибірці: {len(correct_indices)}/{len(all_labels)}")
    print(f"Кількість помилкових прогнозів на чистій тестовій вибірці: {len(incorrect_indices)}/{len(all_labels)}")
    
    # 5 правильних прогнозів на тестовій вибірці
    selected_correct = list(correct_indices[:5])
    
    # Знаходження або доповнення помилкових випадків (стрес-тест завадостійкості)
    selected_incorrect = []
    for idx in incorrect_indices:
        selected_incorrect.append({
            "img": Image.open(all_paths[idx]).convert("RGB"),
            "true_idx": all_labels[idx],
            "pred_idx": all_preds[idx],
            "conf": float(all_probs[idx][all_preds[idx]]),
            "note": "Помилка (тест)"
        })
        
    if len(selected_incorrect) < 5:
        print("Чиста тестова вибірка класифікована бездоганно. Для аналізу помилок моделі проводиться стрес-тест із завадами (шум, розмиття, перекриття)...")
        import cv2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((64, 64))
        ds = VisionDataset("data/dataset_split.json", split="test", transform=None)
        
        for i in range(len(ds)):
            raw_img, lbl, path = ds[i]
            img_arr = np.array(raw_img).copy()
            
            # Спроба 1: Гауссів шум
            noise = np.random.normal(0, 110, img_arr.shape).astype(np.float32)
            noisy = np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            t_noisy = normalize(to_tensor(resize(Image.fromarray(noisy)))).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t_noisy)
                pr = out.argmax(1).item()
                prob = torch.softmax(out, 1)[0, pr].item()
            if pr != lbl:
                selected_incorrect.append({
                    "img": Image.fromarray(noisy),
                    "true_idx": lbl,
                    "pred_idx": pr,
                    "conf": prob,
                    "note": "Гауссів шум"
                })
            if len(selected_incorrect) >= 5:
                break
                
            # Спроба 2: Сильне розмиття
            blurred = cv2.GaussianBlur(img_arr, (31, 31), 15)
            t_blur = normalize(to_tensor(resize(Image.fromarray(blurred)))).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t_blur)
                pr = out.argmax(1).item()
                prob = torch.softmax(out, 1)[0, pr].item()
            if pr != lbl:
                selected_incorrect.append({
                    "img": Image.fromarray(blurred),
                    "true_idx": lbl,
                    "pred_idx": pr,
                    "conf": prob,
                    "note": "Розмиття"
                })
            if len(selected_incorrect) >= 5:
                break
                
            # Спроба 3: Перекриття частини об'єкта (cutout)
            occ = img_arr.copy()
            h, w = occ.shape[:2]
            ch, cw = h // 2, w // 2
            occ[ch-45:ch+45, cw-45:cw+45] = 0
            t_occ = normalize(to_tensor(resize(Image.fromarray(occ)))).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t_occ)
                pr = out.argmax(1).item()
                prob = torch.softmax(out, 1)[0, pr].item()
            if pr != lbl:
                selected_incorrect.append({
                    "img": Image.fromarray(occ),
                    "true_idx": lbl,
                    "pred_idx": pr,
                    "conf": prob,
                    "note": "Перекриття (cutout)"
                })
            if len(selected_incorrect) >= 5:
                break

    # Побудова сітки 2 рядки x 5 стовпців
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.8))
    
    # 5 правильних
    for col, idx in enumerate(selected_correct):
        ax = axes[0, col]
        img_p = all_paths[idx]
        img = Image.open(img_p).convert("RGB")
        true_cls = classes[all_labels[idx]]
        pred_cls = classes[all_preds[idx]]
        conf = all_probs[idx][all_preds[idx]] * 100
        
        ax.imshow(img)
        ax.set_title(f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)\n[Успішно]", color="darkgreen", fontsize=9, fontweight="bold")
        ax.axis("off")
        
    axes[0, 0].text(-0.25, 0.5, "Правильні\nпрогнози", transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight="bold", va="center", ha="right", color="darkgreen")

    # 5 помилкових
    for col in range(5):
        ax = axes[1, col]
        item = selected_incorrect[col]
        img = item["img"]
        true_cls = classes[item["true_idx"]]
        pred_cls = classes[item["pred_idx"]]
        conf = item["conf"] * 100
        
        ax.imshow(img)
        ax.set_title(f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)\n[{item['note']}]", color="darkred", fontsize=9, fontweight="bold")
        ax.axis("off")
        
    axes[1, 0].text(-0.25, 0.5, "Помилкові\nпрогнози", transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight="bold", va="center", ha="right", color="darkred")

    plt.suptitle("Приклади класифікації: 5 правильних та 5 помилкових прогнозів (аналіз стійкості)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = "lab02_cnn/figures/predictions_sample.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Зразок прогнозів збережено: {save_path}")


if __name__ == "__main__":
    train_model(epochs=20, batch_size=16, lr=0.001)
