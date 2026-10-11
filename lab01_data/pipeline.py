import os
import glob
import json
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def analyze_and_clean_dataset(dataset_dir="data/dataset"):
    """
    Сканує датасет, перевіряє цілісність кожного файлу,
    збирає статистику щодо розмірів, кількості та пошкоджених файлів.
    """
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()
    
    total_files = 0
    valid_records = []
    corrupted_records = []
    
    for cls in classes:
        cls_path = os.path.join(dataset_dir, cls)
        file_list = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total_files += len(file_list)
        
        for fname in file_list:
            fpath = os.path.join(cls_path, fname)
            file_size = os.path.getsize(fpath)
            
            # Check 0-byte file
            if file_size == 0:
                corrupted_records.append({
                    "path": fpath,
                    "class": cls,
                    "reason": "Файл має нульовий розмір (0 байт)"
                })
                continue
                
            # Attempt to decode with OpenCV
            img = cv2.imread(fpath)
            if img is None:
                corrupted_records.append({
                    "path": fpath,
                    "class": cls,
                    "reason": "Пошкоджений заголовок або некоректний формат декодування"
                })
                continue
                
            h, w, c = img.shape
            valid_records.append({
                "path": fpath,
                "filename": fname,
                "class": cls,
                "width": w,
                "height": h,
                "channels": c,
                "size_kb": round(file_size / 1024, 2),
                "aspect_ratio": round(w / h, 2)
            })
            
    return classes, valid_records, corrupted_records, total_files

def letterbox_resize(img, target_size=(224, 224), pad_color=(0, 0, 0)):
    """
    Масштабує зображення зі збереженням пропорцій (letterboxing)
    та доповненням полів (padding) до цільового розміру.
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def apply_augmentations(img_rgb, seed=None):
    """
    Реалізація 4 валідних методів аугментації для комп'ютерного зору:
    1. Horizontal Flip
    2. Affine Rotation (-15 to +15 deg)
    3. Contrast & Brightness Jitter
    4. Gaussian Noise
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    # 1. Flip
    flipped = cv2.flip(img_rgb, 1)
    
    # 2. Rotation with scaling
    h, w = img_rgb.shape[:2]
    angle = random.uniform(-15, 15)
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale=1.05)
    rotated = cv2.warpAffine(img_rgb, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 3. Brightness & Contrast
    alpha = random.uniform(0.8, 1.25) # contrast
    beta = random.randint(-20, 20)    # brightness
    jittered = np.clip(alpha * img_rgb.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    
    # 4. Gaussian Noise
    noise = np.random.normal(0, 15, img_rgb.shape).astype(np.float32)
    noisy = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return {
        "original": img_rgb,
        "flipped": flipped,
        "rotated": rotated,
        "jittered": jittered,
        "noisy": noisy
    }

def run_full_pipeline():
    random.seed(13)
    np.random.seed(13)
    
    # Step 1: Analyze & Clean
    classes, valid_records, corrupted_records, total_files = analyze_and_clean_dataset("data/dataset")
    print(f"Total files found: {total_files}")
    print(f"Valid files: {len(valid_records)}, Corrupted: {len(corrupted_records)}")
    
    # Step 2: Stratified Split 70% Train, 15% Val, 15% Test with Seed 13
    paths = [r["path"] for r in valid_records]
    labels = [r["class"] for r in valid_records]
    
    train_paths, test_val_paths, train_labels, test_val_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=13, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_val_paths, test_val_labels, test_size=0.50, random_state=13, stratify=test_val_labels
    )
    
    split_manifest = {
        "random_seed": 13,
        "total_valid": len(valid_records),
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "classes": classes,
        "splits": {
            "train": [{"path": p, "class": l} for p, l in zip(train_paths, train_labels)],
            "val": [{"path": p, "class": l} for p, l in zip(val_paths, val_labels)],
            "test": [{"path": p, "class": l} for p, l in zip(test_paths, test_labels)]
        }
    }
    os.makedirs("data", exist_ok=True)
    with open("data/dataset_split.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)
        
    # Calculate statistics per split and class
    split_stats = {}
    for sp_name, sp_labels in [("Train", train_labels), ("Validation", val_labels), ("Test", test_labels)]:
        split_stats[sp_name] = {cls: sp_labels.count(cls) for cls in classes}
        
    widths = [r["width"] for r in valid_records]
    heights = [r["height"] for r in valid_records]
    aspect_ratios = [r["aspect_ratio"] for r in valid_records]
    sizes_kb = [r["size_kb"] for r in valid_records]
    
    stats_summary = {
        "classes": classes,
        "total_found": total_files,
        "valid_count": len(valid_records),
        "corrupted_count": len(corrupted_records),
        "corrupted_details": corrupted_records,
        "split_stats": split_stats,
        "dimensions": {
            "min_w": int(np.min(widths)), "max_w": int(np.max(widths)), "mean_w": float(np.mean(widths)),
            "min_h": int(np.min(heights)), "max_h": int(np.max(heights)), "mean_h": float(np.mean(heights)),
            "min_kb": float(np.min(sizes_kb)), "max_kb": float(np.max(sizes_kb)), "mean_kb": float(np.mean(sizes_kb))
        }
    }
    with open("data/stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)
        
    print("Statistics summary generated.")
    
    os.makedirs("lab01_data/figures", exist_ok=True)
    # -------------------------------------------------------------
    # Visualization 1: Class distribution across Train/Val/Test
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    x = np.arange(len(classes))
    width = 0.25
    
    train_vals = [split_stats["Train"][cls] for cls in classes]
    val_vals = [split_stats["Validation"][cls] for cls in classes]
    test_vals = [split_stats["Test"][cls] for cls in classes]
    
    plt.bar(x - width, train_vals, width, label='Train (70%)', color='#2E86C1')
    plt.bar(x, val_vals, width, label='Validation (15%)', color='#28B463')
    plt.bar(x + width, test_vals, width, label='Test (15%)', color='#E67E22')
    
    plt.xlabel('Семантичні класи', fontsize=11, fontweight='bold')
    plt.ylabel('Кількість зображень', fontsize=11, fontweight='bold')
    plt.title('Розподіл зразків за класами у сформованих підвибірках', fontsize=12, fontweight='bold')
    plt.xticks(x, [f"Клас: {cls}" for cls in classes], fontsize=10)
    plt.ylim(0, max(train_vals) * 1.25)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i in range(len(classes)):
        plt.text(i - width, train_vals[i] + 1.5, str(train_vals[i]), ha='center', fontsize=9, fontweight='bold')
        plt.text(i, val_vals[i] + 1.5, str(val_vals[i]), ha='center', fontsize=9, fontweight='bold')
        plt.text(i + width, test_vals[i] + 1.5, str(test_vals[i]), ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("lab01_data/figures/chart_class_distribution.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Visualization 2: Sample images grid (4 per class)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(len(classes), 4, figsize=(11, 7.5), dpi=300)
    for c_idx, cls in enumerate(classes):
        cls_records = [r for r in valid_records if r["class"] == cls][:4]
        for s_idx, rec in enumerate(cls_records):
            img_bgr = cv2.imread(rec["path"])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax = axes[c_idx, s_idx]
            ax.imshow(img_rgb)
            ax.set_title(f"{cls} #{s_idx+1}\n({rec['width']}x{rec['height']})", fontsize=9)
            ax.axis('off')
            
    plt.suptitle("Приклади вибірки для трьох досліджуваних класів датасету", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("lab01_data/figures/sample_classes.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Visualization 3: Augmentations comparison (Original vs 4 types)
    # -------------------------------------------------------------
    sample_img_rec = valid_records[0]
    sample_bgr = cv2.imread(sample_img_rec["path"])
    sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
    sample_resized = letterbox_resize(sample_rgb, (224, 224))
    
    aug_dict = apply_augmentations(sample_resized, seed=42)
    
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), dpi=300)
    titles = [
        "Оригінал (Letterbox)",
        "Горизонтальний Flip",
        "Афінний поворот (-15°..15°)",
        "Колір / Контраст (Jitter)",
        "Гауссів шум"
    ]
    keys = ["original", "flipped", "rotated", "jittered", "noisy"]
    
    for ax, title, key in zip(axes, titles, keys):
        ax.imshow(aug_dict[key])
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.axis('off')
        
    plt.suptitle("Результати застосування розроблених методів аугментації", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("lab01_data/figures/augmentation_samples.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Visualization 4: Original dimensions distribution scatter
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 4.5), dpi=300)
    plt.scatter(widths, heights, c='#1F618D', alpha=0.6, edgecolors='none', s=45)
    plt.xlabel('Ширина зображення (пікселі)', fontsize=10, fontweight='bold')
    plt.ylabel('Висота зображення (пікселі)', fontsize=10, fontweight='bold')
    plt.title('Розподіл просторових розмірів сирих зображень датасету', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("lab01_data/figures/dimensions_distribution.png", dpi=300)
    plt.close()
    
    print("All visualizations created in lab01_data/figures/")

if __name__ == "__main__":
    run_full_pipeline()
