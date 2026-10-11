import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from lab02_cnn.model import SimpleCNN
from lab03_computer_vision.models import get_transfer_model, get_base_model, count_parameters


def run_benchmark(num_iterations=120, warmup=15):
    device = torch.device("cpu")
    print(f"--- Бенчмарк продуктивності моделей комп'ютерного зору (Device: {device}) ---")
    print(f"Кількість ітерацій: {num_iterations}, Прогрів (warmup): {warmup} викликів.\n")
    
    # 1. Завантаження конфігурацій
    # Конфігурація 1: Базова SimpleCNN (64x64)
    model_base = get_base_model("models/best_cnn_model.pt", device=device)
    params_base_tot, _ = count_parameters(model_base)
    
    # Конфігурація 2: MobileNetV3-Small Transfer Learning (224x224)
    model_transfer = get_transfer_model(num_classes=3, freeze_backbone=False)
    ckpt = torch.load("models/best_transfer_mobilenet.pt", map_location=device)
    model_transfer.load_state_dict(ckpt["model_state_dict"])
    model_transfer.to(device)
    model_transfer.eval()
    params_trans_tot, _ = count_parameters(model_transfer)
    
    # Підготовка синтетичних та реальних кадрів
    sample_img_paths = [
        "data/dataset/car/car_001.jpg",
        "data/dataset/drone/drone_001.jpg",
        "data/dataset/pedestrian/pedestrian_001.jpg"
    ]
    raw_frames = [cv2.imread(p) for p in sample_img_paths if os.path.exists(p)]
    if not raw_frames:
        raw_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
        
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    configs = [
        {
            "name": "Base SimpleCNN (64x64)",
            "model": model_base,
            "target_size": (64, 64),
            "params": params_base_tot,
            "model_size_mb": os.path.getsize("models/best_cnn_model.pt") / (1024 * 1024)
        },
        {
            "name": "MobileNetV3-Small (224x224)",
            "model": model_transfer,
            "target_size": (224, 224),
            "params": params_trans_tot,
            "model_size_mb": os.path.getsize("models/best_transfer_mobilenet.pt") / (1024 * 1024)
        }
    ]
    
    benchmark_results = {}
    
    for cfg in configs:
        model = cfg["model"]
        size = cfg["target_size"]
        name = cfg["name"]
        print(f"Тестування конфігурації: {name} ...")
        
        # Фаза прогріву (Warmup)
        for i in range(warmup):
            frame = raw_frames[i % len(raw_frames)]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, size)
            tensor = torch.from_numpy(((resized / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(tensor)
                
        # Фаза вимірювань
        t_prep_list = []
        t_inf_list = []
        t_post_list = []
        t_total_list = []
        
        for i in range(num_iterations):
            frame = raw_frames[i % len(raw_frames)]
            
            # Preprocessing
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, size)
            tensor = torch.from_numpy(((resized / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
            t1 = time.perf_counter()
            
            # Inference
            with torch.no_grad():
                logits = model(tensor)
            t2 = time.perf_counter()
            
            # Postprocessing
            _ = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            t3 = time.perf_counter()
            
            t_prep = (t1 - t0) * 1000.0
            t_inf = (t2 - t1) * 1000.0
            t_post = (t3 - t2) * 1000.0
            t_total = (t3 - t0) * 1000.0
            
            t_prep_list.append(t_prep)
            t_inf_list.append(t_inf)
            t_post_list.append(t_post)
            t_total_list.append(t_total)
            
        mean_inf = float(np.mean(t_inf_list))
        std_inf = float(np.std(t_inf_list))
        min_inf = float(np.min(t_inf_list))
        max_inf = float(np.max(t_inf_list))
        median_inf = float(np.median(t_inf_list))
        
        mean_prep = float(np.mean(t_prep_list))
        mean_post = float(np.mean(t_post_list))
        mean_total = float(np.mean(t_total_list))
        fps = float(1000.0 / mean_total)
        
        benchmark_results[name] = {
            "target_resolution": f"{size[0]}x{size[1]}",
            "params_count": cfg["params"],
            "model_size_mb": round(cfg["model_size_mb"], 2),
            "inf_mean_ms": round(mean_inf, 2),
            "inf_std_ms": round(std_inf, 2),
            "inf_median_ms": round(median_inf, 2),
            "inf_min_ms": round(min_inf, 2),
            "inf_max_ms": round(max_inf, 2),
            "prep_mean_ms": round(mean_prep, 2),
            "post_mean_ms": round(mean_post, 2),
            "total_mean_ms": round(mean_total, 2),
            "fps": round(fps, 1)
        }
        
        print(f"  Inference: {mean_inf:.2f} ± {std_inf:.2f} ms (min: {min_inf:.2f}, max: {max_inf:.2f})")
        print(f"  Prep: {mean_prep:.2f} ms | Post: {mean_post:.2f} ms | Total: {mean_total:.2f} ms | FPS: {fps:.1f}\n")
        
    # Збереження результатів у JSON
    with open("lab03_computer_vision/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        
    # Побудова графіка порівняння
    plot_benchmark_comparison(benchmark_results, "lab03_computer_vision/figures/benchmark_comparison.png")


def plot_benchmark_comparison(results, save_path):
    labels = list(results.keys())
    inf_times = [results[k]["inf_mean_ms"] for k in labels]
    total_times = [results[k]["total_mean_ms"] for k in labels]
    fps_values = [results[k]["fps"] for k in labels]
    params_m = [results[k]["params_count"] / 1e6 for k in labels]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    
    # 1. Затримка (Latency)
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, inf_times, width, label="Inference Latency", color="#3b82f6")
    axes[0].bar(x + width/2, total_times, width, label="Total Pipeline Latency", color="#10b981")
    axes[0].set_ylabel("Час обробки (мс)", fontsize=10)
    axes[0].set_title("Затримка обробки (Latency, мс)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["SimpleCNN (64x64)", "MobileNetV3 (224x224)"], fontsize=9)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend()
    for i, v in enumerate(inf_times):
        axes[0].text(i - width/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(total_times):
        axes[0].text(i + width/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # 2. Пропускна спроможність (FPS)
    axes[1].bar(x, fps_values, width=0.45, color="#8b5cf6")
    axes[1].set_ylabel("Кадрів на секунду (FPS)", fontsize=10)
    axes[1].set_title("Пропускна здатність (FPS)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["SimpleCNN (64x64)", "MobileNetV3 (224x224)"], fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(fps_values):
        axes[1].text(i, v + 2, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # 3. Кількість параметрів (Мільйони)
    axes[2].bar(x, params_m, width=0.45, color="#f59e0b")
    axes[2].set_ylabel("Кількість параметрів (млн)", fontsize=10)
    axes[2].set_title("Складність моделі (млн ваг)", fontsize=11, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["SimpleCNN (64x64)", "MobileNetV3 (224x224)"], fontsize=9)
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(params_m):
        axes[2].text(i, v + 0.05, f"{v:.2f}M", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Порівняльний бенчмарк: Базова CNN (ЛР №2) проти Pretrained MobileNetV3 (ЛР №3)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Графік порівняльного бенчмарку збережено: {save_path}")


if __name__ == "__main__":
    run_benchmark(num_iterations=120, warmup=15)
