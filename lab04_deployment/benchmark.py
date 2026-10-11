import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import cv2
import numpy as np
import torch
import onnxruntime as ort
import openvino as ov
import matplotlib.pyplot as plt

from lab02_cnn.model import SimpleCNN


def run_deployment_benchmark(num_iterations=150, warmup=20):
    print("--- Порівняльний бенчмаркінг: PyTorch vs ONNX Runtime vs OpenVINO ---")
    print(f"Умови: CPU, {num_iterations} ітерацій після {warmup} викликів warm-up.\n")
    
    onnx_path = "models/simple_cnn.onnx"
    pt_path = "models/best_cnn_model.pt"
    
    if not os.path.exists(onnx_path) or not os.path.exists(pt_path):
        raise FileNotFoundError("Чекпоінти моделі не знайдено.")
        
    device = torch.device("cpu")
    
    # 1. Ініціалізація PyTorch
    pt_model = SimpleCNN(num_classes=3)
    ckpt = torch.load(pt_path, map_location=device)
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()
    
    # 2. Ініціалізація ONNX Runtime
    ort_opts = ort.SessionOptions()
    ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(onnx_path, sess_options=ort_opts, providers=["CPUExecutionProvider"])
    ort_input_name = ort_session.get_inputs()[0].name
    
    # 3. Ініціалізація OpenVINO
    ov_core = ov.Core()
    ov_model = ov_core.read_model(onnx_path)
    ov_compiled = ov_core.compile_model(ov_model, "CPU")
    ov_output = ov_compiled.output(0)
    
    # Створення тестових даних
    np.random.seed(13)
    input_b1_np = np.random.randn(1, 3, 64, 64).astype(np.float32)
    input_b1_torch = torch.from_numpy(input_b1_np)
    
    input_b8_np = np.random.randn(8, 3, 64, 64).astype(np.float32)
    input_b8_torch = torch.from_numpy(input_b8_np)
    
    runtimes = [
        {
            "name": "PyTorch (CPU)",
            "fn_b1": lambda: pt_model(input_b1_torch),
            "fn_b8": lambda: pt_model(input_b8_torch)
        },
        {
            "name": "ONNX Runtime (CPU)",
            "fn_b1": lambda: ort_session.run(None, {ort_input_name: input_b1_np}),
            "fn_b8": lambda: ort_session.run(None, {ort_input_name: input_b8_np})
        },
        {
            "name": "Intel OpenVINO (CPU)",
            "fn_b1": lambda: ov_compiled([input_b1_np]),
            "fn_b8": lambda: ov_compiled([input_b8_np])
        }
    ]
    
    results = {}
    
    for rt in runtimes:
        name = rt["name"]
        print(f"Тестування: {name} ...")
        fn_b1 = rt["fn_b1"]
        fn_b8 = rt["fn_b8"]
        
        # --- Тест 1: Single-Image (Batch=1) ---
        # Прогрів
        for _ in range(warmup):
            _ = fn_b1()
            
        times_b1 = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = fn_b1()
            t1 = time.perf_counter()
            times_b1.append((t1 - t0) * 1000.0)
            
        # --- Тест 2: Batch Processing (Batch=8) ---
        for _ in range(warmup):
            _ = fn_b8()
            
        times_b8 = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = fn_b8()
            t1 = time.perf_counter()
            times_b8.append((t1 - t0) * 1000.0)
            
        mean_b1 = float(np.mean(times_b1))
        std_b1 = float(np.std(times_b1))
        median_b1 = float(np.median(times_b1))
        min_b1 = float(np.min(times_b1))
        max_b1 = float(np.max(times_b1))
        p95_b1 = float(np.percentile(times_b1, 95))
        fps_b1 = float(1000.0 / mean_b1)
        
        mean_b8 = float(np.mean(times_b8))
        throughput_b8 = float((8 * 1000.0) / mean_b8)
        
        print(f"  Single (B=1): {mean_b1:.2f} ± {std_b1:.2f} ms | Median: {median_b1:.2f} ms | P95: {p95_b1:.2f} ms | FPS: {fps_b1:.1f}")
        print(f"  Batch (B=8): Latency: {mean_b8:.2f} ms | Throughput: {throughput_b8:.1f} img/s\n")
        
        results[name] = {
            "b1_mean_ms": round(mean_b1, 2),
            "b1_std_ms": round(std_b1, 2),
            "b1_median_ms": round(median_b1, 2),
            "b1_min_ms": round(min_b1, 2),
            "b1_max_ms": round(max_b1, 2),
            "b1_p95_ms": round(p95_b1, 2),
            "b1_fps": round(fps_b1, 1),
            "b8_latency_ms": round(mean_b8, 2),
            "b8_throughput_fps": round(throughput_b8, 1)
        }
        
    # Додаємо розміри моделей
    results["model_sizes_mb"] = {
        "PyTorch (.pt)": round(os.path.getsize(pt_path) / (1024 * 1024), 2),
        "ONNX (.onnx)": round(os.path.getsize(onnx_path) / (1024 * 1024), 2)
    }
    
    # Збереження у JSON
    with open("lab04_deployment/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # Візуалізація результатів
    plot_runtimes_benchmark(results, "lab04_deployment/figures/runtimes_benchmark.png")


def plot_runtimes_benchmark(results, save_path):
    names = ["PyTorch (CPU)", "ONNX Runtime (CPU)", "Intel OpenVINO (CPU)"]
    latencies = [results[n]["b1_mean_ms"] for n in names]
    stds = [results[n]["b1_std_ms"] for n in names]
    fps_single = [results[n]["b1_fps"] for n in names]
    fps_batch = [results[n]["b8_throughput_fps"] for n in names]
    
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    
    colors = ["#ef4444", "#3b82f6", "#10b981"]
    
    # 1. Затримка (B=1)
    bars1 = axes[0].bar(names, latencies, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_ylabel("Затримка інференсу (мс)", fontsize=10)
    axes[0].set_title("Single-Image Latency (Batch=1, мс)", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(["PyTorch", "ONNX Runtime", "OpenVINO"], fontsize=9.5)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f} ms", ha="center", fontsize=9, fontweight="bold")
        
    # 2. Пропускна здатність B=1 (FPS)
    bars2 = axes[1].bar(names, fps_single, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_ylabel("Кадрів / сек (FPS)", fontsize=10)
    axes[1].set_title("Single-Image FPS (Batch=1)", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(["PyTorch", "ONNX Runtime", "OpenVINO"], fontsize=9.5)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 15, f"{yval:.0f}", ha="center", fontsize=9, fontweight="bold")
        
    # 3. Batch Throughput B=8 (img/s)
    bars3 = axes[2].bar(names, fps_batch, color=colors, alpha=0.85, edgecolor="black")
    axes[2].set_ylabel("Зображень / сек (Throughput)", fontsize=10)
    axes[2].set_title("Batch Throughput (Batch=8, img/s)", fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(["PyTorch", "ONNX Runtime", "OpenVINO"], fontsize=9.5)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, yval + 30, f"{yval:.0f}", ha="center", fontsize=9, fontweight="bold")
        
    plt.suptitle("Порівняльний бенчмарк середовищ розгортання: PyTorch проти ONNX Runtime проти OpenVINO", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Графік бенчмаркінгу збережено: {save_path}")


if __name__ == "__main__":
    run_deployment_benchmark(num_iterations=150, warmup=20)
