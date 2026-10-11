import os
import sys
import time
import json
import psutil
import numpy as np
import torch
import onnxruntime as ort
import openvino as ov
import matplotlib.pyplot as plt

# Відносні шляхи
PYTORCH_MODEL_PATH = "models/best_cnn_model.pt"
ONNX_FP32_PATH = "models/simple_cnn.onnx"
ONNX_INT8_PATH = "models/simple_cnn_int8.onnx"
BENCHMARK_OUTPUT_PATH = "lab05_optimization/benchmark_results.json"
FIGURES_DIR = "lab05_optimization/figures"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab02_cnn.model import SimpleCNN


def get_current_rss_mb():
    """Повертає поточне споживання оперативної пам'яті (RSS RAM) процесом у МБ"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_pytorch(checkpoint_path=PYTORCH_MODEL_PATH, num_warmup=20, num_runs=150):
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # B=1 Latency
    input_b1 = torch.randn(1, 3, 64, 64, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_b1)
            
        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(input_b1)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
    # B=8 Throughput
    input_b8 = torch.randn(8, 3, 64, 64, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(num_warmup // 2):
            _ = model(input_b8)
            
        t0 = time.perf_counter()
        for _ in range(num_runs // 2):
            _ = model(input_b8)
        total_time = time.perf_counter() - t0
        fps_b8 = ( (num_runs // 2) * 8 ) / total_time

    return {
        "engine": "PyTorch (CPU)",
        "precision": "FP32",
        "device": "CPU",
        "file_size_mb": round(os.path.getsize(checkpoint_path) / (1024 * 1024), 3),
        "latency_b1_mean_ms": round(float(np.mean(latencies)), 3),
        "latency_b1_median_ms": round(float(np.median(latencies)), 3),
        "latency_b1_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_b1_std_ms": round(float(np.std(latencies)), 3),
        "fps_b1": round(1000.0 / float(np.mean(latencies)), 1),
        "fps_b8": round(float(fps_b8), 1),
        "rss_ram_mb": round(get_current_rss_mb(), 1)
    }


def benchmark_onnxruntime(model_path, precision_label="FP32", num_warmup=20, num_runs=150):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # B=1 Latency
    input_b1 = np.random.randn(1, 3, 64, 64).astype(np.float32)
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: input_b1})
        
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: input_b1})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    # B=8 Throughput
    input_b8 = np.random.randn(8, 3, 64, 64).astype(np.float32)
    for _ in range(num_warmup // 2):
        _ = session.run(None, {input_name: input_b8})
        
    t0 = time.perf_counter()
    for _ in range(num_runs // 2):
        _ = session.run(None, {input_name: input_b8})
    total_time = time.perf_counter() - t0
    fps_b8 = ( (num_runs // 2) * 8 ) / total_time

    return {
        "engine": f"ONNX Runtime ({precision_label})",
        "precision": precision_label,
        "device": "CPU",
        "file_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 3),
        "latency_b1_mean_ms": round(float(np.mean(latencies)), 3),
        "latency_b1_median_ms": round(float(np.median(latencies)), 3),
        "latency_b1_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_b1_std_ms": round(float(np.std(latencies)), 3),
        "fps_b1": round(1000.0 / float(np.mean(latencies)), 1),
        "fps_b8": round(float(fps_b8), 1),
        "rss_ram_mb": round(get_current_rss_mb(), 1)
    }


def benchmark_openvino(model_path, device_target="CPU", precision_label="FP32", num_warmup=20, num_runs=150):
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device_target)
    infer_request = compiled_model.create_infer_request()
    
    # B=1 Latency
    input_b1 = np.random.randn(1, 3, 64, 64).astype(np.float32)
    for _ in range(num_warmup):
        _ = infer_request.infer([input_b1])
        
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = infer_request.infer([input_b1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    # B=8 Throughput
    input_b8 = np.random.randn(8, 3, 64, 64).astype(np.float32)
    for _ in range(num_warmup // 2):
        _ = infer_request.infer([input_b8])
        
    t0 = time.perf_counter()
    for _ in range(num_runs // 2):
        _ = infer_request.infer([input_b8])
    total_time = time.perf_counter() - t0
    fps_b8 = ( (num_runs // 2) * 8 ) / total_time

    return {
        "engine": f"OpenVINO ({precision_label} {device_target})",
        "precision": precision_label,
        "device": device_target,
        "file_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 3),
        "latency_b1_mean_ms": round(float(np.mean(latencies)), 3),
        "latency_b1_median_ms": round(float(np.median(latencies)), 3),
        "latency_b1_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_b1_std_ms": round(float(np.std(latencies)), 3),
        "fps_b1": round(1000.0 / float(np.mean(latencies)), 1),
        "fps_b8": round(float(fps_b8), 1),
        "rss_ram_mb": round(get_current_rss_mb(), 1)
    }


def run_comprehensive_benchmark():
    print("\n" + "="*70)
    print(" 4. КОМПЛЕКСНИЙ БЕНЧМАРК ОПТИМІЗАЦІЇ ТА АПАРАТНИХ ПРИСКОРЮВАЧІВ ")
    print("="*70)
    
    results = []
    
    # 1. PyTorch Baseline
    print("1/6. Бенчмарк PyTorch (FP32 CPU)...")
    res_torch = benchmark_pytorch()
    results.append(res_torch)
    print(f"     Latency: {res_torch['latency_b1_mean_ms']:.3f} ms, Throughput B=8: {res_torch['fps_b8']:.1f} FPS")
    
    # 2. ONNX Runtime FP32
    print("2/6. Бенчмарк ONNX Runtime (FP32 CPU)...")
    res_ort_fp32 = benchmark_onnxruntime(ONNX_FP32_PATH, precision_label="FP32")
    results.append(res_ort_fp32)
    print(f"     Latency: {res_ort_fp32['latency_b1_mean_ms']:.3f} ms, Throughput B=8: {res_ort_fp32['fps_b8']:.1f} FPS")
    
    # 3. ONNX Runtime INT8
    print("3/6. Бенчмарк ONNX Runtime (INT8 CPU)...")
    res_ort_int8 = benchmark_onnxruntime(ONNX_INT8_PATH, precision_label="INT8")
    results.append(res_ort_int8)
    print(f"     Latency: {res_ort_int8['latency_b1_mean_ms']:.3f} ms, Throughput B=8: {res_ort_int8['fps_b8']:.1f} FPS")
    
    # 4. OpenVINO FP32 CPU
    print("4/6. Бенчмарк OpenVINO (FP32 CPU)...")
    res_ov_fp32 = benchmark_openvino(ONNX_FP32_PATH, device_target="CPU", precision_label="FP32")
    results.append(res_ov_fp32)
    print(f"     Latency: {res_ov_fp32['latency_b1_mean_ms']:.3f} ms, Throughput B=8: {res_ov_fp32['fps_b8']:.1f} FPS")
    
    # 5. OpenVINO INT8 CPU
    print("5/6. Бенчмарк OpenVINO (INT8 CPU)...")
    res_ov_int8 = benchmark_openvino(ONNX_INT8_PATH, device_target="CPU", precision_label="INT8")
    results.append(res_ov_int8)
    print(f"     Latency: {res_ov_int8['latency_b1_mean_ms']:.3f} ms, Throughput B=8: {res_ov_int8['fps_b8']:.1f} FPS")
    
    # 6. OpenVINO FP32 dGPU (NVIDIA RTX 4050)
    print("6/6. Бенчмарк OpenVINO (FP32 dGPU RTX 4050)...")
    try:
        res_ov_gpu = benchmark_openvino(ONNX_FP32_PATH, device_target="GPU.0", precision_label="FP32")
        results.append(res_ov_gpu)
        print(f"     Latency: {res_ov_gpu['latency_b1_mean_ms']:.3f} ms, Throughput B=8: {res_ov_gpu['fps_b8']:.1f} FPS")
    except Exception as e:
        print(f"     Помилка dGPU: {e}")
        
    # Додавання статусу апаратних прискорювачів
    hw_status = {
        "cpu": "AMD Ryzen 5 7640HS w/ Radeon 760M Graphics (6 cores, 12 threads)",
        "dgpu": "NVIDIA GeForce RTX 4050 Laptop GPU (Доступний через OpenVINO GPU.0)",
        "igpu": "AMD Radeon 760M Graphics (Не підтримується OpenVINO Intel-GPU драйвером)",
        "npu": "Відсутній в апаратній конфігурації системи"
    }
    
    benchmark_payload = {
        "hardware_environment": hw_status,
        "configurations": results
    }
    
    with open(BENCHMARK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[Збереження] Результати комплексного бенчмарку збережено: {BENCHMARK_OUTPUT_PATH}")
    
    # Побудова графіка
    plot_optimization_benchmark(results, os.path.join(FIGURES_DIR, "optimization_benchmark.png"))


def plot_optimization_benchmark(results, save_path):
    """
    Побудова зведеного порівняльного графіка бенчмарку (Latency, Throughput, Model Size)
    """
    labels = [r["engine"].replace(" (CPU)", "").replace(" (FP32 CPU)", " FP32").replace(" (INT8 CPU)", " INT8").replace(" (FP32 GPU.0)", " GPU") for r in results]
    latencies = [r["latency_b1_mean_ms"] for r in results]
    throughputs = [r["fps_b8"] for r in results]
    sizes = [r["file_size_mb"] for r in results]
    
    colors = ["#94a3b8", "#3b82f6", "#10b981", "#6366f1", "#059669", "#f59e0b"][:len(results)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. Latency B=1
    ax = axes[0]
    bars1 = ax.bar(labels, latencies, color=colors, edgecolor="black", width=0.6)
    ax.set_ylabel("Затримка B=1 (мс)", fontsize=10, fontweight="bold")
    ax.set_title("Inference Latency (B=1, менше - краще)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f} ms", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    # 2. Throughput B=8
    ax = axes[1]
    bars2 = ax.bar(labels, throughputs, color=colors, edgecolor="black", width=0.6)
    ax.set_ylabel("Пропускна здатність B=8 (FPS)", fontsize=10, fontweight="bold")
    ax.set_title("Throughput (B=8, більше - краще)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 15, f"{int(h)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    # 3. Model Size
    ax = axes[2]
    bars3 = ax.bar(labels, sizes, color=colors, edgecolor="black", width=0.6)
    ax.set_ylabel("Розмір моделі (MB)", fontsize=10, fontweight="bold")
    ax.set_title("Розмір файлу моделі (MB)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f} MB", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    plt.suptitle("Комплексне порівняння runtime-середовищ, рівнів точності та апаратних платформ", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Графік] Бенчмарк збережено: {save_path}")


if __name__ == "__main__":
    run_comprehensive_benchmark()
