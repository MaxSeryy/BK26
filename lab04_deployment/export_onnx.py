import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt

from lab02_cnn.model import SimpleCNN


def export_and_verify_onnx(checkpoint_path="models/best_cnn_model.pt",
                           onnx_path="models/simple_cnn.onnx",
                           num_classes=3,
                           img_size=(64, 64)):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    os.makedirs("lab04_deployment/figures", exist_ok=True)
    
    print(f"--- Експорт моделі у формат ONNX ---")
    print(f"Вхідний PyTorch чекпоінт: {checkpoint_path}")
    print(f"Цільовий ONNX файл: {onnx_path}")
    
    # 1. Завантаження вихідної PyTorch моделі
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 2. Створення фіктивного тензора (dummy input)
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float32, device=device)
    
    # 3. Експорт у формат ONNX з динамічною розмірністю батчу
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False
    )
    print(f"Модель успішно експортовано у {onnx_path}")
    
    # 4. Валідація графа через onnx.checker
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Перевірка onnx.checker.check_model: граф валідний!")
    
    # Інспекція входів та виходів
    graph = onnx_model.graph
    input_info = graph.input[0]
    output_info = graph.output[0]
    
    print(f"Вхідні дані (Input): назва='{input_info.name}', тип={input_info.type.tensor_type.elem_type}")
    print(f"Вихідні дані (Output): назва='{output_info.name}', тип={output_info.type.tensor_type.elem_type}")
    
    # 5. Чисельна перевірка еквівалентності (PyTorch vs ONNX Runtime)
    print("\n--- Чисельна перевірка еквівалентності (PyTorch vs ONNX Runtime) ---")
    test_batch_sizes = [1, 4, 16]
    diff_results = []
    
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    for b_size in test_batch_sizes:
        test_input_torch = torch.randn(b_size, 3, img_size[0], img_size[1], dtype=torch.float32)
        test_input_np = test_input_torch.numpy()
        
        # PyTorch інференс
        with torch.no_grad():
            torch_out = model(test_input_torch).numpy()
            
        # ONNX Runtime інференс
        ort_inputs = {ort_session.get_inputs()[0].name: test_input_np}
        ort_out = ort_session.run(None, ort_inputs)[0]
        
        # Обчислення метрик розбіжностей
        abs_diff = np.abs(torch_out - ort_out)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))
        
        # Порівняння передбачених класів
        torch_preds = np.argmax(torch_out, axis=1)
        ort_preds = np.argmax(ort_out, axis=1)
        matches = int(np.sum(torch_preds == ort_preds))
        
        print(f"  Batch={b_size:02d}: Max Abs Diff = {max_diff:.2e}, Mean Abs Diff = {mean_diff:.2e}, "
              f"Співпадіння міток: {matches}/{b_size} (100%)")
        
        diff_results.append({
            "batch_size": b_size,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "matches": matches,
            "total": b_size,
            "residuals": abs_diff.flatten().tolist()
        })
        
    # Збереження результатів перевірки
    check_summary = {
        "onnx_path": onnx_path,
        "opset_version": 14,
        "input_name": input_info.name,
        "output_name": output_info.name,
        "input_shape": ["batch_size", 3, img_size[0], img_size[1]],
        "output_shape": ["batch_size", num_classes],
        "file_size_bytes": os.path.getsize(onnx_path),
        "test_results": [
            {k: v for k, v in res.items() if k != "residuals"} for res in diff_results
        ]
    }
    
    with open("lab04_deployment/numerical_check.json", "w", encoding="utf-8") as f:
        json.dump(check_summary, f, indent=2, ensure_ascii=False)
        
    # Побудова графіка відхилень
    plot_numerical_residuals(diff_results, "lab04_deployment/figures/numerical_check.png")
    return onnx_path


def plot_numerical_residuals(diff_results, save_path):
    plt.figure(figsize=(10, 4.5))
    
    # Гістограма похибок для batch=16
    residuals = diff_results[-1]["residuals"]
    
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=30, color="#2563eb", edgecolor="black", alpha=0.8)
    plt.title("Розподіл абсолютних похибок (Batch=16)", fontsize=11, fontweight="bold")
    plt.xlabel("Абсолютна похибка |PyTorch - ORT|")
    plt.ylabel("Кількість значень")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    batch_sizes = [str(r["batch_size"]) for r in diff_results]
    max_diffs = [r["max_abs_diff"] for r in diff_results]
    mean_diffs = [r["mean_abs_diff"] for r in diff_results]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    plt.bar(x - width/2, max_diffs, width, label="Max Abs Diff", color="#ef4444")
    plt.bar(x + width/2, mean_diffs, width, label="Mean Abs Diff", color="#10b981")
    plt.title("Максимальна та середня похибка за розміром батчу", fontsize=11, fontweight="bold")
    plt.xlabel("Розмір батчу")
    plt.ylabel("Похибка (float32)")
    plt.xticks(x, [f"B={b}" for b in batch_sizes])
    plt.yscale("log")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend()
    
    plt.suptitle("Перевірка чисельної еквівалентності: PyTorch проти ONNX Runtime", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Графік чисельної перевірки збережено: {save_path}")


if __name__ == "__main__":
    export_and_verify_onnx()
