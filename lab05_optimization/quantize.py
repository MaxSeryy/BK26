import os
import sys
import time
import json
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.quantization as oq
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Базові константи та відносні шляхи репозиторію
FP32_MODEL_PATH = "models/simple_cnn.onnx"
INT8_MODEL_PATH = "models/simple_cnn_int8.onnx"
MANIFEST_PATH = "data/dataset_split.json"
OUTPUT_METRICS_PATH = "lab05_optimization/quant_metrics.json"
FIGURES_DIR = "lab05_optimization/figures"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageCalibrationDataReader(oq.CalibrationDataReader):
    """
    Калібрувальний DataReader для ста pasoчного квантування INT8 (PTQ) в ONNX Runtime.
    Послідовно подає нормовані кадри з навчальної вибірки для розрахунку діапазонів активацій.
    """
    def __init__(self, manifest_path=MANIFEST_PATH, input_name="input", target_size=(64, 64), max_samples=60):
        super().__init__()
        self.input_name = input_name
        self.target_size = target_size
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        train_samples = manifest["splits"]["train"]
        if max_samples and max_samples < len(train_samples):
            # Рівномірний вибір з кожної категорії для репрезентативного калібрування
            class_groups = {}
            for item in train_samples:
                class_groups.setdefault(item["class"], []).append(item)
            selected = []
            per_class = max_samples // len(class_groups)
            for cls_name, items in class_groups.items():
                selected.extend(items[:per_class])
            self.samples = selected
        else:
            self.samples = train_samples
            
        self.data_iter = iter(self.samples)
        print(f"[CalibrationDataReader] Ініціалізовано {len(self.samples)} зразків для калібрування.")

    def get_next(self):
        sample = next(self.data_iter, None)
        if sample is None:
            return None
            
        img_path = sample["path"]
        bgr = cv2.imread(img_path)
        if bgr is None:
            return self.get_next()
            
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor_chw = normalized.transpose(2, 0, 1)
        batch_input = np.expand_dims(tensor_chw, axis=0).astype(np.float32)
        
        return {self.input_name: batch_input}


def run_post_training_quantization(fp32_model_path=FP32_MODEL_PATH,
                                    int8_model_path=INT8_MODEL_PATH,
                                    manifest_path=MANIFEST_PATH):
    """
    Виконує статичне пост-тренувальне квантування (Static PTQ INT8) графа ONNX.
    """
    os.makedirs(os.path.dirname(int8_model_path), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print(" 1. СТАТИЧНЕ ПОСТ-ТРЕНУВАЛЬНЕ КВАНТУВАННЯ (STATIC PTQ INT8) ")
    print("="*70)
    print(f"Вихідна FP32 модель: {fp32_model_path}")
    print(f"Цільова INT8 модель:  {int8_model_path}")
    
    # 1. Створення калібрувального читача
    calib_reader = ImageCalibrationDataReader(manifest_path=manifest_path, max_samples=60)
    
    # 2. Виконання статичного квантування
    t_start = time.perf_counter()
    oq.quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=calib_reader,
        quant_format=oq.QuantFormat.QDQ,
        per_channel=False,
        weight_type=oq.QuantType.QInt8,
        activation_type=oq.QuantType.QUInt8,
        calibrate_method=oq.CalibrationMethod.MinMax
    )
    t_quant = time.perf_counter() - t_start
    print(f"[Квантування] Успішно завершено за {t_quant:.2f} с.")
    
    # 3. Валідація створеної моделі
    onnx_int8 = onnx.load(int8_model_path)
    onnx.checker.check_model(onnx_int8)
    print("[Валідація] Граф квантованої моделі успішно пройшов onnx.checker.")
    
    # 4. Порівняння розмірів файлів
    size_fp32_bytes = os.path.getsize(fp32_model_path)
    size_int8_bytes = os.path.getsize(int8_model_path)
    size_fp32_mb = size_fp32_bytes / (1024 * 1024)
    size_int8_mb = size_int8_bytes / (1024 * 1024)
    compression_ratio = size_fp32_bytes / size_int8_bytes
    
    print(f"\n[Розмір моделі]")
    print(f"  FP32:  {size_fp32_mb:.3f} MB ({size_fp32_bytes:,} байт)")
    print(f"  INT8:  {size_int8_mb:.3f} MB ({size_int8_bytes:,} байт)")
    print(f"  Ступінь компресії: {compression_ratio:.2f}x (зменшення на {(1 - 1/compression_ratio)*100:.1f}%)")
    
    return {
        "size_fp32_bytes": size_fp32_bytes,
        "size_int8_bytes": size_int8_bytes,
        "size_fp32_mb": round(size_fp32_mb, 4),
        "size_int8_mb": round(size_int8_mb, 4),
        "compression_ratio": round(compression_ratio, 2),
        "quant_time_sec": round(t_quant, 2)
    }


def evaluate_model_on_test_set(model_path, manifest_path=MANIFEST_PATH):
    """
    Оцінює точність класифікації моделі на тестовій підвибірці (45 зображень).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    classes = manifest["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    test_samples = manifest["splits"]["test"]
    
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    y_true = []
    y_pred = []
    latencies = []
    
    for item in test_samples:
        img_path = item["path"]
        label_idx = class_to_idx[item["class"]]
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
            
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        inp = np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
        
        t0 = time.perf_counter()
        logits = session.run(None, {input_name: inp})[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        
        pred_idx = int(np.argmax(logits, axis=1)[0])
        y_true.append(label_idx)
        y_pred.append(pred_idx)
        latencies.append(dt_ms)
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred).tolist()
    mean_lat = float(np.mean(latencies))
    
    return {
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1), 4),
        "mean_latency_ms": round(mean_lat, 3),
        "confusion_matrix": cm,
        "classes": classes,
        "sample_count": len(y_true)
    }


def plot_quantization_comparison(fp32_eval, int8_eval, size_stats, save_path):
    """
    Побудова графіка порівняння метрик FP32 та INT8 моделей.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # 1. Розмір моделі на диску
    ax = axes[0]
    bars = ax.bar(["FP32", "INT8"], [size_stats["size_fp32_mb"], size_stats["size_int8_mb"]],
                  color=["#3b82f6", "#10b981"], width=0.55, edgecolor="black")
    ax.set_ylabel("Розмір файлу (MB)", fontsize=10, fontweight="bold")
    ax.set_title(f"Розмір на диску (Стиснення: {size_stats['compression_ratio']}x)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f} MB", ha="center", va="bottom", fontweight="bold")
        
    # 2. Точність та F1-Score
    ax = axes[1]
    x = np.arange(2)
    width = 0.35
    accs = [fp32_eval["accuracy"] * 100, int8_eval["accuracy"] * 100]
    f1s = [fp32_eval["f1_macro"] * 100, int8_eval["f1_macro"] * 100]
    
    b1 = ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#2563eb", edgecolor="black")
    b2 = ax.bar(x + width/2, f1s, width, label="F1-Macro (%)", color="#f59e0b", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(["FP32", "INT8"], fontweight="bold")
    ax.set_ylabel("Метрика якості (%)", fontsize=10, fontweight="bold")
    ax.set_title("Якість класифікації на Test (45 зразків)", fontsize=11, fontweight="bold")
    ax.set_ylim(80, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    # 3. Середня затримка інференсу на тест-сеті
    ax = axes[2]
    lats = [fp32_eval["mean_latency_ms"], int8_eval["mean_latency_ms"]]
    speedup = fp32_eval["mean_latency_ms"] / (int8_eval["mean_latency_ms"] + 1e-6)
    bars3 = ax.bar(["FP32", "INT8"], lats, color=["#ef4444", "#8b5cf6"], width=0.55, edgecolor="black")
    ax.set_ylabel("Затримка інференсу (мс)", fontsize=10, fontweight="bold")
    ax.set_title(f"Середня затримка (Прискорення: {speedup:.2f}x)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.2f} мс", ha="center", va="bottom", fontweight="bold")
        
    plt.suptitle("Порівняльний аналіз оптимізації моделі: FP32 проти INT8 (ONNX Runtime)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Графік] Порівняння оптимізації збережено: {save_path}")


def main():
    # 1. Запуск PTQ
    size_stats = run_post_training_quantization()
    
    # 2. Оцінка FP32 та INT8
    print("\n" + "="*70)
    print(" 2. ОЦІНКА ТОЧНОСТІ ТА МЕТРИК НА ТЕСТОВІЙ ВИБІРЦІ ")
    print("="*70)
    fp32_eval = evaluate_model_on_test_set(FP32_MODEL_PATH)
    int8_eval = evaluate_model_on_test_set(INT8_MODEL_PATH)
    
    print(f"FP32: Accuracy={fp32_eval['accuracy']*100:.2f}%, F1-Macro={fp32_eval['f1_macro']*100:.2f}%, Latency={fp32_eval['mean_latency_ms']:.2f} ms")
    print(f"INT8: Accuracy={int8_eval['accuracy']*100:.2f}%, F1-Macro={int8_eval['f1_macro']*100:.2f}%, Latency={int8_eval['mean_latency_ms']:.2f} ms")
    delta_acc = (int8_eval['accuracy'] - fp32_eval['accuracy']) * 100
    print(f"Дельта точності (INT8 - FP32): {delta_acc:+.2f}%")
    
    # 3. Збереження звіту в JSON
    summary = {
        "quantization_type": "Static Post-Training Quantization (PTQ)",
        "quant_format": "QDQ (Quantize-Dequantize)",
        "weight_type": "QInt8",
        "activation_type": "QUInt8",
        "calibration_samples": 60,
        "size_comparison": size_stats,
        "fp32_evaluation": fp32_eval,
        "int8_evaluation": int8_eval,
        "accuracy_delta_percent": round(delta_acc, 2)
    }
    
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Збереження] Метрики збережено у: {OUTPUT_METRICS_PATH}")
    
    # 4. Побудова графіків
    plot_quantization_comparison(fp32_eval, int8_eval, size_stats,
                                 os.path.join(FIGURES_DIR, "quantization_evaluation.png"))


if __name__ == "__main__":
    main()
