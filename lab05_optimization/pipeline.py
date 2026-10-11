import os
import sys
import time
import json
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# Відносні шляхи
FP32_MODEL_PATH = "models/simple_cnn.onnx"
INT8_MODEL_PATH = "models/simple_cnn_int8.onnx"
MANIFEST_PATH = "data/dataset_split.json"
PROFILE_OUTPUT_PATH = "lab05_optimization/pipeline_profile.json"
FIGURES_DIR = "lab05_optimization/figures"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class IntegratedVisionPipeline:
    """
    Інтегрований AI-конвеєр комп'ютерного зору:
    Джерело даних -> Preprocessing -> Inference (FP32/INT8) -> Postprocessing -> Візуалізація результату.
    """
    def __init__(self, model_path=INT8_MODEL_PATH, target_size=(64, 64), num_threads=4):
        self.model_path = model_path
        self.target_size = target_size
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        # Завантаження класів
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            self.classes = json.load(f)["classes"]
            
        # Прогрів сесії
        dummy = np.zeros((1, 3, target_size[0], target_size[1]), dtype=np.float32)
        for _ in range(5):
            self.session.run(None, {self.input_name: dummy})

    def acquire_frame(self, image_path):
        """Етап 1: Отримання та декодування кадру з диска"""
        t0 = time.perf_counter()
        frame = cv2.imread(image_path)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return frame, t_ms

    def preprocess(self, frame_bgr):
        """Етап 2: Масштабування, перетворення кольорів, нормалізація ImageNet, NCHW"""
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor_chw = normalized.transpose(2, 0, 1)
        batch_tensor = np.expand_dims(tensor_chw, axis=0).astype(np.float32)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return batch_tensor, t_ms

    def infer(self, input_tensor):
        """Етап 3: Виконання прямого проходу моделі (Inference)"""
        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: input_tensor})[0]
        t_ms = (time.perf_counter() - t0) * 1000.0
        return logits, t_ms

    def postprocess(self, logits):
        """Етап 4: Softmax, пошук аргмаксимуму, обчислення впевненості"""
        t0 = time.perf_counter()
        # Стабільний софтмакс
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        pred_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0, pred_idx])
        pred_label = self.classes[pred_idx]
        t_ms = (time.perf_counter() - t0) * 1000.0
        return pred_label, confidence, probs[0], t_ms

    def render_hud(self, frame_bgr, pred_label, confidence, timings_ms):
        """Етап 5: Рендеринг інформаційного оверлею (HUD)"""
        t0 = time.perf_counter()
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        
        # Напівпрозора плашка телеметрії зверху
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        
        t_total = sum(timings_ms.values())
        fps = 1000.0 / (t_total + 1e-6)
        
        # Відображення результату класифікації
        title = f"CLASS: {pred_label.upper()} ({confidence*100:.1f}%)"
        cv2.putText(annotated, title, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
                    
        # Відображення телеметрії затримок
        telemetry = (f"Decode: {timings_ms['acquisition']:.2f}ms | "
                     f"Prep: {timings_ms['preprocess']:.2f}ms | "
                     f"Infer: {timings_ms['inference']:.2f}ms | "
                     f"Post: {timings_ms['postprocess']:.2f}ms | "
                     f"FPS: {fps:.0f}")
        cv2.putText(annotated, telemetry, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
                    
        t_ms = (time.perf_counter() - t0) * 1000.0
        return annotated, t_ms

    def process_image(self, image_path):
        """Повний цикл обробки одного кадру з таймінгом усіх етапів"""
        t_start = time.perf_counter()
        
        frame, t_acq = self.acquire_frame(image_path)
        if frame is None:
            raise FileNotFoundError(f"Неможливо відкрити зображення: {image_path}")
            
        inp_tensor, t_prep = self.preprocess(frame)
        logits, t_inf = self.infer(inp_tensor)
        label, conf, probs, t_post = self.postprocess(logits)
        
        stage_timings = {
            "acquisition": t_acq,
            "preprocess": t_prep,
            "inference": t_inf,
            "postprocess": t_post
        }
        
        annotated_frame, t_render = self.render_hud(frame, label, conf, stage_timings)
        stage_timings["render"] = t_render
        stage_timings["total"] = (time.perf_counter() - t_start) * 1000.0
        
        return {
            "label": label,
            "confidence": conf,
            "probabilities": probs.tolist(),
            "timings_ms": stage_timings,
            "annotated_frame": annotated_frame
        }


def profile_pipeline(num_runs=100):
    """
    Детальне профілювання усіх етапів конвеєра для FP32 та INT8 конфігурацій.
    """
    print("\n" + "="*70)
    print(" 3. ПРОФІЛЮВАННЯ ПОВНОГО AI-КОНВЕЄРА (FP32 vs INT8) ")
    print("="*70)
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    test_samples = manifest["splits"]["test"]
    
    results = {}
    sample_annotated_img = None
    
    for mode_name, model_path in [("FP32", FP32_MODEL_PATH), ("INT8", INT8_MODEL_PATH)]:
        print(f"\n--- Профілювання конфігурації: {mode_name} ({model_path}) ---")
        pipeline = IntegratedVisionPipeline(model_path=model_path)
        
        # Прогрів
        for _ in range(10):
            sample_path = test_samples[0]["path"]
            _ = pipeline.process_image(sample_path)
            
        stage_records = {
            "acquisition": [],
            "preprocess": [],
            "inference": [],
            "postprocess": [],
            "render": [],
            "total": []
        }
        
        for i in range(num_runs):
            sample = test_samples[i % len(test_samples)]
            res = pipeline.process_image(sample["path"])
            for k, v in res["timings_ms"].items():
                stage_records[k].append(v)
            if mode_name == "INT8" and i == 0:
                sample_annotated_img = res["annotated_frame"]
                
        # Розрахунок середніх та часток
        stats = {}
        t_total_mean = float(np.mean(stage_records["total"]))
        
        for k, vals in stage_records.items():
            mean_val = float(np.mean(vals))
            stats[k] = {
                "mean_ms": round(mean_val, 3),
                "std_ms": round(float(np.std(vals)), 3),
                "p95_ms": round(float(np.percentile(vals, 95)), 3),
                "percentage": round((mean_val / t_total_mean) * 100.0, 1) if k != "total" else 100.0
            }
            
        # Визначення вузького місця (bottleneck)
        stages_only = {k: v["mean_ms"] for k, v in stats.items() if k != "total"}
        bottleneck = max(stages_only.items(), key=lambda x: x[1])
        
        print(f"Статистика етапів ({mode_name}):")
        for k, v in stages_only.items():
            pct = stats[k]["percentage"]
            print(f"  {k:12s}: {v:6.3f} мс ({pct:5.1f}%)")
        print(f"  Всього (Total): {stats['total']['mean_ms']:6.3f} мс (FPS: {1000.0/stats['total']['mean_ms']:.1f})")
        print(f"  --> ВУЗЬКЕ МІСЦЕ (Bottleneck): {bottleneck[0].upper()} ({bottleneck[1]:.3f} мс, {stats[bottleneck[0]]['percentage']}%)")
        
        results[mode_name] = {
            "model_path": model_path,
            "stages": stats,
            "bottleneck_stage": bottleneck[0],
            "bottleneck_time_ms": round(bottleneck[1], 3),
            "bottleneck_percentage": stats[bottleneck[0]]["percentage"],
            "pipeline_fps": round(1000.0 / stats["total"]["mean_ms"], 1)
        }
        
    # Збереження результатів профілювання в JSON
    with open(PROFILE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Профілювання] Результати збережено: {PROFILE_OUTPUT_PATH}")
    
    # Збереження прикладу анотованого кадру
    annotated_out_path = os.path.join(FIGURES_DIR, "pipeline_annotated.png")
    cv2.imwrite(annotated_out_path, sample_annotated_img)
    print(f"[Кадр] Приклад анотованого кадру збережено: {annotated_out_path}")
    
    # Побудова графіка розподілу часу по етапах
    plot_pipeline_breakdown(results, os.path.join(FIGURES_DIR, "pipeline_breakdown.png"))


def plot_pipeline_breakdown(results, save_path):
    """
    Побудова графіка декомпозиції затримок конвеєра (Breakdown bar chart & percentage)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    stages = ["acquisition", "preprocess", "inference", "postprocess", "render"]
    stage_labels = ["Image Decode", "Preprocessing", "Inference", "Postprocessing", "HUD Render"]
    colors = ["#64748b", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
    
    modes = ["FP32", "INT8"]
    
    # 1. Абсолютний час за етапами (мс)
    x = np.arange(len(stages))
    width = 0.35
    
    fp32_times = [results["FP32"]["stages"][s]["mean_ms"] for s in stages]
    int8_times = [results["INT8"]["stages"][s]["mean_ms"] for s in stages]
    
    b1 = ax1.bar(x - width/2, fp32_times, width, label="FP32 Pipeline", color="#3b82f6", edgecolor="black")
    b2 = ax1.bar(x + width/2, int8_times, width, label="INT8 Pipeline", color="#10b981", edgecolor="black")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_labels, rotation=15, fontweight="bold", fontsize=9)
    ax1.set_ylabel("Час виконання (мс)", fontsize=10, fontweight="bold")
    ax1.set_title("Затримка окремих етапів конвеєра (мс)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend()
    
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        
    # 2. Складений стек (Stacked Bar) для частки кожного етапу у %
    bottom_fp32 = 0
    bottom_int8 = 0
    bar_width = 0.5
    
    for i, s in enumerate(stages):
        pct_fp32 = results["FP32"]["stages"][s]["percentage"]
        pct_int8 = results["INT8"]["stages"][s]["percentage"]
        
        ax2.bar(["FP32", "INT8"], [pct_fp32, pct_int8], bar_width,
                bottom=[bottom_fp32, bottom_int8], label=stage_labels[i], color=colors[i], edgecolor="black")
                
        # Підписи часток
        if pct_fp32 > 8:
            ax2.text(0, bottom_fp32 + pct_fp32/2, f"{pct_fp32:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if pct_int8 > 8:
            ax2.text(1, bottom_int8 + pct_int8/2, f"{pct_int8:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            
        bottom_fp32 += pct_fp32
        bottom_int8 += pct_int8
        
    ax2.set_ylabel("Частка у загальному часі (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Розподіл часу між компонентами конвеєра (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    
    plt.suptitle("Профілювання повного конвеєра та виявлення вузьких місць", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Графік] Розподіл етапів збережено: {save_path}")


if __name__ == "__main__":
    profile_pipeline(num_runs=100)
