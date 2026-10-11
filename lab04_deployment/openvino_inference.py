import os
import sys
import time
import json
import cv2
import numpy as np
import openvino as ov


class StandaloneOpenVINOClassifier:
    """
    Автономний інференс-модуль Intel OpenVINO.
    Завантажує ONNX-модель, оптимізує та компілює її під цільовий процесор.
    """
    def __init__(self, onnx_model_path="models/simple_cnn.onnx",
                 classes=None,
                 target_size=(64, 64),
                 device_name="CPU"):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX модель не знайдено: {onnx_model_path}")
            
        self.onnx_model_path = onnx_model_path
        self.target_size = target_size
        self.classes = classes or ["car", "drone", "pedestrian"]
        
        # Ініціалізація OpenVINO Core
        self.core = ov.Core()
        
        # Читання та компіляція моделі
        self.model = self.core.read_model(model=onnx_model_path)
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device_name)
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # ImageNet константи
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, frame_bgr):
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor_np = np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
        t_prep_ms = (time.perf_counter() - t0) * 1000.0
        return tensor_np, t_prep_ms

    def predict(self, frame_bgr):
        tensor_np, t_prep_ms = self.preprocess(frame_bgr)
        
        t0 = time.perf_counter()
        infer_result = self.compiled_model([tensor_np])
        logits = infer_result[self.output_layer][0]
        t_inf_ms = (time.perf_counter() - t0) * 1000.0
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        conf = float(probs[pred_idx])
        
        return {
            "class": pred_class,
            "confidence": conf,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "latency_prep_ms": round(t_prep_ms, 3),
            "latency_inf_ms": round(t_inf_ms, 3),
            "total_latency_ms": round(t_prep_ms + t_inf_ms, 3)
        }


if __name__ == "__main__":
    test_img = "data/dataset/pedestrian/pedestrian_001.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        
    print(f"--- Standalone OpenVINO Runtime Inference ---")
    if not os.path.exists(test_img):
        print(f"Файл {test_img} не знайдено.")
        sys.exit(1)
        
    classifier = StandaloneOpenVINOClassifier()
    img = cv2.imread(test_img)
    result = classifier.predict(img)
    
    print(f"Зображення: {test_img}")
    print(f"Передбачений клас: {result['class']} (Впевненість: {result['confidence'] * 100:.2f}%)")
    print(f"Розподіл ймовірностей: {result['probabilities']}")
    print(f"Час Preprocessing: {result['latency_prep_ms']} мс")
    print(f"Час OpenVINO Inference: {result['latency_inf_ms']} мс")
    print(f"Загальний час: {result['total_latency_ms']} мс")
