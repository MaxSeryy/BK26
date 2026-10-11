import os
import sys
import time
import json
import cv2
import numpy as np
import onnxruntime as ort


class StandaloneONNXClassifier:
    """
    Автономний інференс-модуль ONNX Runtime.
    Працює виключно на базі OpenCV, NumPy та onnxruntime без використання PyTorch.
    """
    def __init__(self, onnx_model_path="models/simple_cnn.onnx",
                 classes=None,
                 target_size=(64, 64)):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX модель не знайдено: {onnx_model_path}")
            
        self.onnx_model_path = onnx_model_path
        self.target_size = target_size
        self.classes = classes or ["car", "drone", "pedestrian"]
        
        # Ініціалізація ONNX Runtime сесії з оптимізаціями CPU
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        
        # Інспекція метаданих графа
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # ImageNet константи для стандартизації
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, frame_bgr):
        """Передобробка засобами OpenCV та NumPy"""
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        # Формування NCHW масиву
        tensor_np = np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
        t_prep_ms = (time.perf_counter() - t0) * 1000.0
        return tensor_np, t_prep_ms

    def predict(self, frame_bgr):
        """Повний цикл класифікації одного кадру"""
        tensor_np, t_prep_ms = self.preprocess(frame_bgr)
        
        # Виконання ONNX Runtime інференсу
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: tensor_np})
        t_inf_ms = (time.perf_counter() - t0) * 1000.0
        
        # NumPy Softmax постобробка
        logits = outputs[0][0]
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
    test_img = "data/dataset/car/car_001.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        
    print(f"--- Standalone ONNX Runtime Inference ---")
    if not os.path.exists(test_img):
        print(f"Файл {test_img} не знайдено.")
        sys.exit(1)
        
    classifier = StandaloneONNXClassifier()
    img = cv2.imread(test_img)
    result = classifier.predict(img)
    
    print(f"Зображення: {test_img}")
    print(f"Передбачений клас: {result['class']} (Впевненість: {result['confidence'] * 100:.2f}%)")
    print(f"Розподіл ймовірностей: {result['probabilities']}")
    print(f"Час Preprocessing: {result['latency_prep_ms']} мс")
    print(f"Час ONNX Inference: {result['latency_inf_ms']} мс")
    print(f"Загальний час: {result['total_latency_ms']} мс")
