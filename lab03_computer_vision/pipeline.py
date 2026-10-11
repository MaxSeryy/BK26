import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from lab03_computer_vision.models import get_transfer_model


class VisionInferencePipeline:
    """
    Прикладний OpenCV-конвеєр комп'ютерного зору:
    Отримання кадру -> Preprocessing -> Inference -> Postprocessing -> HUD Візуалізація.
    """
    def __init__(self, checkpoint_path="models/best_transfer_mobilenet.pt",
                 num_classes=3, target_size=(224, 224), device="cpu"):
        self.device = torch.device(device)
        self.target_size = target_size
        
        # Ініціалізація моделі
        self.model = get_transfer_model(num_classes=num_classes, freeze_backbone=False)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.classes = checkpoint.get("classes", ["car", "drone", "pedestrian"])
            print(f"Завантажено чекпоінт: {checkpoint_path}")
        else:
            self.classes = ["car", "drone", "pedestrian"]
            print(f"Попередження: Чекпоінт {checkpoint_path} не знайдено, використано базові ваги.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # ImageNet параметри нормалізації
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, frame_bgr):
        """Етап 1: Передобробка кадру (BGR -> RGB, Resize, Normalize, Chw Tensor)"""
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return tensor, t_ms

    def predict(self, tensor):
        """Етап 2: Прямий прохід моделі (Inference)"""
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return logits, t_ms

    def postprocess(self, logits):
        """Етап 3: Постобробка (Softmax, Confidence, Top-1 Class)"""
        t0 = time.perf_counter()
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        conf = float(probs[pred_idx])
        t_ms = (time.perf_counter() - t0) * 1000.0
        return pred_class, conf, probs, t_ms

    def annotate(self, frame_bgr, pred_class, conf, timing_info):
        """Етап 4: Нанесення графічного оверлею (HUD)"""
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        
        # 1. Прямокутник області детекції (Bounding Reticle)
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.12)
        box_color = (0, 220, 100) if conf > 0.85 else (0, 165, 255)
        cv2.rectangle(annotated, (margin_x, margin_y), (w - margin_x, h - margin_y), box_color, 2)
        
        # Кутові акцентні лінії для ефекту системи технічного зору
        corner_len = 16
        cv2.line(annotated, (margin_x, margin_y), (margin_x + corner_len, margin_y), box_color, 4)
        cv2.line(annotated, (margin_x, margin_y), (margin_x, margin_y + corner_len), box_color, 4)
        cv2.line(annotated, (w - margin_x, margin_y), (w - margin_x - corner_len, margin_y), box_color, 4)
        cv2.line(annotated, (w - margin_x, margin_y), (w - margin_x, margin_y + corner_len), box_color, 4)
        cv2.line(annotated, (margin_x, h - margin_y), (margin_x + corner_len, h - margin_y), box_color, 4)
        cv2.line(annotated, (margin_x, h - margin_y), (margin_x, h - margin_y - corner_len), box_color, 4)
        cv2.line(annotated, (w - margin_x, h - margin_y), (w - margin_x - corner_len, h - margin_y), box_color, 4)
        cv2.line(annotated, (w - margin_x, h - margin_y), (w - margin_x, h - margin_y - corner_len), box_color, 4)
        
        # 2. Верхній плашка-банер з класом та впевненістю
        banner_h = 42
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (25, 25, 25), -1)
        text_label = f"AI VISION: {pred_class.upper()} | Conf: {conf * 100:.1f}%"
        cv2.putText(annotated, text_label, (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 3. Нижня плашка телеметрії (Latency & FPS)
        cv2.rectangle(annotated, (0, h - 34), (w, h), (25, 25, 25), -1)
        telemetry_text = (f"Inf: {timing_info['t_inf']:.1f}ms | "
                          f"Prep: {timing_info['t_prep']:.1f}ms | "
                          f"Total: {timing_info['t_total']:.1f}ms | "
                          f"FPS: {timing_info['fps']:.1f}")
        cv2.putText(annotated, telemetry_text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 230), 1, cv2.LINE_AA)
        
        return annotated

    def process_frame(self, frame_bgr):
        """Повний цикл обробки одного кадру"""
        t_start = time.perf_counter()
        tensor, t_prep = self.preprocess(frame_bgr)
        logits, t_inf = self.predict(tensor)
        pred_class, conf, probs, t_post = self.postprocess(logits)
        t_total = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / t_total if t_total > 0 else 0.0
        
        timing_info = {
            "t_prep": t_prep,
            "t_inf": t_inf,
            "t_post": t_post,
            "t_total": t_total,
            "fps": fps
        }
        
        annotated_frame = self.annotate(frame_bgr, pred_class, conf, timing_info)
        return annotated_frame, pred_class, conf, timing_info


def generate_pipeline_figure(save_path="lab03_computer_vision/figures/annotated_pipeline.png"):
    """
    Генерує демонстраційну сітку з 3 кадрів різних класів з накладеним оверлеєм прикладного конвеєра.
    """
    pipeline = VisionInferencePipeline()
    sample_paths = [
        ("car", "data/dataset/car/car_001.jpg"),
        ("drone", "data/dataset/drone/drone_001.jpg"),
        ("pedestrian", "data/dataset/pedestrian/pedestrian_001.jpg")
    ]
    
    annotated_frames = []
    for cls_name, p in sample_paths:
        if os.path.exists(p):
            img = cv2.imread(p)
            ann, pred_cls, conf, timing = pipeline.process_frame(img)
            annotated_frames.append(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
        else:
            print(f"Зображення {p} не знайдено.")
            
    if len(annotated_frames) == 3:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
        titles = ["Розпізнавання авто (Car)", "Розпізнавання БПЛА (Drone)", "Розпізнавання пішохода (Pedestrian)"]
        for idx, ax in enumerate(axes):
            ax.imshow(annotated_frames[idx])
            ax.set_title(titles[idx], fontsize=11, fontweight="bold")
            ax.axis("off")
            
        plt.suptitle("Прикладний OpenCV-конвеєр комп'ютерного зору з візуалізацією телеметрії", fontsize=13, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Демонстраційний кадр конвеєра збережено: {save_path}")


if __name__ == "__main__":
    generate_pipeline_figure()
