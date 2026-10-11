import os
import json
import random
import numpy as np
import cv2

class VisionDatasetLoader:
    """
    Відтворюваний модуль завантаження та пакетної генерації даних
    для моделей комп'ютерного зору (призначений для використання в ЛР №2).
    """
    def __init__(self, manifest_path="data/dataset_split.json", split="train",
                 target_size=(224, 224), normalize=True, standardize=True,
                 augment=False, random_seed=13):
        self.manifest_path = manifest_path
        self.split = split
        self.target_size = target_size
        self.normalize = normalize
        self.standardize = standardize
        self.augment = augment
        self.random_seed = random_seed
        
        # ImageNet mean & std for standardization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Маніфест розбиття не знайдено за шляхом: {manifest_path}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
            
        self.classes = self.manifest["classes"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        if split not in self.manifest["splits"]:
            raise ValueError(f"Невідома підвибірка '{split}'. Доступні: {list(self.manifest['splits'].keys())}")
            
        self.items = self.manifest["splits"][split]
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def __len__(self):
        return len(self.items)

    def _letterbox_resize(self, img):
        target_w, target_h = self.target_size
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def _augment(self, img):
        # 1. Flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        # 2. Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
        # 3. Brightness/Contrast jitter
        if random.random() > 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.randint(-15, 15)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return img

    def get_sample(self, idx):
        item = self.items[idx]
        img_bgr = cv2.imread(item["path"])
        if img_bgr is None:
            raise IOError(f"Не вдалося зчитати файл: {item['path']}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = self._letterbox_resize(img_rgb)
        
        if self.augment:
            img_resized = self._augment(img_resized)
            
        tensor = img_resized.astype(np.float32)
        if self.normalize:
            tensor = tensor / 255.0
            if self.standardize:
                tensor = (tensor - self.mean) / self.std
                
        label_idx = self.class_to_idx[item["class"]]
        return tensor, label_idx

    def get_batches(self, batch_size=32, shuffle=True):
        indices = list(range(len(self.items)))
        if shuffle:
            random.shuffle(indices)
            
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_x, batch_y = [], []
            for idx in batch_idx:
                x, y = self.get_sample(idx)
                batch_x.append(x)
                batch_y.append(y)
            yield np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.int64)

if __name__ == "__main__":
    # Демонстрація роботи завантажувача
    print("Ініціалізація завантажувача даних для навчальної вибірки...")
    try:
        loader = VisionDatasetLoader(manifest_path="data/dataset_split.json", split="train", augment=True)
        print(f"Знайдено зразків: {len(loader)}, Кількість класів: {len(loader.classes)}")
        for x_batch, y_batch in loader.get_batches(batch_size=16):
            print(f"Отримано батч: X={x_batch.shape}, y={y_batch.shape}, діапазон значень: [{x_batch.min():.2f}, {x_batch.max():.2f}]")
            break
    except Exception as e:
        print(f"Попередження: {e}")
