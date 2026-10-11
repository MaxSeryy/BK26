import os
import random
import numpy as np
import cv2

def create_synthetic_dataset(base_dir="data/dataset", samples_per_class=100, seed=13):
    random.seed(seed)
    np.random.seed(seed)
    
    classes = ["drone", "car", "pedestrian"]
    resolutions = [
        (640, 480),
        (800, 600),
        (320, 240),
        (500, 500),
        (720, 576),
        (640, 360)
    ]
    
    os.makedirs(base_dir, exist_ok=True)
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(1, samples_per_class + 1):
            w, h = random.choice(resolutions)
            # Create a background with varied natural tones (sky, road, ground)
            if cls == "drone":
                # Sky-like tones: light blues, greys, whites
                bg_color = np.array([random.randint(180, 240), random.randint(160, 220), random.randint(140, 200)], dtype=np.uint8)
            elif cls == "car":
                # Road/asphalt/urban tones: greys, dark browns
                bg_color = np.array([random.randint(80, 140), random.randint(80, 140), random.randint(80, 140)], dtype=np.uint8)
            else: # pedestrian
                # Sidewalk/park tones: greens, muted urban
                bg_color = np.array([random.randint(70, 150), random.randint(90, 170), random.randint(70, 130)], dtype=np.uint8)
                
            img = np.full((h, w, 3), bg_color, dtype=np.uint8)
            
            # Add subtle texture/gradient noise
            noise = np.random.randint(-25, 25, (h, w, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Draw distinctive object features
            cx, cy = w // 2 + random.randint(-w // 8, w // 8), h // 2 + random.randint(-h // 8, h // 8)
            
            if cls == "drone":
                # Quadcopter shape: central body + 4 arms + rotors
                body_color = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))
                cv2.circle(img, (cx, cy), int(min(w, h) * 0.08), body_color, -1)
                arm_len = int(min(w, h) * 0.18)
                cv2.line(img, (cx - arm_len, cy - arm_len), (cx + arm_len, cy + arm_len), (40, 40, 40), 4)
                cv2.line(img, (cx - arm_len, cy + arm_len), (cx + arm_len, cy - arm_len), (40, 40, 40), 4)
                # Rotors
                for dx, dy in [(-arm_len, -arm_len), (arm_len, arm_len), (-arm_len, arm_len), (arm_len, -arm_len)]:
                    cv2.ellipse(img, (cx + dx, cy + dy), (int(arm_len * 0.4), int(arm_len * 0.15)), random.randint(0, 180), 0, 360, (10, 10, 10), 2)
            
            elif cls == "car":
                # Car silhouette: chassis rectangle + cabin + wheels
                car_color = (random.randint(180, 240), random.randint(30, 80), random.randint(30, 80)) # red/blue/dark
                car_w = int(min(w, h) * 0.5)
                car_h = int(car_w * 0.35)
                # Chassis
                cv2.rectangle(img, (cx - car_w // 2, cy), (cx + car_w // 2, cy + car_h), car_color, -1)
                # Cabin
                cabin_pts = np.array([
                    [cx - car_w // 3, cy],
                    [cx - car_w // 5, cy - int(car_h * 0.7)],
                    [cx + car_w // 5, cy - int(car_h * 0.7)],
                    [cx + car_w // 3, cy]
                ], dtype=np.int32)
                cv2.fillPoly(img, [cabin_pts], (random.randint(100, 150), random.randint(180, 220), random.randint(200, 250)))
                # Wheels
                wheel_r = int(car_h * 0.28)
                cv2.circle(img, (cx - car_w // 3, cy + car_h), wheel_r, (20, 20, 20), -1)
                cv2.circle(img, (cx + car_w // 3, cy + car_h), wheel_r, (20, 20, 20), -1)
                
            else: # pedestrian
                # Human silhouette: head circle + torso rect + legs lines + arms
                human_color = (random.randint(30, 90), random.randint(30, 90), random.randint(120, 200))
                ped_h = int(min(w, h) * 0.5)
                head_r = int(ped_h * 0.1)
                cv2.circle(img, (cx, cy - ped_h // 2 + head_r), head_r, (210, 180, 140), -1)
                # Torso
                cv2.rectangle(img, (cx - head_r, cy - ped_h // 2 + 2 * head_r), (cx + head_r, cy + int(ped_h * 0.1)), human_color, -1)
                # Legs
                cv2.line(img, (cx - head_r // 2, cy + int(ped_h * 0.1)), (cx - head_r, cy + ped_h // 2), (40, 40, 70), 5)
                cv2.line(img, (cx + head_r // 2, cy + int(ped_h * 0.1)), (cx + head_r, cy + ped_h // 2), (40, 40, 70), 5)
                # Arms
                cv2.line(img, (cx - head_r, cy - int(ped_h * 0.15)), (cx - int(head_r * 2), cy + int(ped_h * 0.05)), human_color, 4)
                cv2.line(img, (cx + head_r, cy - int(ped_h * 0.15)), (cx + int(head_r * 2), cy + int(ped_h * 0.05)), human_color, 4)
                
            filename = f"{cls}_{i:03d}.jpg"
            file_path = os.path.join(cls_dir, filename)
            cv2.imwrite(file_path, img)

    # Insert 2 defective files to realistically test data validation
    # 1. Zero-byte file
    with open(os.path.join(base_dir, "car", "car_corrupted_empty.jpg"), "wb") as f:
        pass
        
    # 2. Corrupted bytes header
    with open(os.path.join(base_dir, "drone", "drone_corrupted_header.jpg"), "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_IMAGE_HEADER_DATA_123456789")

    print(f"Dataset generated at '{base_dir}': 3 classes, {samples_per_class} valid images each, plus 2 corrupted test files.")

if __name__ == "__main__":
    create_synthetic_dataset()
