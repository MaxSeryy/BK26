# Лабораторна робота №3: Попередньо навчені моделі та прикладний комп’ютерний зір

## Мета роботи
Опанувати використання попередньо навчених моделей (pre-trained models), застосування підходу Transfer Learning для класифікації прикладного набору даних, побудову конвеєра комп'ютерного зору в реальному часі на базі OpenCV, вимірювання затримок (Latency) і пропускної спроможності (FPS), а також проведення порівняльного бенчмаркінгу з базовою згортковою мережею з ЛР №2.

---

## Структура модуля
```text
lab03_computer_vision/
├── models.py                 # Адаптація MobileNetV3-Small (Feature Extraction) та завантажувач SimpleCNN
├── train_transfer.py         # Навчання адаптованої моделі методом Transfer Learning
├── pipeline.py               # Прикладний OpenCV-конвеєр комп'ютерного зору з HUD телеметрією
├── benchmark.py              # Порівняльний бенчмаркінг (Latency, FPS, параметри) SimpleCNN vs MobileNetV3
├── cv_flowchart.drawio       # Блок-схема прикладного конвеєра в форматі Draw.io
├── transfer_metrics.json     # Результати тестування моделі MobileNetV3 на контрольній вибірці
├── benchmark_results.json    # Числові результати бенчмаркінгу за 120 ітераціями
└── figures/                  # Графічні матеріали:
    ├── cv_flowchart.png             # Експортована блок-схема конвеєра (Draw.io)
    ├── transfer_learning_curves.png # Криві навчання та валідації MobileNetV3-Small
    ├── annotated_pipeline.png       # Приклади роботи OpenCV-конвеєра з оверлеєм HUD
    └── benchmark_comparison.png     # Порівняльні стовпчикові діаграми затримок та FPS
```

---

## Інструкція із запуску

### 1. Навчання моделі Transfer Learning
Запуск навчання адаптованої голови `MobileNetV3-Small` на цільовому датасеті (3 класи: `car`, `drone`, `pedestrian`):
```powershell
python -m lab03_computer_vision.train_transfer
```
Ваги найкращої моделі зберігаються у `models/best_transfer_mobilenet.pt`.

### 2. Запуск прикладного OpenCV-конвеєра
Запуск візуалізаційного конвеєра з профілюванням у реальному часі:
```powershell
python -m lab03_computer_vision.pipeline
```
Створює зразок кадру системи технічного зору у `lab03_computer_vision/figures/annotated_pipeline.png`.

### 3. Запуск порівняльного бенчмарку
Вимірювання затримок Preprocess, Inference, Postprocess та розрахунок FPS на 120 викликах після 15 прогрівних запусків:
```powershell
python -m lab03_computer_vision.benchmark
```
Формує підсумковий графік `lab03_computer_vision/figures/benchmark_comparison.png` та файл метрик `benchmark_results.json`.

---

## Підсумки порівняльного аналізу
| Параметр | SimpleCNN (ЛР №2) | MobileNetV3-Small (ЛР №3) |
| :--- | :---: | :---: |
| **Роздільна здатність** | $64 \times 64$ | $224 \times 224$ |
| **Параметри моделі** | 0.36 млн | 1.52 млн |
| **Час інференсу (Inference)** | **1.25 мс** | 7.77 мс |
| **Час передобробки (Prep)** | **0.36 мс** | 1.96 мс |
| **Повна затримка (Total)** | **1.64 мс** | 9.77 мс |
| **Пропускна здатність (FPS)** | **610.7 кадр/с** | 102.3 кадр/с |
| **Точність (Accuracy)** | 100.0% | 100.0% |
| **Узагальнююча здатність** | Базова (прості силуети) | Висока (ознаки ImageNet) |
