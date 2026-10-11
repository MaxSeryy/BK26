import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline_flowchart(output_path="lab01_data/figures/flowchart_pipeline.png"):
    # Створюємо теку для вихідного файлу, якщо вона відсутня
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Збільшено ширину полотна для комфортного розміщення блоку "Ні"
    fig, ax = plt.subplots(figsize=(13.5, 14.5), dpi=300)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 14.5)
    ax.axis('off')
    
    # Відступ, який Matplotlib додає до FancyBboxPatch
    box_pad = 0.14

    def draw_box(x, y, w, h, text, bg_color="#FFFFFF", edge_color="#2B3A42", shape="round", text_color="#1A252C"):
        shadow_offset = 0.05
        if shape == "round":
            shadow = patches.FancyBboxPatch(
                (x - w/2 + shadow_offset, y - h/2 - shadow_offset), w, h,
                boxstyle=f"round,pad={box_pad},rounding_size=0.18",
                facecolor="#000000", edgecolor="none", alpha=0.08, zorder=2
            )
            box = patches.FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle=f"round,pad={box_pad},rounding_size=0.18",
                facecolor=bg_color, edgecolor=edge_color, linewidth=1.6, zorder=3
            )
            ax.add_patch(shadow)
            ax.add_patch(box)
        elif shape == "diamond":
            shadow_poly = patches.Polygon([
                [x + shadow_offset, y + h/2 - shadow_offset],
                [x + w/2 + shadow_offset, y - shadow_offset],
                [x + shadow_offset, y - h/2 - shadow_offset],
                [x - w/2 + shadow_offset, y - shadow_offset]
            ], closed=True, facecolor="#000000", edgecolor="none", alpha=0.08, zorder=2)
            diamond = patches.Polygon([
                [x, y + h/2],
                [x + w/2, y],
                [x, y - h/2],
                [x - w/2, y]
            ], closed=True, facecolor=bg_color, edgecolor=edge_color, linewidth=1.6, zorder=3)
            ax.add_patch(shadow_poly)
            ax.add_patch(diamond)

        ax.text(x, y, text, ha='center', va='center', fontsize=9.2, fontweight='600',
                color=text_color, linespacing=1.3, zorder=5)

    def draw_arrow(x1, y1, x2, y2, label="", label_pos=(0, 0), label_color="#C0392B"):
        # zorder=4 гарантує, що стрілка буде поверх фону блоків, але під текстом
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='-|>',
                color='#34495E',
                lw=1.6,
                mutation_scale=15
            ),
            zorder=4
        )
        if label:
            lx = (x1 + x2) / 2 + label_pos[0]
            ly = (y1 + y2) / 2 + label_pos[1]
            ax.text(lx, ly, label, fontsize=8.5, fontweight='bold', color=label_color,
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#D5DBDB", lw=0.8))

    # --- Координати блоків (Центральна вісь X = 6.5) ---
    
    # 1. Початок
    b1_x, b1_y, b1_w, b1_h = 6.5, 13.5, 5.5, 0.75
    draw_box(b1_x, b1_y, b1_w, b1_h, "Вхідний сирий датасет (зображення 3 класів)", bg_color="#EBF5FB", edge_color="#2980B9")
    
    # 2. Сканування
    b2_x, b2_y, b2_w, b2_h = 6.5, 11.9, 5.8, 0.75
    draw_box(b2_x, b2_y, b2_w, b2_h, "Сканування директорій та аналіз цілісності файлів", bg_color="#FEF9E7", edge_color="#D4AC0D")
    
    # 3. Рішення (Ромб)
    b3_x, b3_y, b3_w, b3_h = 6.5, 9.9, 4.2, 1.1
    draw_box(b3_x, b3_y, b3_w, b3_h, "Файл читабельний\nта валідний?", bg_color="#FFFDF7", edge_color="#D68910", shape