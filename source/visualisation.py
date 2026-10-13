# source/visualisation.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter

def tensor_to_np01(img_tensor):
    """
    Convert single tensor CxHxW (or batch NxCxHxW) to numpy in [0,1], HxWx3.
    Keeps aspect ratio; returns HxWx3 float in [0,1].
    """
    if isinstance(img_tensor, torch.Tensor):
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        arr = img_tensor.detach().cpu().permute(1,2,0).numpy()
    else:
        arr = np.array(img_tensor)
    # ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.clip(arr, 0.0, 1.0)
    return arr

def _to_pil(arr01):
    """Convert HxWx3 float [0,1] -> PIL RGB"""
    arr = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def _from_pil(img_pil):
    """PIL -> HxWx3 float [0,1]"""
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[...,:3]
    arr = np.clip(arr, 0.0, 1.0)
    return arr

def make_diff_heatmap(orig_np, adv_np, cmap="inferno", vmax=None, blur_radius=0):
    """
    Тепловая карта интенсивности изменения пикселей (без легенды).
    Возвращает: PIL.Image RGB.
    """
    diff = np.mean(np.abs(adv_np - orig_np), axis=2)

    if vmax is None:
        vmax = diff.max() if diff.max() > 0 else 1.0

    diff_norm = np.clip(diff / vmax, 0.0, 1.0)

    if blur_radius > 0:
        tmp = Image.fromarray((diff_norm * 255).astype(np.uint8))
        tmp = tmp.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        diff_norm = np.asarray(tmp).astype(np.float32) / 255.0

    cmap_fn = plt.get_cmap(cmap)
    colored = cmap_fn(diff_norm)[:, :, :3]
    return Image.fromarray((colored * 255).astype(np.uint8))

def make_overlay(orig_np, heatmap_pil, alpha=0.6):
    """
    Overlay heatmap (PIL RGB) on original image (orig_np float [0,1]) with given alpha.
    Returns PIL.Image (RGB).
    """
    orig_pil = _to_pil(orig_np)
    # ensure same size
    if heatmap_pil.size != orig_pil.size:
        heatmap_pil = heatmap_pil.resize(orig_pil.size, resample=Image.BILINEAR)
    blended = Image.blend(orig_pil, heatmap_pil, alpha=alpha)
    return blended

def plot_results_grid(orig_batch, adv_batch, heatmaps=None, max_cols=4,
                       title="", eps=None, asr=None):
    """
    Визуализация: для каждого изображения — три панели:
      - оригинал
      - искажённое (adversarial)
      - heatmap разности
    Плюс общий заголовок с параметрами эксперимента.
    """
    N = orig_batch.shape[0]
    items = min(N, max_cols)

    # Считаем макет: items строк, 3 колонки
    fig, axes = plt.subplots(items, 3, figsize=(9, items * 2.6),
                              facecolor="white",
                              gridspec_kw={"wspace": 0.02, "hspace": 0.15})
    if items == 1:
        axes = axes.reshape(1, -1)

    for i in range(items):
        orig_np = tensor_to_np01(orig_batch[i])
        adv_np  = tensor_to_np01(adv_batch[i])

        # Оригинал — без размытия, чтобы видны были детали
        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_title("Оригинал", fontsize=9)
        axes[i, 0].axis("off")

        # Adversarial — без размытия
        axes[i, 1].imshow(adv_np)
        axes[i, 1].set_title("Искажённое", fontsize=9)
        axes[i, 1].axis("off")

        # Heatmap
        if heatmaps is not None:
            axes[i, 2].imshow(heatmaps[i])
            axes[i, 2].set_title("Разность", fontsize=9)
            axes[i, 2].axis("off")

    # Общий заголовок
    subtitle = title
    if eps is not None:
        subtitle += f"  |  ε = {eps}"
    if asr is not None:
        subtitle += f"  |  ASR = {asr*100:.1f}%"
    if subtitle:
        fig.suptitle(subtitle, fontsize=11, y=0.995)

    plt.tight_layout()
    return fig

def draw_landmarks_on_image(img_np, landmarks, color=(0, 255, 0, 128), radius=2):
    """
    Рисует ключевые точки на изображении.
    
    img_np:     HxWx3 float [0,1] или uint8 [0,255]
    landmarks:  dict {name: (x, y)} — как в metrics._extract_landmarks
    color:      RGB цвет точек
    radius:     радиус точки в пикселях
    
    Возвращает: HxWx3 float [0,1] с нарисованными точками.
    """
    import numpy as np
    from PIL import Image, ImageDraw
    
    if img_np.dtype != np.uint8:
        img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = img_np.copy()
    
    img_pil = Image.fromarray(img_u8)
    draw = ImageDraw.Draw(img_pil)
    
    for name, (x, y) in landmarks.items():
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(0, 0, 0, 128),
            width=1,
        )
    
    return np.array(img_pil).astype(np.float32) / 255.0

def draw_face_mesh_on_image(
    img_np,
    landmarks,
    color=(0, 255, 0, 100),
    line_color=(0, 255, 0, 60),
    radius=2,
    line_width=1,
):
    """
    Рисует ключевые точки и соединяющие линии (сетку) на изображении.
    
    img_np:      HxWx3 float [0,1] или uint8 [0,255]
    landmarks:   dict {name: (x, y)} — как в metrics._extract_landmarks
    color:       RGBA цвет точек (с прозрачностью)
    line_color:  RGBA цвет линий
    radius:      радиус точки в пикселях
    line_width:  толщина линии в пикселях
    
    Возвращает: HxWx3 float [0,1] с нарисованными точками и сеткой.
    """
    import numpy as np
    from PIL import Image, ImageDraw
    
    # Приведение к RGBA-uint8
    if img_np.dtype != np.uint8:
        img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = img_np.copy()
    
    # Работаем в RGBA для прозрачности
    img_pil = Image.fromarray(img_u8).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # --- Компактный каркас геометрии лица ---
    # Связи заданы явно, чтобы визуализация повторяла схему
    # геометрических дескрипторов, а не превращалась в плотную mesh-сетку.
    edges = [
        # Верхняя часть лица
        ("forehead", "left_eye"),
        ("forehead", "right_eye"),
        ("forehead", "nose"),

        # Линия глаз и боковые границы
        ("left_eye", "right_eye"),
        ("left_eye", "cheek_left"),
        ("right_eye", "cheek_right"),
        ("left_eye", "nose"),
        ("right_eye", "nose"),

        # Средняя часть лица
        ("cheek_left", "nose"),
        ("cheek_right", "nose"),
        ("nose", "mouth_left"),
        ("nose", "mouth_right"),

        # Рот
        ("mouth_left", "mouth_right"),
        ("cheek_left", "mouth_left"),
        ("cheek_right", "mouth_right"),

        # Нижняя часть лица
        ("mouth_left", "chin"),
        ("mouth_right", "chin"),
        ("cheek_left", "chin"),
        ("cheek_right", "chin"),
        ("nose", "chin"),
    ]
    
    # --- Рисуем пунктирные линии ---
    def _draw_dashed_line(x1, y1, x2, y2, dash=4, gap=3):
        dx = x2 - x1
        dy = y2 - y1
        length = float((dx * dx + dy * dy) ** 0.5)
        if length <= 0:
            return

        ux = dx / length
        uy = dy / length
        pos = 0.0

        while pos < length:
            end = min(pos + dash, length)
            draw.line(
                [
                    (x1 + ux * pos, y1 + uy * pos),
                    (x1 + ux * end, y1 + uy * end),
                ],
                fill=line_color,
                width=line_width,
            )
            pos += dash + gap

    for a, b in edges:
        if a in landmarks and b in landmarks:
            x1, y1 = landmarks[a]
            x2, y2 = landmarks[b]
            _draw_dashed_line(x1, y1, x2, y2)
    
    # --- Рисуем точки ---
    for name, (x, y) in landmarks.items():
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(0, 0, 0, 80),
            width=1,
        )
    
    # --- Накладываем overlay на оригинал ---
    out = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return np.array(out).astype(np.float32) / 255.0