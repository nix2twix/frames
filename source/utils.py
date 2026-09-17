import os
import tempfile
import streamlit as st
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import numpy as np
import zipfile
import io
import pandas as pd
from source.visualisation import tensor_to_np01

def init_session():
    if "session_initialized" not in st.session_state:
        st.session_state.update({
            "imgs": None, "labels": None, "targets": None, "emb_targets": None,
            "custom_model": None, "_cached_model_name": None,
            "adv": None, "emb_before": None, "emb_after": None,
            "heatmaps": None, "metrics": None, "df": None,
            "asr": None, "elapsed": None,
            "accuracy_drop": None, "confidence_shift": None,
            "acc_clean": None, "acc_adv": None,
            "dataset_root": None, "batch_ready": False, "_batch_signature": None,
            "series_results": None, "series_param_name": None, "series_param_label": None,
            "session_initialized": True
        })

def export_metrics_to_csv(metrics: dict) -> bytes:
    """
    metrics: словарь {метрика: [значения по выборке]}
    Возвращает бинарный CSV для отдачи через download_button.
    """
    df = pd.DataFrame(metrics)
    return df.to_csv(index=False).encode("utf-8")

def export_heatmaps_to_zip(heatmaps, labels):
    """
    heatmaps: список PIL.Image
    labels: список или тензор с метками
    Возвращает готовый zip (bytes)
    """
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i, heat in enumerate(heatmaps): # heat имеет тип PIL.Image
            img_bytes = io.BytesIO()
            heat.save(img_bytes, format="PNG")
            class_label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
            zf.writestr(f"heatmap_{i}_class_{class_label}.png", img_bytes.getvalue())

    return zip_buf.getvalue()

def export_adversarial_to_zip(adv_batch, labels, tensor_to_np01):
    """
    adv_batch: tensor (B, 3, H, W) в диапазоне [0,1]
    labels: tensor (B,)
    tensor_to_np01: функция преобразования тензора в numpy HxWxC
    """
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i in range(len(adv_batch)):
            img = tensor_to_np01(adv_batch[i])           # -> float HWC
            img = (img * 255).astype("uint8")
            img_pil = Image.fromarray(img)

            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")

            zf.writestr(f"adv_{i}_class_{labels[i]}.png", img_bytes.getvalue())

    return zip_buf.getvalue()

def load_imagefolder_batch(folder_path, batch_size=8, shuffle=True, num_workers=0):
    """
    Load an ImageFolder and return one batch (images tensor, labels tensor, class_names list, sample_paths list)
    """
    transform = transforms.ToTensor()  # returns float [0,1]
    ds = datasets.ImageFolder(folder_path, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    images, labels = next(iter(loader))
    sample_paths = [p for p, _ in ds.samples[:len(images)]]
    return images, labels, ds.classes, sample_paths

def tensor_to_numpy_list(batch_tensor):
    """
    Convert batch tensor (N,3,H,W) to list of HxWx3 numpy float arrays in [0,1]
    """
    out = []
    for i in range(batch_tensor.size(0)):
        arr = batch_tensor[i].detach().cpu().permute(1,2,0).numpy()
        arr = np.clip(arr, 0.0, 1.0)
        out.append(arr)
    return out

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


def preprocessImage(image: Image.Image):
    """Convert PIL-Image to tensor [1,3,H,W]"""
    image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def postprocessTensor(tensor: torch.Tensor):
    """Convert tensor [1,3,H,W] to PIL.Image"""
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def save_uploaded_file(uploaded_file, suffix: str = None) -> str:
    """
    Сохраняет объект UploadedFile (из st.file_uploader) во временный файл.
    Возвращает путь к сохраненному файлу.
    """
    if suffix is None:
        # Извлекаем расширение из имени файла
        _, suffix = os.path.splitext(uploaded_file.name)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

def export_adversarial_dataset(adv_batch, orig_batch, labels,
                                attack_type, epsilon, mode,
                                sim_before, sim_after, success_arr,
                                output_dir: str):
    """
    Формирует структурированный датасет для дообучения.
    Структура:
      output_dir/
        images/
          adv_0000_class_X.png
          orig_0000_class_X.png
        metadata.csv
        metadata.json
    """
    import json
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    metadata = []
    for i in range(len(adv_batch)):
        class_label = int(labels[i]) if isinstance(labels[i], torch.Tensor) else labels[i]

        adv_np = tensor_to_np01(adv_batch[i])
        adv_img = Image.fromarray((adv_np * 255).astype("uint8"))
        adv_path = f"images/adv_{i:04d}_class_{class_label}.png"
        adv_img.save(os.path.join(output_dir, adv_path))

        orig_np = tensor_to_np01(orig_batch[i])
        orig_img = Image.fromarray((orig_np * 255).astype("uint8"))
        orig_path = f"images/orig_{i:04d}_class_{class_label}.png"
        orig_img.save(os.path.join(output_dir, orig_path))

        metadata.append({
            "id": i,
            "class": class_label,
            "adv_path": adv_path,
            "orig_path": orig_path,
            "attack_type": attack_type,
            "epsilon": float(epsilon),
            "mode": mode,
            "sim_before": float(sim_before[i]),
            "sim_after": float(sim_after[i]),
            "success": bool(success_arr[i]),
        })

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, output_dir)
                zf.write(full_path, arcname)

    return zip_buf.getvalue()

def run_attack_series(
    model,
    imgs,
    emb_targets,
    attack_type: str,
    mode: str,
    param_name: str,
    param_values: list,
    fixed_params: dict,
    show_progress=True
):
    """
    Универсальный прогон серии экспериментов: варьируется ОДИН параметр атаки,
    остальные фиксированы.

    Параметры:
      model        — модель-эмбеддер
      imgs         — батч изображений (N×C×H×W)
      emb_targets  — эмбеддинги целей (N×D)
      attack_type  — "FGSM" / "PGD" / "CW" / "PatchAttack"
      mode         — "Подмена личности" / "Обход аутентификации"
      param_name   — имя варьируемого параметра: 
                     "epsilon" / "alpha" / "iters" / "c" / "lr" / "patch_size"
      param_values — список значений параметра
      fixed_params — dict с остальными фиксированными параметрами:
                     {"epsilon": 0.08, "alpha": 0.01, "iters": 15, "c": 1e-2, 
                      "lr": 0.01, "patch_size": 32, "thr": 0.5}

    Возвращает:
      results — dict со списками метрик по каждому значению параметра
    """
    import time
    from stqdm import stqdm
    from source.attacks import fgsm_embed, pgd_embed, cw_embed, patch_attack_embed
    from source.metrics import (
        attack_success_rate, accuracy_drop, confidence_shift,
        linf_per_image, l2_per_image, cosine_similarity_batch,
    )

    emb_before = model.get_embeddings(imgs)
    sim_before = cosine_similarity_batch(emb_before, emb_targets)
    thr = fixed_params.get("thr", 0.5)
    a_mode = "impersonation" if mode == "Подмена личности" else "deauth"

    results = {
        "param": [], "ASR": [], "AD": [], "CS": [],
        "Linf_mean": [], "L2_mean": [], "Scos_mean": [], "time": [],
    }

    total = len(param_values)

    iterator = stqdm(range(total), desc=f"Прогон серии ({param_name})") if show_progress else range(total)

    for idx in iterator:
        value = param_values[idx]
        params = {**fixed_params, param_name: float(value)}
        t0 = time.time()

        if attack_type == "FGSM":
            adv = fgsm_embed(model, imgs, emb_targets, params["epsilon"], mode.lower(), show_progress=False)
        elif attack_type == "PGD":
            adv = pgd_embed(model, imgs, emb_targets,
                            params["epsilon"], params.get("alpha", 0.01),
                            int(params.get("iters", 15)), mode.lower(),
                            show_progress=False)
        elif attack_type == "Carlini-Wagner":
            adv = cw_embed(model, imgs, emb_targets,
                           c=params.get("c", 1e-2), lr=params.get("lr", 0.01),
                           iters=int(params.get("iters", 50)), mode=mode.lower(),
                           show_progress=False)
        elif attack_type == "PatchAttack":
            adv = patch_attack_embed(model, imgs, emb_targets,
                                     patch_size=int(params.get("patch_size", 32)),
                                     lr=params.get("lr", 0.01),
                                     iters=int(params.get("iters", 100)),
                                     mode=mode.lower(), show_progress=False)
        else:
            raise ValueError(f"Неизвестный тип атаки: {attack_type}")

        elapsed = time.time() - t0
        emb_after = model.get_embeddings(adv)
        sim_after = cosine_similarity_batch(emb_after, emb_targets)

        _, asr = attack_success_rate(emb_before, emb_after, emb_targets, threshold=thr, mode=a_mode)
        _, _, ad = accuracy_drop(sim_before, sim_after, threshold=thr)
        cs = confidence_shift(emb_before, emb_after, emb_targets, threshold=thr)
        linf = float(linf_per_image(imgs, adv).mean())
        l2 = float(l2_per_image(imgs, adv).mean())
        scos = float(cosine_similarity_batch(emb_before, emb_after).mean())

        results["param"].append(float(value))
        results["ASR"].append(float(asr))
        results["AD"].append(float(ad))
        results["CS"].append(float(cs))
        results["Linf_mean"].append(linf)
        results["L2_mean"].append(l2)
        results["Scos_mean"].append(scos)
        results["time"].append(elapsed)

        if show_progress:
            try:
                iterator.set_description(f"Прогон серии ({param_name}={value:.4g})")
            except AttributeError:
                pass

    return results

def export_metrics_to_html(metrics, model_name, attack_type, asr, ad, cs, elapsed, params=None):
    df = pd.DataFrame(metrics)

    params_html = ""
    if params:
        rows = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in params.items()
        )
        params_html = f"""
        <h2>Параметры атаки</h2>
        <table style="width:auto;">
            {rows}
        </table>
        """
    
    html = f"""<!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>FRAMES Report</title>
    <style>
    body{{font-family:Arial,sans-serif;margin:40px;}}
    h1{{color:#2c3e50;}}
    table{{border-collapse:collapse;width:100%;margin-top:20px;}}
    th,td{{border:1px solid #ddd;padding:8px;text-align:center;}}
    th{{background:#3498db;color:white;}}
    .metric{{font-size:1.2em;font-weight:bold;color:#e74c3c;}}
    </style></head>
    <body>
    <h1>Отчёт FRAMES</h1>
    <p><b>Модель:</b> {model_name}</p>
    <p><b>Атака:</b> {attack_type}</p>
    <p><b>ASR:</b> <span class="metric">{asr*100:.1f}%</span></p>
    <p><b>Accuracy Drop:</b> {ad*100:.1f} п.п.</p>
    <p><b>Confidence Shift:</b> {cs:.3f}</p>
    <p><b>Время:</b> {elapsed:.2f} с</p>
    {params_html}
    <h2>Метрики по изображениям</h2>
    {df.to_html(index=False)}
    </body></html>"""
    return html.encode("utf-8")


def export_metrics_to_pdf(metrics, model_name, attack_type, asr, ad, cs, elapsed, params=None):
    """
    Генерирует PDF-отчет
    """
    from fpdf import FPDF
    import os
    import matplotlib

    mpl_fonts = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    font_regular = os.path.join(mpl_fonts, "DejaVuSans.ttf")
    font_bold = os.path.join(mpl_fonts, "DejaVuSans-Bold.ttf")

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_font("DejaVu", "", font_regular)
    pdf.add_font("DejaVu", "B", font_bold)
    pdf.add_page()

    # --- Заголовок ---
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, "FRAMES Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # --- Сводные метрики ---
    pdf.set_font("DejaVu", size=11)
    pdf.cell(0, 8, f"Модель: {model_name}   |   Атака: {attack_type}", 
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, 
             f"ASR: {asr*100:.1f}%   |   Accuracy Drop: {ad*100:.1f} п.п.   |   "
             f"Confidence Shift: {cs:.3f}", 
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Время выполнения: {elapsed:.2f} с", 
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # --- Параметры атаки ---
    if params:
        pdf.set_font("DejaVu", "B", 11)
        pdf.cell(0, 8, "Параметры атаки:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("DejaVu", size=10)
        for k, v in params.items():
            pdf.cell(0, 6, f"  {k}: {v}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # --- Таблица метрик ---
    pdf.set_font("DejaVu", "B", 8)
    columns = list(metrics.keys())
    col_width = min(270 / len(columns), 30)  

    for col in columns:
        pdf.cell(col_width, 7, str(col), border=1, align="C")
    pdf.ln()

    pdf.set_font("DejaVu", size=7)
    num_rows = len(metrics[columns[0]]) if columns else 0
    for i in range(num_rows):
        for col in columns:
            val = metrics[col][i]
            if isinstance(val, float):
                text = f"{val:.4f}"
            else:
                text = str(val)
            if len(text) > 12:
                text = text[:10] + ".."
            pdf.cell(col_width, 6, text, border=1, align="C")
        pdf.ln()
    return bytes(pdf.output())