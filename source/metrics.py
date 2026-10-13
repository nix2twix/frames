# source/metrics.py
import numpy as np
import streamlit as st
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import torch
import torch.nn.functional as F

def l0_count_per_image(x: torch.Tensor, x_adv: torch.Tensor, tol: float = 1e-6):
    """
    Count number of pixels (spatial positions) changed for each image in batch.
    x, x_adv: tensors in shape (N, C, H, W), values in [0,1] or [0,255].
    Returns: numpy array shape (N,) with integer counts (number of changed pixels).
    """
    assert x.shape == x_adv.shape
    # bring to CPU numpy float
    x_np = x.detach().cpu().numpy()
    x_adv_np = x_adv.detach().cpu().numpy()
    N, C, H, W = x_np.shape
    counts = np.zeros(N, dtype=int)
    for i in range(N):
        a = x_np[i]
        b = x_adv_np[i]
        # compare per-pixel across channels
        mask = np.any(np.abs(a - b) > tol, axis=0)  # shape HxW
        counts[i] = int(np.sum(mask))
    return counts

def l2_per_image(x: torch.Tensor, x_adv: torch.Tensor):
    """
    L2 (root of squared deviation) per image.
    Returns numpy array (N,) of floats.
    """
    diff = (x - x_adv).detach().cpu().reshape(x.size(0), -1).numpy().astype(np.float64)
    l2 = np.sqrt(np.sum(diff**2, axis=1))
    return l2

def linf_per_image(x: torch.Tensor, x_adv: torch.Tensor):
    """
    L_inf per image (max absolute perturbation across all channels/pixels).
    """
    diff = np.abs((x - x_adv).detach().cpu().numpy()).reshape(x.size(0), -1)
    return diff.max(axis=1)

def cosine_similarity_batch(emb1: torch.Tensor, emb2: torch.Tensor):
    """
    emb1, emb2: torch tensors shape (N, D)
    returns numpy array (N,) of cosine similarities
    """
    emb1 = emb1.detach().cpu()
    emb2 = emb2.detach().cpu()
    emb1_n = F.normalize(emb1, dim=1)
    emb2_n = F.normalize(emb2, dim=1)
    sims = (emb1_n * emb2_n).sum(dim=1).cpu().numpy()
    return sims

def attack_success_rate(emb_before: torch.Tensor, emb_after: torch.Tensor, emb_target: torch.Tensor,
                        threshold: float = 0.5, mode: str = 'impersonation'):
    """
    Compute attack success per-sample and overall ASR.
    - emb_before, emb_after: shape (N, D)
    - emb_target: shape (N, D) OR (D,) (if one target for all)
    mode:
      - 'impersonation': target similarity crosses the threshold from below
      - 'deauth': self-similarity crosses the threshold from above
    Returns: (success_array_bool(N,), ASR_float)
    """
    if emb_target.ndim == 1:
        emb_target = emb_target.unsqueeze(0).repeat(emb_before.size(0), 1)

    sim_before = cosine_similarity_batch(emb_before, emb_target)
    sim_after  = cosine_similarity_batch(emb_after, emb_target)

    mode_lower = str(mode).lower()
    if mode_lower in ("impersonation", "подмена личности"):
        # A genuine source-target pair must not already satisfy the target
        # threshold before the attack. Success means crossing it after attack.
        success = (sim_before < threshold) & (sim_after >= threshold)
    elif mode_lower in ("deauth", "обход аутентификации"):
        # The source must be accepted before the attack and rejected after it.
        success = (sim_before >= threshold) & (sim_after < threshold)
    else:
        raise ValueError("Unknown mode")

    asr = float(success.sum() / success.size)
    return success, asr

def robustness_score(success_arr):
    """
    Robustness = 1 - ASR.
    
    success_arr: (N,) numpy массив bool — успешность атак
    Возвращает: float в [0, 1] — доля отражённых атак.
    """
    asr = float(success_arr.sum() / success_arr.size)
    return 1.0 - asr

def delta_scos(emb_before, emb_after):
    """ΔScos = 1 - cosine_similarity(emb_before, emb_after)."""
    sim = cosine_similarity_batch(emb_before, emb_after)
    return 1.0 - sim


def predict_identity(emb_query, emb_gallery, labels_gallery):
    """
    Определяет, к какой личности из gallery ближе всего query-эмбеддинг.
    
    emb_query:   (N, D) — эмбеддинги query-изображений
    emb_gallery: (M, D) — эталонные эмбеддинги
    labels_gallery: (M,) — метки классов галереи
    
    Возвращает:
      predicted_labels: (N,) numpy array предсказанных классов
      best_sims:        (N,) numpy array косинусных сходств с лучшим совпадением
    """
    emb_q = F.normalize(emb_query.detach().cpu(), dim=1)
    emb_g = F.normalize(emb_gallery.detach().cpu(), dim=1)
    
    # sim[i, j] = сходство query i с gallery j
    sim = emb_q @ emb_g.T  # (N, M)
    
    best_idx = sim.argmax(dim=1).numpy()          # (N,)
    best_sims = sim.max(dim=1).values.numpy()     # (N,)
    labels_np = labels_gallery.detach().cpu().numpy()
    predicted = labels_np[best_idx]
    
    return predicted, best_sims

# Nine compact geometric landmarks used both for GD and visualization.
# MediaPipe indices are selected from the same landmark model as the attack,
# but this compact set is intentionally separate from the 74 attack controls.
GEOMETRIC_LANDMARK_INDICES = {
    "left_eye": 33,
    "right_eye": 263,
    "nose": 1,
    "mouth_left": 61,
    "mouth_right": 291,
    "forehead": 10,
    "chin": 152,
    "cheek_left": 234,
    "cheek_right": 454,
}


def _load_geometric_landmarker():
    """Load and cache the MediaPipe Face Landmarker used for GD."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "face_landmarker.task")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)

    if not hasattr(_load_geometric_landmarker, "_detector"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _load_geometric_landmarker._detector = vision.FaceLandmarker.create_from_options(options)
    return _load_geometric_landmarker._detector


def _extract_landmarks(img_np):
    """Return nine compact MediaPipe landmarks for GD and visualization."""
    detector = _load_geometric_landmarker()
    rgb = img_np[..., ::-1].copy()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:
        return None
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    h, w = img_np.shape[:2]
    return {
        name: (float(lm[idx].x * w), float(lm[idx].y * h))
        for name, idx in GEOMETRIC_LANDMARK_INDICES.items()
    }


def _geometric_descriptors(lm):
    """Calculate six normalized, dimensionless geometric descriptors."""
    p = {k: np.array(v, dtype=np.float64) for k, v in lm.items()}
    eye_center = (p["left_eye"] + p["right_eye"]) / 2.0
    mouth_center = (p["mouth_left"] + p["mouth_right"]) / 2.0

    d_eyes = np.linalg.norm(p["left_eye"] - p["right_eye"])
    d_cheeks = np.linalg.norm(p["cheek_left"] - p["cheek_right"])
    d_face = np.linalg.norm(p["forehead"] - p["chin"])
    d_eye_nose = np.linalg.norm(eye_center - p["nose"])
    d_nose_mouth = np.linalg.norm(p["nose"] - mouth_center)
    d_mouth = np.linalg.norm(p["mouth_left"] - p["mouth_right"])
    d_forehead_eyes = np.linalg.norm(p["forehead"] - eye_center)

    eps = 1e-8
    return {
        "eyes_to_cheeks": d_eyes / max(d_cheeks, eps),
        "mouth_to_cheeks": d_mouth / max(d_cheeks, eps),
        "eyes_nose_to_face": d_eye_nose / max(d_face, eps),
        "nose_mouth_to_face": d_nose_mouth / max(d_face, eps),
        "face_to_cheeks": d_face / max(d_cheeks, eps),
        "forehead_eyes_to_face": d_forehead_eyes / max(d_face, eps),
    }

def geometric_delta(orig_tensor, adv_tensor):
    """
    orig_tensor, adv_tensor: (N, C, H, W) в [0,1].
    Возвращает (N,) numpy массив относительного изменения GD.
    """
    import numpy as np
    N = orig_tensor.size(0)
    deltas = np.full(N, np.nan)

    for i in range(N):
        orig_np = (orig_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        adv_np  = (adv_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        lm_o = _extract_landmarks(orig_np)
        lm_a = _extract_landmarks(adv_np)
        if lm_o is None or lm_a is None:
            continue

        g_o = _geometric_descriptors(lm_o)
        g_a = _geometric_descriptors(lm_a)

        rel = [abs(g_a[k] - g_o[k]) / max(g_o[k], 1e-6) for k in g_o]
        deltas[i] = float(np.mean(rel))

    return deltas