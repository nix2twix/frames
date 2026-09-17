# source/metrics.py
import numpy as np
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
                        threshold: float = 0.5, mode: str = 'impersonation', min_change=0.2):
    """
    Compute attack success per-sample and overall ASR.
    - emb_before, emb_after: shape (N, D)
    - emb_target: shape (N, D) OR (D,) (if one target for all)
    mode:
      - 'impersonation': sim_after >= T
      - 'deauth': sim_before >= T 
    Returns: (success_array_bool(N,), ASR_float)
    """
    if emb_target.ndim == 1:
        emb_target = emb_target.unsqueeze(0).repeat(emb_before.size(0), 1)

    sim_before = cosine_similarity_batch(emb_before, emb_target)
    sim_after  = cosine_similarity_batch(emb_after, emb_target)

    if mode == 'impersonation':
        success = (sim_after >= threshold) & ((sim_after - sim_before) >= min_change)
    elif mode == 'deauth':
        success = (sim_before >= threshold) & ((sim_before - sim_after) >= min_change)
    else:
        raise ValueError("Unknown mode")

    asr = float(success.sum() / success.size)
    return success, asr

def confidence_shift(emb_before: torch.Tensor, emb_after: torch.Tensor, 
                     emb_target: torch.Tensor, threshold: float = 0.5):
    """
    Среднее абсолютное изменение косинусного сходства с целевым эмбеддингом.
    Показывает, насколько атака «сдвинула» эмбеддинг.
    """
    sim_before = cosine_similarity_batch(emb_before, emb_target)
    sim_after = cosine_similarity_batch(emb_after, emb_target)
    shift = np.abs(sim_after - sim_before)
    return float(np.mean(shift))

def accuracy_drop(pred_before: np.ndarray, source_labels: np.ndarray, pred_after: np.ndarray):
    """
    Падение точности: разность между долей правильных решений до и после атаки.
    
    В контексте верификации лиц:
    - «правильное» решение = сходство с эталоном выше порога (для impersonation)
      или ниже порога (для deauth).
    """
    acc_clean = (pred_before == source_labels).mean()
    acc_adv = (pred_after == source_labels).mean()
    AD = acc_clean - acc_adv
    return acc_clean, acc_adv, AD

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