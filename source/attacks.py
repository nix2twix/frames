# source/attacks.py
import torch
import numpy as np
from stqdm import stqdm
from tqdm import tqdm as _tqdm_std
import torch.nn.functional as F

def _get_embeddings(model, imgs):
    """
    Accepts either:
      - a callable model: model(imgs) -> embeddings tensor
      - an object with method get_embeddings(imgs) or embed(imgs)
    Returns embeddings on same device as model.
    """
    if callable(model):
        return model(imgs)
    if hasattr(model, "get_embeddings") and callable(getattr(model, "get_embeddings")):
        return model.get_embeddings(imgs)
    if hasattr(model, "embed") and callable(getattr(model, "embed")):
        return model.embed(imgs)
    return model(imgs)

def cosine_loss(emb1, emb2, mode="impersonation"):
    """Loss to minimize for the two attack scenarios.

    Impersonation: minimize 1 - S_target, i.e. maximize target similarity.
    Deauthentication bypass: minimize S_self, i.e. reduce similarity to the
    source identity.
    """
    cos = F.cosine_similarity(emb1, emb2, dim=1)
    mode_lower = str(mode).lower()
    is_impersonation = mode_lower in ("impersonation", "подмена личности")
    if is_impersonation:
        return (1.0 - cos).mean()
    if mode_lower in ("deauth", "обход аутентификации"):
        return cos.mean()
    raise ValueError(f"Unknown attack mode: {mode}")

def _make_iterator(iters, show_progress, desc):
    """Возвращает итератор с прогрессом или без, в зависимости от флага."""
    if not show_progress:
        return range(iters)
    try:
        # stqdm нужен только в Streamlit-контексте
        return stqdm(range(iters), desc=desc)
    except Exception:
        # fallback — обычный tqdm (например, при запуске из CLI)
        return _tqdm_std(range(iters), desc=desc)

def fgsm_embed(model, img, emb_target, epsilon=0.05, mode="impersonation", show_progress=True):
    try:
        param = next(model.parameters())
        target_device = param.device
    except (StopIteration, AttributeError):
        target_device = img.device

    img = img.clone().detach().to(target_device)
    mode_lower = str(mode).lower()
    is_impersonation = mode_lower in ("impersonation", "подмена личности")

    # For deauthentication, S_self = cos(F(x_adv), F(x)) has zero gradient
    # exactly at x_adv = x because the cosine similarity is locally maximal.
    # A small random start breaks this symmetry; this is the randomized FGSM
    # variant used here for the untargeted/dodging scenario.
    if not is_impersonation:
        # Use half of the budget for the random start and half for the
        # gradient step, so the final perturbation remains within epsilon.
        random_radius = epsilon * 0.5
        noise = torch.empty_like(img).uniform_(-random_radius, random_radius)
        img = torch.clamp(img + noise, 0.0, 1.0)

    img.requires_grad_(True)
    emb = _get_embeddings(model, img)
    loss = cosine_loss(emb, emb_target.to(emb.device), mode=mode)
    if hasattr(model, "zero_grad"):
        model.zero_grad()
    loss.backward()
    # cosine_loss is defined for minimization, therefore use a gradient-descent
    # step. This increases target similarity in impersonation and decreases
    # self-similarity in deauthentication mode.
    step = epsilon if is_impersonation else epsilon * 0.5
    adv = img - step * img.grad.sign()
    return torch.clamp(adv, 0.0, 1.0).detach()


def _attack_success(emb, emb_target, emb_orig, threshold, is_impersonation):
    """Возвращает маску достижения сценарной цели для каждого изображения."""
    sim_target = F.cosine_similarity(emb, emb_target, dim=1)
    sim_self = F.cosine_similarity(emb, emb_orig, dim=1)
    if is_impersonation:
        return sim_target >= threshold
    return sim_self < threshold


def pgd_embed(model, img, emb_target, epsilon=0.05, alpha=0.01, iters=10,
              mode="impersonation", threshold=0.5, show_progress=True):
    device = img.device
    orig = img.clone().detach().to(device)
    emb_orig = _get_embeddings(model, orig).detach()
    mode_lower = str(mode).lower()
    is_impersonation = mode_lower in ("impersonation", "подмена личности")
    if not is_impersonation and mode_lower not in ("deauth", "обход аутентификации"):
        raise ValueError(f"Unknown attack mode: {mode}")

    # Для deauth чистое изображение является точкой нулевого градиента
    # косинусного self-similarity, поэтому используем random start.
    if is_impersonation:
        adv = orig.clone().detach()
    else:
        noise = torch.empty_like(orig).uniform_(-epsilon, epsilon)
        adv = torch.clamp(orig + noise, 0.0, 1.0).detach()

    # Состояние каждого изображения ведётся независимо: успешные образцы
    # больше не оптимизируются, остальные продолжают до достижения цели.
    active = torch.ones(orig.size(0), dtype=torch.bool, device=device)
    best_adv = adv.clone().detach()
    best_l2 = torch.full((orig.size(0),), float("inf"), device=device)

    for _ in _make_iterator(iters, show_progress, "PGD"):
        if not active.any():
            break

        adv.requires_grad_(True)
        emb = _get_embeddings(model, adv)
        loss_per_sample = (
            1.0 - F.cosine_similarity(emb, emb_target.to(emb.device), dim=1)
            if is_impersonation
            else F.cosine_similarity(emb, emb_orig, dim=1)
        )
        loss = loss_per_sample[active].mean()

        if hasattr(model, "zero_grad"):
            model.zero_grad()
        loss.backward()

        with torch.no_grad():
            updated = adv - alpha * adv.grad.sign()
            perturbation = torch.clamp(updated - orig, -epsilon, epsilon)
            updated = torch.clamp(orig + perturbation, 0.0, 1.0)
            # Уже достигшие цели образцы сохраняются неизменными.
            adv = torch.where(active[:, None, None, None], updated, best_adv).detach()

            emb_check = _get_embeddings(model, adv)
            success = _attack_success(
                emb_check, emb_target.to(emb_check.device), emb_orig,
                threshold, is_impersonation
            ) & active

            if success.any():
                l2_now = torch.sum((adv - orig) ** 2, dim=(1, 2, 3))
                better = success & (l2_now < best_l2)
                best_adv[better] = adv[better]
                best_l2[better] = l2_now[better]
                active[success] = False

    # Для неуспешных образцов возвращаем последнее состояние; для успешных —
    # лучший найденный успешный вариант с минимальным L2.
    result = adv.clone().detach()
    found = torch.isfinite(best_l2)
    result[found] = best_adv[found]
    return result


def cw_embed(model, img, emb_target, c=1.0, lr=0.01, iters=500,
             epsilon=0.08, mode="impersonation", threshold=0.5,
             show_progress=True, debug=False):
    """
    Оптимизационная атака на основе подхода Carlini-Wagner.

    Оптимизируется входное изображение с L2-штрафом и явным L_inf-
    ограничением. Успешные образцы фиксируются, а для каждого из них
    сохраняется успешное состояние с минимальным L2-возмущением.
    """
    device = img.device
    orig = img.clone().detach().to(device)
    N = img.size(0)

    with torch.no_grad():
        emb_orig = _get_embeddings(model, orig).detach()

    mode_lower = str(mode).lower()
    is_impersonation = mode_lower in ("impersonation", "подмена личности")
    if not is_impersonation and mode_lower not in ("deauth", "обход аутентификации"):
        raise ValueError(f"Unknown attack mode: {mode}")

    if debug:
        print(f"[CW] START: eps={epsilon}, c={c}, lr={lr}, iters={iters}, mode={mode}, threshold={threshold}")

    if is_impersonation:
        adv = orig.clone().detach()
    else:
        # Self-similarity has zero gradient at the clean image.
        noise = torch.empty_like(orig).uniform_(-epsilon, epsilon)
        adv = torch.clamp(orig + noise, 0.0, 1.0).detach()

    active = torch.ones(N, dtype=torch.bool, device=device)
    best_adv = adv.clone().detach()
    best_l2 = torch.full((N,), float("inf"), device=device)

    adv.requires_grad_(True)
    optimizer = torch.optim.Adam([adv], lr=lr)
    iterator = _make_iterator(iters, show_progress, "CW")

    for i, _ in enumerate(iterator):
        if not active.any():
            break

        optimizer.zero_grad()
        emb = _get_embeddings(model, adv)
        cos_target = F.cosine_similarity(emb, emb_target.to(emb.device), dim=1)
        cos_orig = F.cosine_similarity(emb, emb_orig, dim=1)

        if is_impersonation:
            # Target similarity must reach the decision threshold.
            f_per_sample = torch.clamp(threshold - cos_target, min=0.0)
        else:
            # Self-similarity must fall below the decision threshold.
            f_per_sample = torch.clamp(cos_orig - threshold, min=0.0)

        l2_per_sample = torch.sum((adv - orig) ** 2, dim=(1, 2, 3))
        loss = (l2_per_sample[active] + c * f_per_sample[active]).mean()
        loss.backward()

        optimizer.step()

        with torch.no_grad():
            adv.data = torch.clamp(adv.data, orig - epsilon, orig + epsilon)
            adv.data = torch.clamp(adv.data, 0.0, 1.0)

            # Freeze samples that are no longer active.
            adv.data[~active] = best_adv[~active]

            emb_check = _get_embeddings(model, adv)
            success = _attack_success(
                emb_check, emb_target.to(emb_check.device), emb_orig,
                threshold, is_impersonation
            ) & active

            if success.any():
                l2_now = torch.sum((adv - orig) ** 2, dim=(1, 2, 3))
                better = success & (l2_now < best_l2)
                best_adv[better] = adv[better]
                best_l2[better] = l2_now[better]
                active[success] = False

            if debug and i % 50 == 0:
                diff = adv - orig
                print(f"[CW] iter {i:4d} | loss={loss.item():.4f} | "
                      f"cos_t={cos_target.mean().item():.4f} | "
                      f"cos_o={cos_orig.mean().item():.4f} | "
                      f"Linf={diff.abs().max().item():.4f} | "
                      f"success={int((~active).sum().item())}/{N}")

    with torch.no_grad():
        result = adv.detach().clone()
        found = torch.isfinite(best_l2)
        result[found] = best_adv[found]
        result = torch.clamp(result, 0.0, 1.0)

    return result


def _load_face_landmarker():
    """Load MediaPipe Face Landmarker lazily and cache the detector."""
    import os
    import urllib.request
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "face_landmarker.task")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)

    if not hasattr(_load_face_landmarker, "_detector"):
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
        _load_face_landmarker._detector = vision.FaceLandmarker.create_from_options(options)
    return _load_face_landmarker._detector


def _extract_dense_face_landmarks(img_u8):
    """Return a stable subset of MediaPipe Face Mesh landmarks as (x, y)."""
    import mediapipe as mp

    detector = _load_face_landmarker()
    rgb = img_u8[..., ::-1].copy()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    h, w = img_u8.shape[:2]

    # 68 semantically distributed control points: face contour, brows,
    # eyes, nose and lips. The same indices are used for source and target,
    # so the correspondence is explicit and does not depend on triangulation.
    indices = [
        # Face contour
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        # Eyebrows
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        # Eyes
        33, 133, 160, 159, 158, 157, 173, 362, 263, 387, 386, 385,
        384, 398,
        # Nose
        1, 2, 98, 327, 168, 197,
        # Lips
        61, 291, 0, 17, 13, 14, 78, 308,
    ]
    return np.array([(float(lm[i].x * w), float(lm[i].y * h)) for i in indices], dtype=np.float32)


def _piecewise_affine_warp(orig_u8, src_pts, dst_pts):
    """Warp image using a fixed source Delaunay triangulation."""
    import cv2

    H, W = orig_u8.shape[:2]
    # Add image corners as fixed control points so the background is preserved.
    corners = np.array([
        [0.0, 0.0], [W - 1.0, 0.0],
        [W - 1.0, H - 1.0], [0.0, H - 1.0],
    ], dtype=np.float32)
    src_all = np.vstack([src_pts, corners])
    dst_all = np.vstack([dst_pts, corners])

    # Subdiv2D triangulates the source geometry directly. Each returned
    # triangle is mapped back to the exact control-point indices.
    rect = (0, 0, W, H)
    subdiv = cv2.Subdiv2D(rect)
    for p in src_all:
        x = float(np.clip(p[0], 0, W - 1))
        y = float(np.clip(p[1], 0, H - 1))
        try:
            subdiv.insert((x, y))
        except cv2.error:
            pass

    triangles = subdiv.getTriangleList()
    warped = np.zeros_like(orig_u8)

    for t in triangles:
        tri = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.float32)
        idx = []
        valid = True
        for p in tri:
            d = np.linalg.norm(src_all - p, axis=1)
            j = int(np.argmin(d))
            if d[j] > 1.5:
                valid = False
                break
            idx.append(j)
        if not valid or len(set(idx)) != 3:
            continue

        tri_src = src_all[idx]
        tri_dst = dst_all[idx]
        src_det = (tri_src[1, 0] - tri_src[0, 0]) * (tri_src[2, 1] - tri_src[0, 1]) - \
                  (tri_src[1, 1] - tri_src[0, 1]) * (tri_src[2, 0] - tri_src[0, 0])
        dst_det = (tri_dst[1, 0] - tri_dst[0, 0]) * (tri_dst[2, 1] - tri_dst[0, 1]) - \
                  (tri_dst[1, 1] - tri_dst[0, 1]) * (tri_dst[2, 0] - tri_dst[0, 0])
        if abs(src_det) < 1e-4 or abs(dst_det) < 1e-4:
            continue

        M = cv2.getAffineTransform(tri_src, tri_dst)
        warped_tri = cv2.warpAffine(
            orig_u8, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(tri_dst).astype(np.int32), 255)
        warped[mask > 0] = warped_tri[mask > 0]

    # Triangles can leave tiny holes due to numerical rounding. Preserve the
    # original image there instead of introducing black pixels.
    empty = np.all(warped == 0, axis=2)
    warped[empty] = orig_u8[empty]
    return warped


GEOMETRIC_LANDMARK_NAMES = (
    "left_eye", "right_eye", "nose", "mouth_left", "mouth_right",
    "forehead", "chin", "cheek_left", "cheek_right",
)
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


def _extract_geometric_landmarks(img_u8):
    """Return the nine compact MediaPipe landmarks used by the geometric attack."""
    detector = _load_face_landmarker()
    import mediapipe as mp

    rgb = img_u8[..., ::-1].copy()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:
        return None
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    h, w = img_u8.shape[:2]
    return np.array(
        [[float(lm[GEOMETRIC_LANDMARK_INDICES[name]].x * w),
          float(lm[GEOMETRIC_LANDMARK_INDICES[name]].y * h)]
         for name in GEOMETRIC_LANDMARK_NAMES],
        dtype=np.float64,
    )


def _geometric_features(points):
    """Six dimensionless geometric descriptors for the nine compact points."""
    p = np.asarray(points, dtype=np.float64)
    left_eye, right_eye, nose, mouth_left, mouth_right, forehead, chin, cheek_left, cheek_right = p
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (mouth_left + mouth_right) / 2.0

    d_eyes = np.linalg.norm(left_eye - right_eye)
    d_cheeks = np.linalg.norm(cheek_left - cheek_right)
    d_face = np.linalg.norm(forehead - chin)
    d_eye_nose = np.linalg.norm(eye_center - nose)
    d_nose_mouth = np.linalg.norm(nose - mouth_center)
    d_mouth = np.linalg.norm(mouth_left - mouth_right)
    d_forehead_eyes = np.linalg.norm(forehead - eye_center)

    eps = 1e-8
    return np.array([
        d_eyes / max(d_cheeks, eps),
        d_mouth / max(d_cheeks, eps),
        d_eye_nose / max(d_face, eps),
        d_nose_mouth / max(d_face, eps),
        d_face / max(d_cheeks, eps),
        d_forehead_eyes / max(d_face, eps),
    ], dtype=np.float64)


def _solve_geometric_landmarks(src_points, target_features, max_iter=200):
    """Find the minimum-displacement 9-point configuration with target geometry."""
    from scipy.optimize import minimize

    src = np.asarray(src_points, dtype=np.float64)
    scale = max(np.linalg.norm(src[7] - src[8]), 1.0)
    x0 = src.reshape(-1).copy()

    def objective(x):
        d = x.reshape(-1, 2) - src
        return float(np.sum(d * d) / (scale * scale))

    def equality(x):
        return _geometric_features(x.reshape(-1, 2)) - target_features

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints={"type": "eq", "fun": equality},
        options={"maxiter": int(max_iter), "ftol": 1e-10, "disp": False},
    )

    if not result.success:
        # SLSQP can stop early on nearly degenerate faces. The returned point
        # is still useful only if the geometric residual is demonstrably small.
        candidate = result.x.reshape(-1, 2)
    else:
        candidate = result.x.reshape(-1, 2)

    residual = np.linalg.norm(_geometric_features(candidate) - target_features)
    if not np.isfinite(residual) or residual > 1e-5:
        return None
    return candidate.astype(np.float64)


def _interpolate_geometric_displacement(src_dense, anchor_src, anchor_dst):
    """Spread nine anchor displacements to dense controls with thin-plate splines."""
    from scipy.interpolate import RBFInterpolator

    displacement = anchor_dst - anchor_src
    # Thin-plate spline interpolation is the standard minimum-bending smooth
    # extension of sparse landmark displacements; no region-specific weights
    # are introduced here.
    rbf = RBFInterpolator(
        anchor_src,
        displacement,
        kernel="thin_plate_spline",
        degree=1,
        smoothing=0.0,
    )
    return np.asarray(src_dense, dtype=np.float64) + rbf(src_dense)



def _solve_deauth_landmarks(src_points, strength, max_iter=250):
    """Maximize geometric deviation under a conservative displacement budget."""
    from scipy.optimize import minimize

    src = np.asarray(src_points, dtype=np.float64)
    scale = max(np.linalg.norm(src[7] - src[8]), 1.0)

    # Ограничиваем суммарное перемещение ориентиров.
    # strength=0.5 соответствует бюджету 7.5% ширины лица.
    budget = 0.15 * float(np.clip(strength, 0.0, 1.0)) * scale

    if budget <= 0:
        return src.copy()

    x0 = src.reshape(-1).copy()

    rng = np.random.default_rng(42)
    displacement = rng.normal(size=src.shape)
    displacement_norm = np.linalg.norm(displacement)

    if displacement_norm > 0:
        displacement *= 0.5 * budget / displacement_norm
        x0 = (src + displacement).reshape(-1)

    def objective(x):
        g = _geometric_features(x.reshape(-1, 2))
        g0 = _geometric_features(src)
        return -float(np.sum((g - g0) ** 2))

    def budget_constraint(x):
        d = x.reshape(-1, 2) - src
        return budget * budget - float(np.sum(d * d))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints={"type": "ineq", "fun": budget_constraint},
        options={"maxiter": int(max_iter), "ftol": 1e-10, "disp": False},
    )

    candidate = result.x.reshape(-1, 2)

    if not np.isfinite(candidate).all():
        return None

    if np.linalg.norm(candidate - src) > budget + 1e-5:
        return None

    # Не принимаем решение, если оптимизатор фактически остался в исходной точке.
    if np.linalg.norm(candidate - src) < 1e-4:
        return None

    return candidate



def geometric_attack_embed(model, img, target_img, strength=1.0,
                           mode="impersonation", show_progress=True,
                           return_valid_mask=False):
    """
    Model-independent geometric attack.

    For impersonation, the target image supplies six normalized geometric
    descriptors. Nine compact landmarks are moved by solving a constrained
    minimum-displacement inverse problem. The resulting nine displacements
    are extended to the dense 74-point control mesh with a thin-plate spline,
    after which the existing piecewise-affine image warp is applied.

    For deauthentication, the same model-independent geometric mechanism is
    used without a target identity: the normalized geometric profile is
    changed as much as possible under the relative landmark-displacement
    budget given by strength. No recognition model is used.
    """
    from source.visualisation import tensor_to_np01

    device = img.device
    N = img.size(0)
    mode_lower = str(mode).lower()
    if mode_lower in ("impersonation", "подмена личности"):
        is_imp = True
    elif mode_lower in ("deauth", "обход аутентификации"):
        is_imp = False
    else:
        raise ValueError(f"Unknown attack mode: {mode}")

    strength = float(np.clip(strength, 0.0, 1.0))
    adv_list = []
    valid_mask = []

    for i in range(N):
        orig_np = tensor_to_np01(img[i])
        orig_u8 = (np.clip(orig_np, 0, 1) * 255).astype(np.uint8)
        src_dense = _extract_dense_face_landmarks(orig_u8)
        src_anchor = _extract_geometric_landmarks(orig_u8)

        if src_dense is None or src_anchor is None:
            adv_list.append(img[i].clone())
            valid_mask.append(False)
            continue

        src_features = _geometric_features(src_anchor)

        if is_imp:
            target_np = tensor_to_np01(target_img[i])
            target_u8 = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)
            target_anchor = _extract_geometric_landmarks(target_u8)
            if target_anchor is None:
                adv_list.append(img[i].clone())
                valid_mask.append(False)
                continue
            target_features = _geometric_features(target_anchor)
        else:
            # No target identity is required for deauthentication. Instead,
            # maximize the change of the normalized geometric profile under a
            # dimensionless landmark-displacement budget.
            dst_anchor = _solve_deauth_landmarks(src_anchor, strength)

        if is_imp:
            goal_features = src_features + strength * (target_features - src_features)
            dst_anchor = _solve_geometric_landmarks(src_anchor, goal_features)

        if dst_anchor is None:
            adv_list.append(img[i].clone())
            valid_mask.append(False)
            continue
        dst_dense = _interpolate_geometric_displacement(
            src_dense, src_anchor, dst_anchor
        )

        # Ограничиваем максимальное смещение любой точки плотной сетки.
        # TPS может усиливать смещения за пределами опорных точек.
        dense_displacement = dst_dense - src_dense

        scale = max(
            np.linalg.norm(src_anchor[7] - src_anchor[8]),
            1.0,
        )
        max_shift = 0.15 * strength * scale

        displacement_norms = np.linalg.norm(
            dense_displacement, axis=1
        )
        max_observed_shift = displacement_norms.max(initial=0.0)

        if max_observed_shift > max_shift and max_observed_shift > 0:
            dense_displacement *= max_shift / max_observed_shift
            dst_dense = src_dense + dense_displacement


        # Keep the four image corners fixed through _piecewise_affine_warp.
        warped = _piecewise_affine_warp(
            orig_u8,
            src_dense.astype(np.float32),
            dst_dense.astype(np.float32),
        )
    
        warped_landmarks = _extract_geometric_landmarks(warped)
        if warped_landmarks is None:
            adv_list.append(img[i].clone())
            valid_mask.append(False)
            continue

        warped_features = _geometric_features(warped_landmarks)

        if is_imp:
            # Подмена личности: сравниваем с целевой геометрией.
            d_warp = np.linalg.norm(warped_features - goal_features)
            d_target = np.linalg.norm(warped_features - target_features)
        else:
            # Обход аутентификации: оцениваем изменение
            # геометрии относительно исходного изображения.
            d_warp = np.linalg.norm(warped_features - src_features)

        adv_np = warped.astype(np.float32) / 255.0
        adv_list.append(torch.from_numpy(adv_np).permute(2, 0, 1).to(device))
        valid_mask.append(True)

    adv = torch.stack(adv_list, dim=0)
    valid = torch.tensor(valid_mask, dtype=torch.bool, device=device)
    if return_valid_mask:
        return adv, valid
    return adv
