# source/attacks.py
import torch
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
    cos = F.cosine_similarity(emb1, emb2, dim=1)
    if mode == "impersonation":
        return (1.0 - cos).mean()
    else:
        return cos.mean()

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
    img.requires_grad_(True)
    emb = _get_embeddings(model, img)
    loss = cosine_loss(emb, emb_target.to(emb.device), mode=mode)
    if hasattr(model, "zero_grad"):
        model.zero_grad()
    loss.backward()
    adv = img + epsilon * img.grad.sign()
    return torch.clamp(adv, 0.0, 1.0).detach()


def pgd_embed(model, img, emb_target, epsilon=0.05, alpha=0.01, iters=10,
                mode="impersonation", show_progress=True):
    device = img.device
    orig = img.clone().detach().to(device)
    adv = orig.clone().detach()
    for _ in _make_iterator(iters, show_progress, "I-FGSM"):
        adv.requires_grad_(True)
        emb = _get_embeddings(model, adv)
        loss = cosine_loss(emb, emb_target.to(emb.device), mode=mode)
        if hasattr(model, "zero_grad"):
            model.zero_grad()
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        perturbation = torch.clamp(adv - orig, -epsilon, epsilon)
        adv = torch.clamp(orig + perturbation, 0.0, 1.0).detach()
    return adv


def cw_embed(model, img, emb_target, c=1.0, lr=0.01, iters=500,
             epsilon=0.08, mode="impersonation", show_progress=True):
    """
    CW-подобная атака с прямой оптимизацией delta.
    L∞-ограничение через clamp, CW-лосс f = max(0, cos_orig - cos_target).
    """
    device = img.device
    orig = img.clone().detach().to(device)
    N = img.size(0)

    # Эмбеддинг оригинала для CW-лосса
    with torch.no_grad():
        emb_orig = _get_embeddings(model, orig)

    # Прямая оптимизация возмущения
    delta = torch.zeros_like(orig, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    iterator = _make_iterator(iters, show_progress, "CW")

    for _ in iterator:
        optimizer.zero_grad()

        # L∞-ограничение
        delta_clamped = torch.clamp(delta, -epsilon, epsilon)
        adv = torch.clamp(orig + delta_clamped, 0.0, 1.0)

        emb = _get_embeddings(model, adv)

        cos_target = F.cosine_similarity(emb, emb_target.to(emb.device), dim=1)
        cos_orig = F.cosine_similarity(emb, emb_orig, dim=1)

        if mode == "impersonation":
            # Хотим cos_target > cos_orig
            f = torch.clamp(cos_orig - cos_target, min=0.0)
        else:
            # Хотим cos_orig < cos_target
            f = torch.clamp(cos_target - cos_orig, min=0.0)

        # L2-штраф (как в оригинальном CW)
        l2 = torch.sum((adv - orig) ** 2) / N

        # Полный лосс: L2 + c * f
        loss = l2 + c * f.mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        delta_clamped = torch.clamp(delta, -epsilon, epsilon)
        adv = torch.clamp(orig + delta_clamped, 0.0, 1.0)
    return adv

def patch_attack_embed(
    model, img, emb_target,
    patch_size=64,
    epsilon=0.3,
    iters=300,
    lr=0.005,
    mode="impersonation",
    show_progress=True,
):
    """
    PatchAttack: обучаемый патч в центральной области лица.
    
    Особенности:
      - патч инициализируется фрагментом кожи (не случайным шумом);
      - маска эллиптическая — патч плавно смешивается с лицом;
      - позиция случайна в центральной области лица;
      - каждые 10 итераций патч сдвигается на ±2 пикселя 
        для повышения робастности;
      - регуляризация L2 не даёт патчу «улететь» в белый/чёрный.
    """
    device = img.device
    N, C, H, W = img.shape
    ph = pw = min(patch_size, int(H * 0.7), int(W * 0.7))

    # --- 1. Инициализация: фрагмент кожи со щеки (не из центра) ---
    with torch.no_grad():
        ref_y = int(H * 0.55)
        ref_x = int(W * 0.30)
        ref_y = max(0, min(H - ph, ref_y))
        ref_x = max(0, min(W - pw, ref_x))
        patch_init = img[:, :, ref_y:ref_y + ph, ref_x:ref_x + pw].clone()

    patch = torch.nn.Parameter(patch_init.clone())
    patch_init_detached = patch_init.detach().clone()

    # --- 2. Эллиптическая маска для плавного смешивания ---
    yy, xx = torch.meshgrid(
        torch.arange(ph, device=device),
        torch.arange(pw, device=device),
        indexing="ij",
    )
    mask = (
        ((yy - ph / 2) / (ph / 2)) ** 2
        + ((xx - pw / 2) / (pw / 2)) ** 2
    ) <= 1.0
    mask = mask.float().view(1, 1, ph, pw)          # (1, 1, ph, pw)
    mask = mask.expand(1, C, ph, pw)                 # (1, C, ph, pw)

    # --- 3. Позиция патча: центральная область лица со сдвигом ---
    tops, lefts = [], []
    for _ in range(N):
        t = int(H * 0.35) + torch.randint(-H // 15, H // 15, (1,)).item()
        l = int(W * 0.35) + torch.randint(-W // 15, W // 15, (1,)).item()
        t = max(0, min(H - ph, t))
        l = max(0, min(W - pw, l))
        tops.append(t)
        lefts.append(l)

    # --- 4. Оптимизатор ---
    optimizer = torch.optim.Adam([patch], lr=lr)

    # --- 5. Основной цикл ---
    iterator = _make_iterator(iters, show_progress, "PatchAttack")

    for it_idx in iterator:
        optimizer.zero_grad()

        # Каждые 10 итераций — небольшой сдвиг патча (робастность)
        if it_idx > 0 and it_idx % 10 == 0:
            for i in range(N):
                t = max(0, min(H - ph, tops[i] + torch.randint(-2, 3, (1,)).item()))
                l = max(0, min(W - pw, lefts[i] + torch.randint(-2, 3, (1,)).item()))
                tops[i] = t
                lefts[i] = l

        # --- Наложение патча через маску ---
        adv = img.clone()
        for i in range(N):
            t = tops[i]
            l = lefts[i]
            orig_patch = adv[i, :, t:t + ph, l:l + pw].unsqueeze(0)  # (1, C, ph, pw)
            blended = mask * patch[i:i + 1] + (1 - mask) * orig_patch
            adv[i, :, t:t + ph, l:l + pw] = blended.squeeze(0)

        # --- Forward ---
        emb = _get_embeddings(model, adv)
        loss_sim = cosine_loss(emb, emb_target.to(emb.device), mode=mode)

        # --- Регуляризация L2: патч не должен сильно отличаться от кожи ---
        loss_reg = torch.mean((patch - patch_init_detached) ** 2) * 0.5

        loss = loss_sim + loss_reg
        loss.backward()
        optimizer.step()

        # --- Ограничение диапазона ---
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

    # --- 6. Финальное наложение ---
    with torch.no_grad():
        adv = img.clone()
        for i in range(N):
            t = tops[i]
            l = lefts[i]
            orig_patch = adv[i, :, t:t + ph, l:l + pw].unsqueeze(0)
            blended = mask * patch[i:i + 1] + (1 - mask) * orig_patch
            adv[i, :, t:t + ph, l:l + pw] = blended.squeeze(0)

    return torch.clamp(adv, 0.0, 1.0).detach()