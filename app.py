# --- ENVIRONMENTS
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
device = "cpu"

# --- GENERAL LIBS
import time
import torch
import logging
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime
from pathlib import Path

# --- CUSTOM LIBS
from source.models import get_embedder, CustomEmbedder
from source.attacks import fgsm_embed, pgd_embed, cw_embed, patch_attack_embed
from source.dataset import load_dataset_from_zip, validate_data_root, get_dataloader
from source.visualisation import make_diff_heatmap, tensor_to_np01
from source.metrics import (
    l0_count_per_image, l2_per_image, linf_per_image,
    cosine_similarity_batch, attack_success_rate,
    confidence_shift, accuracy_drop, predict_identity
)
from source.utils import (
    export_metrics_to_csv, export_heatmaps_to_zip, export_adversarial_to_zip,
    save_uploaded_file, export_adversarial_dataset, export_metrics_to_pdf,
    run_attack_series, export_metrics_to_html, init_session
)
from styles.style import load_styles, load_header

# --- APP CONFIGURATION
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / f"frames_{datetime.now():%Y%m%d}.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)

# --- CONFIGURATION
load_styles()
load_header()
init_session()

col_settings, col_general = st.columns([1, 3])
with col_settings:
    st.markdown("#### ⚙️ Общие настройки")
    model_name = st.radio(
        "Модель:",
        ["FaceNet", "ArcFace", "MobileNet", "Custom"],
        key="model_select",
    )
    use_default = st.toggle("Встроенный датасет", value=True, key="use_default_ds")
    data_root_valid = False
    aligned_data = True
    data_root = None

    if use_default:
        default_path = Path.cwd() / "data" / "celeba_subset"
        val = validate_data_root(default_path)
        if val["ok"]:
            print(f"Найдено {len(val['classes'])} классов")
            st.session_state["dataset_root"] = default_path
            data_root = default_path
            data_root_valid = True
            aligned_data = True
        else:
            st.error("Встроенный датасет повреждён.")
    else:
        uploaded_zip = st.file_uploader(
            "Архив с изображениями (.zip)", type=["zip"], key="up_zip"
        )
        if uploaded_zip is not None:
            data_root = load_dataset_from_zip(uploaded_zip)
            validation = validate_data_root(data_root)
            if validation["ok"]:
                print(f"Найдено {len(validation['classes'])} классов")
                st.session_state["dataset_root"] = data_root
                data_root_valid = True
                aligned_data = False
            else:
                st.error(validation["summary"])
    batch_size = st.slider("Размер батча", 2, 30, 6, key="batch_size_slider")

    if model_name == "Custom":
        uploaded_weights = st.file_uploader(
            "Веса модели (.pt, .pth)", type=["pt", "pth"], key="up_weights"
        )
        uploaded_arch = st.file_uploader(
            "Файл архитектуры (.py)", type=["py"], key="up_arch"
        )
        class_name = st.text_input(
            "Имя класса модели", value="Embedder", key="class_name"
        )

        if uploaded_weights is not None and uploaded_arch is not None:
            if st.button("Загрузить модель", key="load_custom_btn"):
                weights_path = save_uploaded_file(uploaded_weights, suffix=".pt")
                arch_path = save_uploaded_file(uploaded_arch, suffix=".py")
                try:
                    model = CustomEmbedder(
                        model_path=weights_path,
                        framework="pytorch",
                        arch_path=arch_path,
                        class_name=class_name,
                        device=device,
                    )
                    st.session_state["custom_model"] = model
                    st.session_state["_cached_model_name"] = model_name
                    st.session_state["_batch_signature"] = None
                    st.success("Модель загружена.")
                except Exception as e:
                    st.error(f"Ошибка загрузки: {e}")
    else:
        if st.session_state.get("_cached_model_name") != model_name:
            with st.spinner(f"Загрузка модели {model_name}..."):
                st.session_state["custom_model"] = get_embedder(model_name, device=device)
                st.session_state["_cached_model_name"] = model_name
                st.session_state["_batch_signature"] = None

if data_root_valid and st.session_state.get("custom_model") is not None:
    current_signature = (
        st.session_state.get("_cached_model_name"),
        str(st.session_state.get("dataset_root")),
        batch_size,
        aligned_data,
    )
    if st.session_state.get("_batch_signature") != current_signature:
        with st.spinner("Подготовка батча данных..."):
            model = st.session_state["custom_model"]
            loader = get_dataloader(
                root=st.session_state["dataset_root"],
                batch_size=batch_size,
                shuffle=True,
                aligned=aligned_data,
            )
            imgs, labels = next(iter(loader))
            imgs = imgs.to(device)

            st.session_state["imgs"] = imgs
            st.session_state["labels"] = labels
            st.session_state["emb_before"] = model.get_embeddings(imgs)
            st.session_state["_batch_signature"] = current_signature
            st.session_state["batch_ready"] = True

with col_general:
    tab_general, tab_series = st.tabs(["🔬 Одиночный эксперимент", "📈 Серия экспериментов"])
    with tab_general:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("#### Параметры атаки")
            attack_type = st.selectbox(
                "Тип атаки:",
                ["FGSM", "PGD", "Carlini-Wagner", "PatchAttack"],
                key="attack_type_general",
            )

            with st.expander("Настройки атаки", expanded=True):
                epsilon = st.slider("Макс. возмущение ε", 0.0, 0.3, 0.08, 0.01, key="eps_general")
                alpha = 0.01
                iters = 15
                c = 1.0
                lr = 0.01
                patch_size = 32

                if attack_type == "PGD":
                    alpha = st.slider("Шаг α", 0.001, 0.05, 0.01, key="alpha_general")
                    iters = st.slider("Итераций", 1, 100, 15, key="iters_general")

                elif attack_type == "Carlini-Wagner":
                    c = st.number_input(
                        "C (регуляризация)",
                        value=1.0, min_value=0.01, max_value=100.0, step=0.1,
                        format="%.2f", key="c_general",
                        help="Вес CW-лосса. 1.0 — баланс, 10.0 — сильнее атака."
                    )
                    lr = st.number_input("LR", value=0.01, format="%.3f", key="lr_general")
                    iters = st.slider("Итераций", 100, 1000, 500, 50, key="iters_cw_general")
                    
                elif attack_type == "PatchAttack":
                    patch_size = st.slider("Размер патча", 8, 128, 64, 8, key="ps_general")
                    lr = st.number_input("LR", value=0.005, format="%.3f", key="lr_pa_general")
                    iters = st.slider("Итераций", 10, 500, 300, 50, key="iters_pa_general")

                mode = st.selectbox(
                    "Режим атаки", ["Подмена личности", "Обход аутентификации"],
                    key="mode_general",
                )
                thr = st.slider("Порог сходства", 0.0, 1.0, 0.5, 0.01, key="thr_general")

            run = st.button(
                "🚀 Запустить атаку",
                width="stretch",
                disabled=not st.session_state.get("batch_ready", False),
                key="run_attack_btn",
            )

        with col2:
            st.markdown("#### 📊 Результаты")
            if run:
                model = st.session_state["custom_model"]
                imgs = st.session_state["imgs"]
                emb_before = st.session_state["emb_before"]
                labels = st.session_state["labels"]

                emb_gallery = emb_before
                labels_gallery = labels

                targets = torch.roll(imgs, shifts=1, dims=0) if mode == "Подмена личности" else imgs.clone()
                emb_targets = model.get_embeddings(targets)
                st.session_state["targets"] = targets
                st.session_state["emb_targets"] = emb_targets

                t0 = time.time()
                if attack_type == "FGSM":
                    adv = fgsm_embed(model, imgs, emb_targets, epsilon, mode.lower())

                elif attack_type == "PGD":
                    adv = pgd_embed(model, imgs, emb_targets, epsilon, alpha, iters, mode.lower())
                    
                elif attack_type == "Carlini-Wagner":
                    adv = cw_embed(
                        model, imgs, emb_targets,
                        c=c,
                        lr=lr,
                        iters=iters,
                        epsilon=epsilon,
                        mode=mode.lower(),
                    )
                elif attack_type == "PatchAttack":
                    adv = patch_attack_embed(
                        model, imgs, emb_targets,
                        patch_size=patch_size,
                        lr=lr,
                        iters=iters,
                        mode=mode.lower(),
                    )
                elapsed = time.time() - t0

                emb_after = model.get_embeddings(adv)

                pred_after, best_sim_after = predict_identity(emb_after, emb_gallery, labels_gallery)
                pred_before, best_sim_before = predict_identity(emb_before, emb_gallery, labels_gallery)

                N_batch = imgs.size(0)
                labels_np = labels.detach().cpu().numpy()
                if mode == "Подмена личности":
                    target_labels = labels_np[(np.arange(N_batch) + 1) % N_batch]
                else:
                    target_labels = labels_np

                sim_before = cosine_similarity_batch(emb_before, emb_targets)
                sim_after = cosine_similarity_batch(emb_after, emb_targets)
                cs_value = confidence_shift(emb_before, emb_after, emb_targets, threshold=thr)
                acc_clean, acc_adv, ad_value = accuracy_drop(pred_before, labels_np, pred_after)
                cos_vals = cosine_similarity_batch(emb_before, emb_after)
                l0 = l0_count_per_image(imgs, adv)
                l2 = l2_per_image(imgs, adv)
                linf = linf_per_image(imgs, adv)
                success_arr, asr = attack_success_rate(
                    emb_before, emb_after, emb_targets,
                    threshold=thr,
                    mode="impersonation" if mode == "Подмена личности" else "deauth",
                )

                N = imgs.size(0)
                df = pd.DataFrame({
                    "L₀": l0, "L₂": l2, "L∞": linf, "Scos": cos_vals,
                    "Sim_before": sim_before, "Sim_after": sim_after,
                    "Source": labels_np,
                    "Target": target_labels,
                    "Predicted": pred_after,
                    "Deauth": success_arr,
                    "Match": pred_after == target_labels,
                })

                st.session_state.update({
                    "adv": adv, "emb_after": emb_after, "df": df,
                    "asr": float(asr), "elapsed": elapsed,
                    "accuracy_drop": float(ad_value),
                    "confidence_shift": float(cs_value),
                    "acc_clean": float(acc_clean), "acc_adv": float(acc_adv),
                    "metrics": {
                        "L0": l0, "L2": l2, "L∞": linf, "Scos": cos_vals,
                        "Cos_target_before": sim_before, "Cos_target_after": sim_after,
                        "Success": success_arr,
                        "SourceLabel": labels_np,
                        "TargetLabel": target_labels,
                        "PredictedBefore": pred_before,
                        "PredictedAfter": pred_after,
                        "MatchTarget": pred_after == target_labels,
                    },
                })
                logging.info(
                    f"Attack: model={model_name}, attack={attack_type}, mode={mode}, "
                    f"eps={epsilon}, N={N}, ASR={asr:.4f}, AD={ad_value:.4f}, "
                    f"Cs={cs_value:.4f}, time={elapsed:.2f}s"
                )

            if st.session_state.get("adv") is not None:
                colM1, colM2, colM3 = st.columns([2, 1, 1], vertical_alignment="top")
                with colM1:
                    st.markdown(f"Успешность атаки")
                    asr_val = float(st.session_state["asr"])
                    asr_pct = max(0.0, min(100.0, asr_val * 100.0))
                    r, g, b = int(255 * asr_val), int(255 * (1 - asr_val)), 0
                    fill_color = f"rgb({r},{g},{b})"
                    bg_color = "#444"
                    bar_h = 30
                    bar_html = f"""
                    <div style="width:100%; max-width:100%; box-sizing:border-box;">
                    <div style="width:100%; height:{bar_h}px; background:{bg_color}; border-radius:{bar_h/2}px; overflow:hidden; position:relative;">
                    <div style="height:100%; width:{asr_pct}%; background:{fill_color}; border-radius:{bar_h/2}px;"></div>
                    <div style="position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;
                                font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.6);pointer-events:none;">{asr_pct:.1f}%</div>
                    </div></div>"""
                    st.markdown(bar_html, unsafe_allow_html=True)
                    st.caption("Attack Success Rate")
                with colM2:
                    st.metric("Сдвиг уверенности", f"{float(st.session_state['confidence_shift']):.3f}")
                with colM3:
                    st.metric("Время атаки", f"{st.session_state['elapsed']:.2f} сек")

                st.dataframe(st.session_state["df"], width="stretch")

                _container = st.container(horizontal=True, vertical_alignment="bottom")
                with _container:
                    export_choice = st.selectbox(
                        "Формат экспорта",
                        [
                            "Таблица метрик (.csv)",
                            "Отчёт (.html)",
                            "Отчёт (.pdf)",
                            "Тепловые карты (.zip)",
                            "Искажённые изображения (.zip)",
                            "Данные для дообучения (.zip)",
                            "Логи (.log)"
                        ],
                        key="export_choice_general",
                    )

                    choice_bytes = None
                    filename = "export.bin"

                    if export_choice == "Таблица метрик (.csv)" and st.session_state.get("metrics") is not None:
                        choice_bytes = export_metrics_to_csv(st.session_state["metrics"])
                        filename = "metrics.csv"

                    elif export_choice == "Тепловые карты (.zip)" and st.session_state.get("heatmaps") is not None:
                        choice_bytes = export_heatmaps_to_zip(
                            st.session_state["heatmaps"], st.session_state["labels"]
                        )
                        filename = "heatmaps.zip"

                    elif export_choice == "Искажённые изображения (.zip)" and st.session_state.get("adv") is not None:
                        choice_bytes = export_adversarial_to_zip(
                            st.session_state["adv"],
                            st.session_state["labels"],
                            tensor_to_np01,
                        )
                        filename = "adversarial_examples.zip"

                    elif export_choice == "Данные для дообучения (.zip)" and st.session_state.get("adv") is not None:
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            choice_bytes = export_adversarial_dataset(
                                st.session_state["adv"],
                                st.session_state["imgs"],
                                st.session_state["labels"],
                                attack_type=attack_type,
                                epsilon=epsilon,
                                mode=mode,
                                sim_before=st.session_state["metrics"]["Cos_target_before"],
                                sim_after=st.session_state["metrics"]["Cos_target_after"],
                                success_arr=st.session_state["metrics"]["Success"],
                                output_dir=tmpdir,
                            )
                        filename = "adversarial_dataset.zip"

                    elif export_choice == "Отчёт (.html)":
                        params = {
                            "attack_type": attack_type,
                            "mode": mode,
                            "epsilon": epsilon,
                            "alpha": alpha if attack_type == "PGD" else None,
                            "iters": iters,
                            "c": c if attack_type == "Carlini-Wagner" else None,
                            "lr": lr if attack_type in ("Carlini-Wagner", "PatchAttack") else None,
                            "patch_size": patch_size if attack_type == "PatchAttack" else None,
                            "thr": thr,
                        }
                        params = {k: v for k, v in params.items() if v is not None}

                        choice_bytes = export_metrics_to_html(
                            st.session_state["metrics"], model_name, attack_type,
                            float(st.session_state["asr"]),
                            float(st.session_state["accuracy_drop"]),
                            float(st.session_state["confidence_shift"]),
                            float(st.session_state["elapsed"]),
                            params=params,
                        )
                        filename = "report.html"

                    elif export_choice == "Отчёт (.pdf)":
                        params = {
                            "attack_type": attack_type,
                            "mode": mode,
                            "epsilon": epsilon,
                            "alpha": alpha if attack_type == "PGD" else None,
                            "iters": iters,
                            "c": c if attack_type == "Carlini-Wagner" else None,
                            "lr": lr if attack_type in ("Carlini-Wagner", "PatchAttack") else None,
                            "patch_size": patch_size if attack_type == "PatchAttack" else None,
                            "thr": thr,
                        }
                        params = {k: v for k, v in params.items() if v is not None}

                        choice_bytes = export_metrics_to_pdf(
                            st.session_state["metrics"], model_name, attack_type,
                            float(st.session_state["asr"]),
                            float(st.session_state["accuracy_drop"]),
                            float(st.session_state["confidence_shift"]),
                            float(st.session_state["elapsed"]),
                            params=params,
                        )
                        filename = "report.pdf"

                    elif export_choice == "Логи (.log)":
                        log_files = sorted(LOG_DIR.glob("frames_*.log"))
                        if log_files:
                            choice_bytes = log_files[-1].read_bytes()
                            filename = log_files[-1].name

                    if choice_bytes is not None:
                        st.download_button(
                            "Скачать ⬇️",
                            bytes(choice_bytes),
                            file_name=filename,
                            key="dl_general",
                        )

        if st.session_state.get("adv") is not None:
            with st.expander("🖼 Показать визуализацию изображений и карт возмущений", expanded=True):
                imgs = st.session_state["imgs"]
                adv = st.session_state["adv"]
                targets = st.session_state.get("targets", imgs)
                N = imgs.size(0)
                n_show = min(8, N)

                is_impersonation = (mode == "Подмена личности")

                if is_impersonation:
                    header_cols = st.columns([1, 1, 1, 1, 0.3])
                    with header_cols[0]: st.markdown("**Оригинал**")
                    with header_cols[1]: st.markdown("**Цель**")
                    with header_cols[2]: st.markdown("**Атака**")
                    with header_cols[3]: st.markdown("**Разность**")
                    with header_cols[4]: st.markdown("**Успех**")
                else:
                    header_cols = st.columns([1, 1, 1, 0.3])
                    with header_cols[0]: st.markdown("**Оригинал**")
                    with header_cols[1]: st.markdown("**Атака**")
                    with header_cols[2]: st.markdown("**Разность**")
                    with header_cols[3]: st.markdown("**Успех**")

                heatmaps = []
                for i in range(n_show):
                    orig_np = tensor_to_np01(imgs[i])
                    adv_np = tensor_to_np01(adv[i])
                    target_np = tensor_to_np01(targets[i]) if is_impersonation else None

                    heatmap = make_diff_heatmap(
                        orig_np, adv_np,
                        cmap="gray_r",
                        vmax=epsilon
                    )
                    heatmaps.append(heatmap)

                    success = bool(st.session_state["metrics"]["Success"][i])

                    if is_impersonation:
                        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 0.3])
                        with c1: st.image(orig_np, width="stretch", clamp=True)
                        with c2: st.image(target_np, width="stretch", clamp=True)
                        with c3: st.image(adv_np, width="stretch", clamp=True)
                        with c4: st.image(np.array(heatmap), width="stretch", clamp=True)
                        with c5: st.markdown("✅" if success else "❌")
                    else:
                        c1, c2, c3, c4 = st.columns([1, 1, 1, 0.3])
                        with c1: st.image(orig_np, width="stretch", clamp=True)
                        with c2: st.image(adv_np, width="stretch", clamp=True)
                        with c3: st.image(np.array(heatmap), width="stretch", clamp=True)
                        with c4: st.markdown("✅" if success else "❌")

                st.session_state["heatmaps"] = heatmaps

    with tab_series:
        if not st.session_state.get("batch_ready", False):
            st.warning("Модель или датасет не готовы. Проверьте блок настроек сверху.")
            st.stop()

        colS1, colS2 = st.columns([1, 2])

        with colS1:
            st.markdown("#### Параметры серии")
            _container = st.container(horizontal=True, vertical_alignment="bottom")
            with _container:
                series_attack = st.selectbox(
                    "Атака:", ["FGSM", "PGD", "Carlini-Wagner", "PatchAttack"], key="series_attack"
                )
                param_options = {
                    "FGSM": [("epsilon", "ε")],
                    "PGD": [("epsilon", "ε"), ("alpha", "α"), ("iters", "Итераций")],
                    "Carlini-Wagner": [("epsilon", "ε"), ("c", "c"), ("iters", "Итераций")],
                    "PatchAttack": [("patch_size", "Размер патча"), ("lr", "LR"), ("iters", "Итераций")],
                }
                options = param_options[series_attack]
                chosen_label = st.selectbox(
                    "Варьируемый параметр:", [v for _, v in options], key="series_param"
                )
                param_name = next(k for k, v in options if v == chosen_label)

            series_mode = st.selectbox(
                "Режим:", ["Подмена личности", "Обход аутентификации"], key="series_mode"
            )
            run_series = st.button("🚀 Запустить серию", width="stretch", key="run_series_btn")
            with st.expander("Настройки серии", expanded=True):
                PARAM_SPECS = {
                    "epsilon":    {"lo": 0.0,   "hi": 0.5,  "default_range": (0.01, 0.10), "default_step": 0.01,  "decimals": 3},
                    "alpha":      {"lo": 0.0001,"hi": 0.1,  "default_range": (0.005, 0.05),"default_step": 0.005, "decimals": 4},
                    "iters":      {"lo": 1,     "hi": 1000, "default_range": (50, 500),    "default_step": 50,    "decimals": 0},
                    "c":          {"lo": 0.01,  "hi": 100.0,"default_range": (0.1, 10.0),  "default_step": 0.5,   "decimals": 2},
                    "lr":         {"lo": 1e-5,  "hi": 1.0,  "default_range": (0.001, 0.05),"default_step": 0.005, "decimals": 4},
                    "patch_size": {"lo": 8,     "hi": 128,  "default_range": (32, 96),     "default_step": 8,     "decimals": 0},
                }
                spec = PARAM_SPECS[param_name]

                rng = st.slider(
                    f"Диапазон {chosen_label}",
                    min_value=spec["lo"],
                    max_value=spec["hi"],
                    value=spec["default_range"],
                    step=spec["default_step"],
                    format=f"%.{spec['decimals']}f",
                    key=f"{param_name}_range_s",
                )

                step_options = {
                    "epsilon":    [0.005, 0.01, 0.02, 0.05, 0.1],
                    "alpha":      [0.001, 0.005, 0.01, 0.02, 0.05],
                    "iters":      [5, 10, 25, 50, 100, 200],
                    "c":          [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
                    "lr":         [0.001, 0.005, 0.01, 0.02, 0.05],
                    "patch_size": [1, 2, 4, 8, 16, 32],
                }
                p_step = st.select_slider(
                    f"Шаг {chosen_label}",
                    options=step_options[param_name],
                    value=spec["default_step"],
                    key=f"{param_name}_step_s",
                )

                p_min, p_max = rng
                if param_name in ("iters", "patch_size"):
                    p_min = int(p_min)
                    p_max = int(p_max)
                    p_step = int(p_step)

                fixed_params = {
                    "thr": st.slider("Порог сходства (серия)", 0.0, 1.0, 0.5, 0.01, key="thr_series")
                }

                if series_attack == "Carlini-Wagner":
                    defaults = {"epsilon": 0.08, "alpha": 0.01, "iters": 500, "c": 1.0, "lr": 0.01, "patch_size": 32}
                elif series_attack == "PatchAttack":
                    defaults = {"epsilon": 0.08, "alpha": 0.01, "iters": 300, "c": 1.0, "lr": 0.005, "patch_size": 64}
                else:
                    defaults = {"epsilon": 0.08, "alpha": 0.01, "iters": 15, "c": 1.0, "lr": 0.01, "patch_size": 32}

                _container = st.container(horizontal=True, vertical_alignment="bottom")
                with _container:
                    for key, val in defaults.items():
                        if key != param_name:
                            if key == "iters":
                                fixed_params[key] = st.number_input(
                                    f"{key}", value=int(val), min_value=1, max_value=1000, key=f"fix_{key}"
                                )
                            elif key == "patch_size":
                                fixed_params[key] = st.number_input(
                                    f"{key}", value=int(val), min_value=4, max_value=128, key=f"fix_{key}"
                                )
                            else:
                                fixed_params[key] = st.number_input(
                                    f"{key}", value=float(val), format="%.4f", key=f"fix_{key}"
                                )

        with colS2:
            st.markdown("#### 📊 Результаты")
            if param_name in ("iters", "patch_size"):
                param_values = list(range(int(p_min), int(p_max) + 1, int(p_step)))
            else:
                param_values = np.arange(p_min, p_max + 1e-12, p_step).tolist()

            if not param_values:
                st.error("Диапазон пуст.")
                st.stop()

            if run_series:
                model = st.session_state["custom_model"]
                imgs = st.session_state["imgs"]

                targets = torch.roll(imgs, shifts=1, dims=0) if series_mode == "Подмена личности" else imgs.clone()
                emb_targets = model.get_embeddings(targets)

                results = run_attack_series(
                    model=model,
                    imgs=imgs,
                    emb_targets=emb_targets,
                    attack_type=series_attack,
                    mode=series_mode,
                    param_name=param_name,
                    param_values=param_values,
                    fixed_params=fixed_params,
                    show_progress=True
                )
                st.session_state["series_results"] = results
                st.session_state["series_param_label"] = chosen_label

            if st.session_state.get("series_results") is not None:
                import plotly.graph_objects as go
                results = st.session_state["series_results"]
                x = results["param"]
                x_label = st.session_state.get("series_param_label", "Параметр")
                with st.expander(f"Метрики устойчивости", expanded=True):
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=x, y=results["ASR"], name="ASR", mode="lines+markers"))
                    fig1.add_trace(go.Scatter(x=x, y=results["AD"], name="Accuracy Drop", mode="lines+markers"))
                    fig1.add_trace(go.Scatter(x=x, y=results["CS"], name="Confidence Shift", mode="lines+markers"))
                    fig1.update_layout(
                        xaxis_title=x_label, yaxis_title="Значение",
                        height=450, hovermode="x unified",
                    )
                    st.plotly_chart(fig1, width="stretch")
                with st.expander(f"Метрики искажения", expanded=False):
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=x, y=results["Linf_mean"], name="L∞", mode="lines+markers"))
                    fig2.add_trace(go.Scatter(x=x, y=results["L2_mean"], name="L2", mode="lines+markers"))
                    fig2.update_layout(
                        xaxis_title=x_label, yaxis_title="Значение",
                        height=450, hovermode="x unified",
                    )
                    st.plotly_chart(fig2, width="stretch")
                with st.expander(f"Косинусное сходство", expanded=False):
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=x, y=results["Scos_mean"], name="Scos", mode="lines+markers"))
                    fig3.update_layout(
                        xaxis_title=x_label, yaxis_title="Scos", height=400,
                    )
                    st.plotly_chart(fig3, width="stretch")
                st.dataframe(pd.DataFrame(results), width="stretch")

                csv_bytes = pd.DataFrame(results).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Скачать результаты серии (.csv)",
                    csv_bytes,
                    file_name=f"series_{param_name}.csv",
                    key="dl_series",
                )