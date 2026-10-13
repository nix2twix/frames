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
from source.attacks import (
    fgsm_embed, pgd_embed, cw_embed,
    geometric_attack_embed
)
from source.dataset import (
    load_dataset_from_zip, validate_data_root, get_dataloader
)
from source.visualisation import (
    make_diff_heatmap, tensor_to_np01, draw_face_mesh_on_image
)

from source.configuration import (
    ConfigurationError, create_config, load_config, serialize_config,
)
from source.metrics import (
    l0_count_per_image, l2_per_image, linf_per_image,
    cosine_similarity_batch, attack_success_rate,
    predict_identity, geometric_delta, _extract_landmarks
)
from source.utils import (
    export_metrics_to_csv, export_heatmaps_to_zip, export_adversarial_to_zip,
    save_uploaded_file, export_adversarial_dataset, export_metrics_to_pdf,
    run_attack_series, export_metrics_to_html, init_session, clear_attack_result
)
from styles.style import load_styles, load_header

def apply_uploaded_config_general():
    """Применяет загруженную конфигурацию до создания виджетов."""

    uploaded = st.session_state.get("config_upload_general")
    if uploaded is None:
        return

    try:
        config = load_config(uploaded)
        attack = config["attack"]
        scenario = config["scenario"]
        params = attack["parameters"]

        # Переключаем тип атаки на указанный в конфигурации.
        st.session_state["attack_type_general"] = attack["type"]

        # Сбрасываем параметры к значениям по умолчанию,
        # чтобы не сохранялись настройки предыдущей атаки.
        st.session_state["eps_general"] = params.get("epsilon", 0.08)
        st.session_state["alpha_general"] = params.get("alpha", 0.01)
        st.session_state["iters_general"] = (
            params["iters"] if attack["type"] == "PGD" else 15
        )
        st.session_state["c_general"] = params.get("c", 1.0)
        st.session_state["lr_general"] = params.get("lr", 0.01)
        st.session_state["iters_cw_general"] = (
            params["iters"]
            if attack["type"] == "Carlini-Wagner"
            else 100
        )
        st.session_state["strength_geo"] = params.get("strength", 0.5)

        st.session_state["mode_general"] = (
            "Подмена личности"
            if scenario["type"] == "impersonation"
            else "Обход аутентификации"
        )
        st.session_state["thr_general"] = scenario["threshold"]

        st.session_state["loaded_config_general"] = config
        st.session_state["config_status_general"] = (
            "success",
            "Конфигурация загружена и применена."
        )

    except ConfigurationError as exc:
        st.session_state["config_status_general"] = (
            "error",
            f"Не удалось загрузить конфигурацию: {exc}"
        )


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
        ["FaceNet", "Custom"],
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
    batch_size = st.slider("Размер батча", 2, 30, 10, key="batch_size_slider")

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
        clear_attack_result()
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
                ["FGSM", "PGD", "Carlini-Wagner", 
                "Geometric Attack"],
                key="attack_type_general",
            )

            with st.expander("Настройки атаки", expanded=True):
                epsilon = st.slider("Макс. возмущение ε", 0.0, 0.3, 0.08, 0.01, key="eps_general")
                alpha = 0.01
                iters = 15
                c = 1.0
                lr = 0.01

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
                    lr = st.number_input(
                        "LR",
                        value=0.01,
                        min_value=0.00001,
                        max_value=1.0,
                        step=0.001,
                        format="%.5f",
                        key="lr_general",
                    )
                    iters = st.slider("Итераций", 10, 1000, 100, 10, key="iters_cw_general")
                
                elif attack_type == "Geometric Attack":
                    geo_strength = st.slider(
                        "Сила деформации", 0.0, 1.0, 0.5, 0.05,
                        key="strength_geo",
                        help="0 — без деформации, 1 — полная замена геометрии целью",
                    )
                mode = st.selectbox(
                    "Режим атаки", ["Подмена личности", "Обход аутентификации"],
                    key="mode_general",
                )
                thr = st.slider("Порог сходства", 0.0, 1.0, 0.5, 0.01, key="thr_general")

            run = st.button(
                label="🚀 Запустить атаку",
                width="stretch",
                disabled=not st.session_state.get("batch_ready", False),
                key="run_attack_btn",
                type="primary"
            )

            uploaded_config = st.file_uploader(
                "Загрузить конфигурацию",
                type=["json"],
                key="config_upload_general",
                on_change=apply_uploaded_config_general,
            )

            config_status = st.session_state.get(
                "config_status_general"
            )

            if config_status is not None:
                status_type, status_message = config_status

                if status_type == "success":
                    st.success(status_message)
                else:
                    st.error(status_message)

            # Собираем параметры только выбранного метода атаки.
            if attack_type == "FGSM":
                current_attack_params = {
                    "epsilon": epsilon,
                }

            elif attack_type == "PGD":
                current_attack_params = {
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "iters": iters,
                }

            elif attack_type == "Carlini-Wagner":
                current_attack_params = {
                    "epsilon": epsilon,
                    "c": c,
                    "lr": lr,
                    "iters": iters,
                }

            elif attack_type == "Geometric Attack":
                current_attack_params = {
                    "strength": geo_strength,
                }

            scenario_type = (
                "impersonation"
                if mode == "Подмена личности"
                else "deauth"
            )

            current_config = create_config(
                attack_type=attack_type,
                parameters=current_attack_params,
                scenario_type=scenario_type,
                threshold=thr,
            )

            st.download_button(
                "⬇️ Сохранить конфигурацию",
                data=serialize_config(current_config),
                file_name="frames_config.json",
                mime="application/json",
                key="download_config_general",
                width="stretch",
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

                geometric_valid = torch.ones(imgs.size(0), dtype=torch.bool, device=imgs.device)
                t0 = time.time()
                if attack_type == "FGSM":
                    adv = fgsm_embed(
                        model, imgs, emb_targets, epsilon,
                        mode="impersonation" if mode == "Подмена личности" else "deauth"
                    )

                elif attack_type == "PGD":
                    adv = pgd_embed(
                        model, imgs, emb_targets, epsilon, alpha, iters,
                        mode="impersonation" if mode == "Подмена личности" else "deauth",
                        threshold=thr,
                    )
                    
                elif attack_type == "Carlini-Wagner":
                    adv = cw_embed(
                        model, imgs, emb_targets,
                        c=c,
                        lr=lr,
                        iters=iters,
                        epsilon=epsilon,
                        mode="impersonation" if mode == "Подмена личности" else "deauth",
                        threshold=thr,
                    )
                
                elif attack_type == "Geometric Attack":
                    adv, geometric_valid = geometric_attack_embed(
                        model, imgs, targets,
                        strength=geo_strength,
                        mode="impersonation" if mode == "Подмена личности" else "deauth",
                        return_valid_mask=True,
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
                gd_delta = geometric_delta(imgs, adv)
                l0 = l0_count_per_image(imgs, adv)
                l2 = l2_per_image(imgs, adv)
                linf = linf_per_image(imgs, adv)

                success_arr, asr = attack_success_rate(
                    emb_before, emb_after, emb_targets,
                    threshold=thr,
                    mode="impersonation" if mode == "Подмена личности" else "deauth",
                )
                # A geometric attack without detectable landmarks is a failed
                # attack generation, not a valid unchanged adversarial sample.
                if attack_type == "Geometric Attack":
                    valid_np = geometric_valid.detach().cpu().numpy().astype(bool)
                    success_arr = success_arr & valid_np
                    asr = float(success_arr.sum() / success_arr.size)
                robustness = 1.0 - asr

                N = imgs.size(0)
                s_self = cosine_similarity_batch(emb_before, emb_after)
                delta_s_self = 1.0 - s_self
                delta_s_target = sim_after - sim_before

                valid_np = (
                    geometric_valid.detach().cpu().numpy()
                    if attack_type == "Geometric Attack"
                    else np.ones(N, dtype=bool)
                )

                metrics = {
                    "L0": l0,
                    "L2": l2,
                    "Linf": linf,

                    "S": sim_before,
                    "S_prime": sim_after,
                    "delta_S": delta_s_target,

                    "S_self": s_self,
                    "delta_S_self": delta_s_self,

                    "GD": gd_delta,

                    "Success": success_arr,

                    "Source": labels_np,
                    "Target": target_labels,
                    "PredictedBefore": pred_before,
                    "PredictedAfter": pred_after,
                    "Match": pred_after == target_labels,
                    "Valid": valid_np,
                }

                df = pd.DataFrame({
                    "L₀": l0,
                    "L₂": l2,
                    "L∞": linf,

                    "S": sim_before,
                    "S'": sim_after,
                    "ΔS": delta_s_target,

                    "S_self": s_self,
                    "ΔS_self": delta_s_self,

                    "ΔGD": gd_delta,

                    "Source": labels_np,
                    "Target": target_labels,
                    "PredictedBefore": pred_before,
                    "PredictedAfter": pred_after,
                    "Success": success_arr,
                    "Match": pred_after == target_labels,
                    "Valid": valid_np,
                })

                result_params = {
                    "attack_type": attack_type,
                    "mode": mode,
                    "threshold": thr,
                }

                if attack_type == "FGSM":
                    result_params["epsilon"] = epsilon

                elif attack_type == "PGD":
                    result_params["epsilon"] = epsilon
                    result_params["alpha"] = alpha
                    result_params["iters"] = iters

                elif attack_type == "Carlini-Wagner":
                    result_params["epsilon"] = epsilon
                    result_params["c"] = c
                    result_params["lr"] = lr
                    result_params["iters"] = iters

                elif attack_type == "Geometric Attack":
                    result_params["strength"] = geo_strength

                st.session_state.update({
                    "adv": adv,
                    "targets": targets,
                    "emb_targets": emb_targets,
                    "emb_after": emb_after,

                    "metrics": metrics,
                    "df": df,

                    "asr": asr,
                    "robustness": robustness,
                    "elapsed": elapsed,

                    "heatmaps": None,

                    "result_attack_type": attack_type,
                    "result_mode": mode,
                    "result_params": result_params,
                    "result_config": current_config,
                })

                logging.info(
                    f"Attack: model={model_name}, attack={attack_type}, mode={mode}, "
                    f"eps={epsilon}, N={N}, ASR={asr:.4f}"
                )

            if (
                st.session_state.get("adv") is not None
                and st.session_state.get("metrics") is not None
            ):
                colM1, colM2 = st.columns([2, 1], vertical_alignment="top")
                with colM1:
                    rob_val = float(st.session_state["robustness"])
                    rob_pct = rob_val * 100.0
                    st.metric("Устойчивость модели", f"")
                    rob_val = float(st.session_state["robustness"])
                    rob_pct = max(0.0, min(100.0, rob_val * 100.0))
                    r, g, b = int(255 * (1 - rob_val)), int(255 * rob_val), 0
                    fill_color = f"rgb({r},{g},{b})"
                    bg_color = "#444"
                    bar_h = 30
                    bar_html = f"""
                    <div style="width:100%; max-width:100%; box-sizing:border-box;">
                    <div style="width:100%; height:{bar_h}px; background:{bg_color}; border-radius:{bar_h/2}px; overflow:hidden; position:relative;">
                    <div style="height:100%; width:{rob_pct}%; background:{fill_color}; border-radius:{bar_h/2}px;"></div>
                    <div style="position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;
                                font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.6);pointer-events:none;">{rob_pct:.1f}%</div>
                    </div></div>"""
                    st.markdown(bar_html, unsafe_allow_html=True)
                with colM2:
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
                                sim_before=st.session_state["metrics"]["S"],
                                sim_after=st.session_state["metrics"]["S_prime"],
                                success_arr=st.session_state["metrics"]["Success"],
                                output_dir=tmpdir,
                            )
                        filename = "adversarial_dataset.zip"

                    elif export_choice == "Отчёт (.html)":
                        params = st.session_state["result_params"]
                        choice_bytes = export_metrics_to_html(
                            st.session_state["metrics"], model_name, attack_type,
                            float(st.session_state["asr"]),
                            float(st.session_state["elapsed"]),
                            params=params,
                        )
                        filename = "report.html"

                    elif export_choice == "Отчёт (.pdf)":
                        params = st.session_state["result_params"]
                        choice_bytes = export_metrics_to_pdf(
                            st.session_state["metrics"], model_name, attack_type,
                            float(st.session_state["asr"]),
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

                result_attack_type = st.session_state["result_attack_type"]
                result_mode = st.session_state["result_mode"]
                result_metrics = st.session_state["metrics"]

                N = imgs.size(0)
                n_show = min(8, N)

                is_impersonation = (result_mode == "Подмена личности")

                # Заголовки
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
                    adv_np  = tensor_to_np01(adv[i])
                    target_np = tensor_to_np01(targets[i]) if is_impersonation else None

                    orig_u8 = (np.clip(orig_np, 0, 1) * 255).astype(np.uint8)
                    adv_u8  = (np.clip(adv_np, 0, 1) * 255).astype(np.uint8)
                    target_u8 = (np.clip(target_np, 0, 1) * 255).astype(np.uint8) if is_impersonation else None

                    lm_orig = _extract_landmarks(orig_u8)
                    lm_adv  = _extract_landmarks(adv_u8)
                    lm_target = _extract_landmarks(target_u8) if is_impersonation else None

                    if lm_orig is not None:
                        orig_with_mesh = draw_face_mesh_on_image(
                            orig_np, lm_orig,
                            color=(0, 255, 0, 255),
                            line_color=(0, 255, 0, 120),
                            radius=2,
                        )
                    else:
                        orig_with_mesh = orig_np

                    if lm_adv is not None:
                        adv_with_mesh = draw_face_mesh_on_image(
                            adv_np, lm_adv,
                            color=(255, 0, 0, 255),
                            line_color=(255, 255, 255, 120),
                            radius=2,
                        )
                    else:
                        adv_with_mesh = adv_np

                    if is_impersonation and lm_target is not None:
                        target_with_mesh = draw_face_mesh_on_image(
                            target_np, lm_target,
                            color=(0, 150, 255, 255),
                            line_color=(0, 150, 255, 120),
                            radius=2,
                        )
                    else:
                        target_with_mesh = target_np

                    heatmap = make_diff_heatmap(
                        orig_np, adv_np,
                        cmap="inferno",
                        vmax=epsilon,
                        blur_radius=1, 
                    )
                    heatmaps.append(heatmap)

                    success = bool(result_metrics["Success"][i])

                    if is_impersonation:
                        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 0.3])
                        with c1: st.image(orig_with_mesh, width="stretch", clamp=True)
                        with c2: st.image(target_with_mesh, width="stretch", clamp=True)
                        with c3: st.image(adv_with_mesh, width="stretch", clamp=True)
                        with c4: st.image(np.array(heatmap), width="stretch", clamp=True)
                        with c5: st.markdown("✅" if success else "❌")
                    else:
                        c1, c2, c3, c4 = st.columns([1, 1, 1, 0.3])
                        with c1: st.image(orig_with_mesh, width="stretch", clamp=True)
                        with c2: st.image(adv_with_mesh, width="stretch", clamp=True)
                        with c3: st.image(np.array(heatmap), width="stretch", clamp=True)
                        with c4: st.markdown("✅" if success else "❌")

                    if lm_orig is not None and lm_adv is not None:
                        from source.metrics import _geometric_descriptors
                        g_o = _geometric_descriptors(lm_orig)
                        g_a = _geometric_descriptors(lm_adv)
                        gd_delta = float(np.mean([
                            abs(g_a[k] - g_o[k]) / max(g_o[k], 1e-6) for k in g_o
                        ]))

                    st.markdown("---")

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
                    "Атака:", ["FGSM", "PGD", "Carlini-Wagner"], key="series_attack"
                )
                param_options = {
                    "FGSM": [("epsilon", "ε")],
                    "PGD": [("epsilon", "ε"), ("alpha", "α"), ("iters", "Итераций")],
                    "Carlini-Wagner": [("epsilon", "ε"), ("c", "c"), ("iters", "Итераций")]
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
                    "lr":         {"lo": 1e-5,  "hi": 1.0,  "default_range": (0.001, 0.05),"default_step": 0.005, "decimals": 4}
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
                    "lr":         [0.001, 0.005, 0.01, 0.02, 0.05]
                }
                p_step = st.select_slider(
                    f"Шаг {chosen_label}",
                    options=step_options[param_name],
                    value=spec["default_step"],
                    key=f"{param_name}_step_s",
                )

                p_min, p_max = rng
                if param_name == "iters":
                    p_min = int(p_min)
                    p_max = int(p_max)
                    p_step = int(p_step)

                fixed_params = {
                    "thr": st.slider("Порог сходства (серия)", 0.0, 1.0, 0.5, 0.01, key="thr_series")
                }

                if series_attack == "Carlini-Wagner":
                    defaults = {"epsilon": 0.08, "alpha": 0.01, "iters": 500, "c": 1.0, "lr": 0.01}
                else:
                    defaults = {"epsilon": 0.08, "alpha": 0.01, "iters": 15, "c": 1.0, "lr": 0.01}

                _container = st.container(horizontal=True, vertical_alignment="bottom")
                with _container:
                    for key, val in defaults.items():
                        if key != param_name:
                            if key == "iters":
                                fixed_params[key] = st.number_input(
                                    f"{key}", value=int(val), min_value=1, max_value=1000, key=f"fix_{key}"
                                )
                            else:
                                fixed_params[key] = st.number_input(
                                    f"{key}", value=float(val), format="%.4f", key=f"fix_{key}"
                                )

        with colS2:
            st.markdown("#### 📊 Результаты")
            if param_name == "iters":
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
                    fig1.add_trace(
                        go.Scatter(
                            x=x,
                            y=results["Robustness"],
                            name="Устойчивость",
                            mode="lines+markers",
                        )
                    )

                    fig1.add_trace(
                        go.Scatter(
                            x=x,
                            y=results["ASR"],
                            name="ASR",
                            mode="lines+markers",
                        )
                    )
                    fig1.update_layout(
                        xaxis_title=x_label, yaxis_title="Значение",
                        height=450, hovermode="x unified",
                    )
                    st.plotly_chart(fig1, width="stretch")
                with st.expander(f"Метрики искажения", expanded=False):
                    fig2 = go.Figure()
                    fig2.add_trace(
                        go.Scatter(
                            x=x,
                            y=results["L0_mean"],
                            name="L₀",
                            mode="lines+markers",
                        )
                    )

                    fig2.add_trace(
                        go.Scatter(
                            x=x,
                            y=results["L2_mean"],
                            name="L₂",
                            mode="lines+markers",
                        )
                    )

                    fig2.add_trace(
                        go.Scatter(
                            x=x,
                            y=results["Linf_mean"],
                            name="L∞",
                            mode="lines+markers",
                        )
                    )
                    fig2.update_layout(
                        xaxis_title=x_label, yaxis_title="Значение",
                        height=450, hovermode="x unified",
                    )
                    st.plotly_chart(fig2, width="stretch")
                with st.expander(f"Косинусное сходство", expanded=False):
                    fig3 = go.Figure()
                    fig3.add_trace(
                        go.Scatter(
                            x=x,
                            y=results["S_mean"],
                            name="S",
                            mode="lines+markers",
                        )
                    )

                    fig3.add_trace(
                        go.Scatter(
                            x=x,
                            y=results["S_prime_mean"],
                            name="S'",
                            mode="lines+markers",
                        )
                    )

                    fig3.add_trace(
                        go.Scatter(
                            x=x,
                            y=results["delta_S_mean"],
                            name="ΔS",
                            mode="lines+markers",
                        )
                    )
                    fig3.update_layout(
                        xaxis_title=x_label, yaxis_title="Сходство с целью", height=400,
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