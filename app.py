"""ECOCANAL IA — aplicación Streamlit del proyecto académico.

Ejecución local:
    streamlit run app.py
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import ui
from src.config import (
    APP_DISCLAIMER,
    APP_NAME,
    BASE_DIR,
    CLASSIFICATION_THRESHOLD,
    DATASET_PATH,
    EXAMPLES_DIR,
    FEATURE_COLUMNS,
    FEATURE_HELP,
    FEATURE_LABELS,
    FEATURE_RANGES,
    FEATURE_SHORT_LABELS,
    LOGO_PATH,
    MODEL_LABELS,
    RISK_HIGH_MIN,
    RISK_LEVELS,
    RISK_LOW_MAX,
    TARGET_COLUMN,
)
from src.data_processing import (
    BatchReport,
    DataValidationError,
    make_template_dataframe,
    prepare_batch,
    read_tabular_file,
)
from src.evaluation import evaluate_batch_results
from src.predictions import (
    Artifacts,
    load_artifacts,
    predict_batch,
    predict_single,
)

st.set_page_config(
    page_title=f"{APP_NAME} | Canales y ríos",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "ECOCANAL IA — Prototipo académico de clasificación de riesgo de "
            "residuos en canales de agua. Datos sintéticos."
        )
    },
)

PAGES = [
    "Panel general",
    "Predicción individual",
    "Predicción por lotes",
    "Comparación de modelos",
    "Recomendaciones ambientales",
    "Metodología y documentación",
]

DEMO_CASES: dict[str, dict[str, float | int]] = {
    "Bajo": {
        "basura_acumulada": 12.0,
        "obstruccion": 18.0,
        "flujo_agua": 82.0,
        "frecuencia_lluvias": 8.0,
        "dias_sin_limpieza": 21,
        "residuos_plasticos": 120.0,
    },
    "Medio": {
        "basura_acumulada": 46.0,
        "obstruccion": 52.0,
        "flujo_agua": 49.0,
        "frecuencia_lluvias": 17.0,
        "dias_sin_limpieza": 165,
        "residuos_plasticos": 1200.0,
    },
    "Alto": {
        "basura_acumulada": 82.0,
        "obstruccion": 88.0,
        "flujo_agua": 14.0,
        "frecuencia_lluvias": 26.0,
        "dias_sin_limpieza": 410,
        "residuos_plasticos": 3600.0,
    },
}

MODEL_BATCH_COLUMNS = {
    "logistica": ("prediccion_logistica", "prob_regresion_logistica", "Greens"),
    "red_neuronal": (
        "prediccion_red_neuronal",
        "prob_red_neuronal",
        "Blues",
    ),
    "consenso": ("prediccion_consenso", "prob_consenso", "BuGn"),
}


@st.cache_resource(show_spinner="Cargando modelos de inteligencia artificial...")
def get_artifacts() -> Artifacts:
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    from src.data_processing import load_dataset

    return load_dataset(DATASET_PATH)


@st.cache_data(show_spinner="Analizando canales con ambos modelos...")
def get_analysis(_artifacts: Artifacts, dataset_version: int) -> pd.DataFrame:
    dataset = get_dataset()
    return predict_batch(dataset, _artifacts)


def dataset_version() -> int:
    return DATASET_PATH.stat().st_mtime_ns if DATASET_PATH.exists() else 0


def fmt_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.1%}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value < 1:
        return f"{value:.3f} s"
    return f"{value:.1f} s"


def risk_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["nivel_riesgo_consenso"].value_counts()
    return {level: int(counts.get(level, 0)) for level in RISK_LEVELS}


def apply_demo_case(case: str) -> None:
    for column, value in DEMO_CASES[case].items():
        st.session_state[f"single_{column}"] = value


def initialize_single_inputs() -> None:
    for column, value in DEMO_CASES["Medio"].items():
        st.session_state.setdefault(f"single_{column}", value)


def clear_single_result() -> None:
    st.session_state.pop("single_result", None)
    st.session_state.pop("single_input", None)


def retrain_models() -> None:
    from src.train_models import train_and_save

    with st.spinner("Entrenando y evaluando ambos modelos. Esto puede tardar 1-2 minutos..."):
        train_and_save(verbose=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.pop("single_result", None)
    st.session_state.pop("single_input", None)
    st.session_state.pop("batch_result", None)
    st.session_state["flash_success"] = "Modelos reentrenados y artefactos actualizados correctamente."
    st.rerun()


def sidebar(artifacts: Artifacts | None, rows: int | None, dataset_error: str | None) -> str:
    with st.sidebar:
        ui.sidebar_brand(str(LOGO_PATH), loaded=artifacts is not None, dataset_rows=rows)
        ui.sidebar_section("Navegación")
        page = st.radio(
            "Secciones",
            PAGES,
            label_visibility="collapsed",
            width="stretch",
            key="navigation",
        )
        st.markdown("---")
        ui.sidebar_section("Estado del sistema")
        if artifacts is not None:
            metadata = artifacts.metadata
            models = metadata.get("models", {})
            st.markdown(
                f"""
                <div style="font-size:.76rem;line-height:1.7;color:#C9E2D8;">
                    Regresión Logística · <span style="color:#7FE0C0;">activa</span><br>
                    Red Neuronal · <span style="color:#7FE0C0;">activa</span><br>
                    Dataset · <span style="color:#7FE0C0;">{rows:,} registros</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if models:
                best = metadata.get("best_model_by_f1", "logistica")
                st.markdown(
                    f'<div style="font-size:.72rem;color:#89A99B;margin-top:.45rem;">'
                    f"Mejor F1 en prueba: {MODEL_LABELS.get(best, best)}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:.76rem;color:#F0B4A8;">Modelos no cargados</div>',
                unsafe_allow_html=True,
            )
        if dataset_error:
            st.caption("Dataset no disponible")

        if artifacts is not None:
            with st.expander("Mantenimiento de modelos"):
                st.caption(
                    "Vuelve a ejecutar el entrenamiento reproducible con la misma "
                    "partición estratificada."
                )
                if st.button(
                    "Reentrenar ahora",
                    key="sidebar_retrain",
                    width="stretch",
                ):
                    try:
                        retrain_models()
                    except Exception as exc:  # noqa: BLE001  # pragma: no cover - entorno
                        st.error(f"No fue posible reentrenar: {exc}")

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:.68rem;color:#7EA091;line-height:1.55;">'
            f"{APP_NAME} · v1.0<br>Prototipo académico<br>Datos sintéticos</div>",
            unsafe_allow_html=True,
        )
    return page


def render_artifact_error(error: str) -> None:
    ui.page_header(
        "Sistema",
        "Los modelos aún no están disponibles",
        "La aplicación puede entrenarlos ahora mismo de forma reproducible.",
    )
    st.error(error)
    st.markdown(
        """
        Los artefactos esperados son `models/logistic_model.pkl`,
        `models/neural_network.keras` y `models/preprocessor.pkl`. El entrenamiento
        utiliza `data/dataset_canales.csv` y guarda también las métricas de prueba.
        """
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Entrenar modelos ahora", type="primary", width="stretch"):
            try:
                retrain_models()
            except Exception as exc:  # noqa: BLE001
                st.error(f"El entrenamiento falló: {exc}")
    with col2:
        st.code("python -m src.train_models", language="bash")
    ui.footer()


def dataset_unavailable(error: str) -> None:
    st.error(f"No se pudo cargar el dataset analítico: {error}")
    st.code("python -m src.generate_dataset", language="bash")
    st.caption(
        "Las páginas de predicción individual y por lotes siguen disponibles con "
        "los modelos ya entrenados."
    )


def render_dashboard(
    artifacts: Artifacts, dataset: pd.DataFrame | None, analysis: pd.DataFrame | None
) -> None:
    ui.hero()
    if analysis is None or dataset is None:
        dataset_unavailable(
            "el archivo data/dataset_canales.csv no está disponible o es inválido"
        )
        return

    counts = risk_counts(analysis)
    total = len(analysis)
    metadata = artifacts.metadata
    models = metadata.get("models", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.metric_card(
            "Canales analizados",
            f"{total:,}",
            "Registros sintéticos evaluados con ambos modelos",
            color="#0B3D2E",
        )
    with col2:
        ui.metric_card(
            "Riesgo bajo",
            f"{counts['Bajo']:,}",
            f"{counts['Bajo'] / total:.1%} del total",
            color="#1F8A70",
        )
    with col3:
        ui.metric_card(
            "Riesgo medio",
            f"{counts['Medio']:,}",
            f"{counts['Medio'] / total:.1%} del total",
            color="#E0A100",
        )
    with col4:
        ui.metric_card(
            "Riesgo alto",
            f"{counts['Alto']:,}",
            f"{counts['Alto'] / total:.1%} del total",
            color="#C0392B",
        )

    st.markdown("")
    left, right = st.columns([1, 1.55])
    with left, st.container(border=True):
        st.markdown("#### Distribución del riesgo consensuado")
        st.caption("Promedio de la probabilidad de ambos modelos por registro.")
        st.pyplot(ui.risk_donut(counts), width="stretch")
    with right, st.container(border=True):
        st.markdown("#### Distribución de probabilidades")
        st.caption(
            "Cada barra cuenta canales según la probabilidad estimada de riesgo alto."
        )
        st.pyplot(
            ui.probability_histogram(
                analysis["prob_regresion_logistica"].to_numpy(),
                analysis["prob_red_neuronal"].to_numpy(),
            ),
            width="stretch",
        )

    st.markdown("")
    chart_col, summary_col = st.columns([1.25, 1])
    with chart_col, st.container(border=True):
        st.markdown("#### Comparación rápida de modelos")
        st.caption("Métricas calculadas sobre el mismo conjunto de prueba.")
        st.pyplot(ui.metrics_comparison_chart(metadata), width="stretch")
    with summary_col:
        st.markdown("#### Lectura del sistema")
        if models:
            for key, label in MODEL_LABELS.items():
                metrics = models.get(key, {})
                ui.status_card(
                    label,
                    (
                        f"Accuracy {fmt_pct(metrics.get('accuracy'))} · "
                        f"F1 {fmt_pct(metrics.get('f1'))} · "
                        f"ROC-AUC {fmt_pct(metrics.get('roc_auc'))} · "
                        f"Entrenamiento {fmt_seconds(metrics.get('training_time_seconds'))}"
                    ),
                    color="#1F8A70" if key == "logistica" else "#1E6F9F",
                )
                st.markdown("")
            best_key = metadata.get("best_model_by_f1", "logistica")
            st.info(
                f"El mejor F1 corresponde a **{MODEL_LABELS.get(best_key, best_key)}**. "
                "La comparación completa, matrices de confusión e interpretación "
                "están disponibles en la sección Comparación de modelos."
            )
        else:
            st.warning("No hay metadatos de evaluación disponibles.")

    st.markdown("")
    with st.container(border=True):
        st.markdown("#### Canales simulados con mayor prioridad preventiva")
        st.caption(
            "Ordenados por probabilidad consensuada. Son registros sintéticos: no "
            "representan canales reales."
        )
        top = analysis.sort_values("prob_consenso", ascending=False).head(10).copy()
        top.insert(
            0,
            "Canal simulado",
            top["fila_original"].map(lambda value: f"SIM-{int(value):04d}"),
        )
        top["prob_consenso_mostrada"] = top["prob_consenso"] * 100
        top_view = top[
            [
                "Canal simulado",
                "prob_consenso_mostrada",
                "nivel_riesgo_consenso",
                "prioridad",
                "basura_acumulada",
                "obstruccion",
                "dias_sin_limpieza",
                "recomendacion_principal",
            ]
        ].rename(
            columns={
                "prob_consenso_mostrada": "Prob. consenso (%)",
                "nivel_riesgo_consenso": "Riesgo",
                "prioridad": "Prioridad",
                "basura_acumulada": "Basura (%)",
                "obstruccion": "Obstrucción (%)",
                "dias_sin_limpieza": "Días sin limpieza",
                "recomendacion_principal": "Recomendación principal",
            }
        )
        st.dataframe(
            top_view,
            hide_index=True,
            width="stretch",
            height=350,
            column_config={
                "Prob. consenso (%)": st.column_config.ProgressColumn(
                    "Prob. consenso (%)", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "Basura (%)": st.column_config.NumberColumn(format="%.1f"),
                "Obstrucción (%)": st.column_config.NumberColumn(format="%.1f"),
                "Días sin limpieza": st.column_config.NumberColumn(format="%d"),
                "Recomendación principal": st.column_config.TextColumn(width="large"),
            },
        )

    with st.expander("Análisis exploratorio del dataset sintético"):
        exp_left, exp_right = st.columns([1.15, 1])
        with exp_left:
            st.pyplot(ui.correlation_heatmap(dataset), width="stretch")
        with exp_right:
            summary = dataset[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].describe().T
            summary = summary[["mean", "std", "min", "50%", "max"]].rename(
                columns={
                    "mean": "Media",
                    "std": "Desv. est.",
                    "min": "Mínimo",
                    "50%": "Mediana",
                    "max": "Máximo",
                }
            )
            summary.index = [
                FEATURE_SHORT_LABELS.get(index, "Riesgo alto") for index in summary.index
            ]
            st.dataframe(summary.round(2), width="stretch")
            st.caption(
                f"Prevalencia real de riesgo alto en el dataset: "
                f"{dataset[TARGET_COLUMN].mean():.1%}. Datos simulados con fines académicos."
            )


def render_individual(artifacts: Artifacts) -> None:
    ui.page_header(
        "Predicción individual",
        "Evalúa un canal con los dos modelos",
        (
            "Ingresa las condiciones operativas del tramo. El sistema calculará la "
            "probabilidad con Regresión Logística y con la Red Neuronal, y mostrará "
            "una lectura consensuada con recomendaciones preventivas."
        ),
    )
    initialize_single_inputs()

    st.markdown("**Casos de demostración**")
    demo_cols = st.columns(3)
    for column, (label, caption) in zip(
        demo_cols,
        [
            ("Bajo", "Flujo alto y mantenimiento reciente"),
            ("Medio", "Acumulación moderada y limpieza distante"),
            ("Alto", "Obstrucción crítica y muchos días sin limpieza"),
        ],
    ):
        with column:
            st.button(
                f"Caso {label.lower()}",
                key=f"demo_{label}",
                on_click=apply_demo_case,
                args=(label,),
                width="stretch",
            )
            st.caption(caption)

    with st.form("individual_prediction_form", clear_on_submit=False):
        st.markdown("#### Condiciones del canal")
        first_row = st.columns(3)
        second_row = st.columns(3)
        widget_columns = [
            (first_row[0], "basura_acumulada"),
            (first_row[1], "obstruccion"),
            (first_row[2], "flujo_agua"),
            (second_row[0], "frecuencia_lluvias"),
            (second_row[1], "dias_sin_limpieza"),
            (second_row[2], "residuos_plasticos"),
        ]
        for container, feature in widget_columns:
            low, high = FEATURE_RANGES[feature]
            step = 1 if feature == "dias_sin_limpieza" else (
                10.0 if feature == "residuos_plasticos" else 1.0
            )
            with container:
                if feature == "dias_sin_limpieza":
                    st.number_input(
                        FEATURE_LABELS[feature],
                        min_value=int(low),
                        max_value=int(high),
                        step=int(step),
                        help=FEATURE_HELP[feature],
                        key=f"single_{feature}",
                        format="%d",
                    )
                else:
                    st.number_input(
                        FEATURE_LABELS[feature],
                        min_value=float(low),
                        max_value=float(high),
                        step=float(step),
                        help=FEATURE_HELP[feature],
                        key=f"single_{feature}",
                        format="%.1f",
                    )
        st.caption(
            "Los rangos son límites operativos del prototipo académico. "
            "El resultado es una estimación, no una medición ambiental."
        )
        submitted = st.form_submit_button(
            "PREDECIR RIESGO",
            type="primary",
            width="stretch",
        )

    if submitted:
        values = {
            feature: st.session_state[f"single_{feature}"]
            for feature in FEATURE_COLUMNS
        }
        try:
            result = predict_single(values, artifacts)
        except DataValidationError as exc:
            st.error(f"Revisa los datos ingresados: {exc}")
        else:
            st.session_state["single_result"] = result
            st.session_state["single_input"] = values

    result = st.session_state.get("single_result")
    if not result:
        return

    st.markdown("---")
    title_col, clear_col = st.columns([4, 1])
    with title_col:
        st.markdown("#### Resultado de la predicción")
    with clear_col:
        st.button(
            "Limpiar resultado",
            on_click=clear_single_result,
            width="stretch",
        )

    consensus = result["consenso"]
    agreement = (
        "Los modelos coinciden en la clasificación binaria."
        if consensus["modelos_coinciden"]
        else "Los modelos no coinciden; conviene validar en campo."
    )
    ui.result_banner(
        consensus["nivel_riesgo"],
        consensus["probabilidad"],
        f"Prioridad {consensus['prioridad']} · {agreement}",
    )

    model_cols = st.columns(2)
    for column, key, color in (
        (model_cols[0], "logistica", "#1F8A70"),
        (model_cols[1], "red_neuronal", "#1E6F9F"),
    ):
        model_result = result[key]
        with column, st.container(border=True):
            st.markdown(f"#### {MODEL_LABELS[key]}")
            st.markdown(
                ui.risk_badge(model_result["nivel_riesgo"]),
                unsafe_allow_html=True,
            )
            ui.probability_bar(
                "Probabilidad de riesgo alto",
                model_result["probabilidad"],
                model_result["nivel_riesgo"],
            )
            st.markdown(
                f"<div style='font-size:.82rem;color:#5C746A;'>"
                f"Clasificación binaria: <strong style='color:{color};'>"
                f"{model_result['etiqueta_prediccion']}</strong> "
                f"(umbral {CLASSIFICATION_THRESHOLD:.2f}).</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Probabilidad estimada: {model_result['probabilidad']:.4f}"
            )

    st.markdown("#### Recomendaciones preventivas")
    for index, recommendation in enumerate(result["recomendaciones"], start=1):
        ui.recommendation_card(recommendation, index=index)

    with st.expander("Ver variables ingresadas y criterio de clasificación"):
        input_frame = pd.DataFrame(
            {
                "Variable": [FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS],
                "Valor": [
                    result["entrada"][feature] for feature in FEATURE_COLUMNS
                ],
            }
        )
        st.dataframe(input_frame, hide_index=True, width="stretch")
        st.markdown(
            f"""
            - **Riesgo bajo:** probabilidad menor a **{RISK_LOW_MAX:.2f}**.
            - **Riesgo medio:** probabilidad entre **{RISK_LOW_MAX:.2f}** y **{RISK_HIGH_MIN:.2f}**.
            - **Riesgo alto:** probabilidad igual o superior a **{RISK_HIGH_MIN:.2f}**
              (mismo umbral usado para la variable objetivo binaria).

            La probabilidad consensuada es el promedio simple de ambos modelos.
            La clasificación no reemplaza una inspección técnica.
            """
        )


def render_metrics_table(metrics: Mapping[str, Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    labels = {
        "logistica": "Regresión Logística",
        "red_neuronal": "Red Neuronal",
        "consenso": "Consenso (promedio)",
    }
    for key, label in labels.items():
        if key not in metrics:
            continue
        item = metrics[key]
        rows.append(
            {
                "Modelo": label,
                "Accuracy": item.get("accuracy"),
                "Precision": item.get("precision"),
                "Recall": item.get("recall"),
                "F1-score": item.get("f1"),
                "ROC-AUC": item.get("roc_auc"),
                "Registros evaluados": item.get("n_samples"),
            }
        )
    metric_frame = pd.DataFrame(rows)
    st.dataframe(
        metric_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1-score": st.column_config.NumberColumn(format="%.3f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Registros evaluados": st.column_config.NumberColumn(format="%d"),
        },
    )


def render_batch_results(
    result: pd.DataFrame,
    report: BatchReport,
    metrics: Mapping[str, Mapping[str, Any]] | None,
    source_name: str,
) -> None:
    st.markdown("---")
    st.markdown(f"#### Resultados del lote · {source_name}")
    st.caption(
        f"Procesado el {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "Las predicciones se generan con ambos modelos y una lectura consensuada."
    )

    valid_rows = len(result)
    high_count = int((result["nivel_riesgo_consenso"] == "Alto").sum())
    metric_cols = st.columns(4)
    with metric_cols[0]:
        ui.metric_card("Filas leídas", f"{report.total_rows:,}", "Registros del archivo", "#0B3D2E")
    with metric_cols[1]:
        ui.metric_card("Filas procesadas", f"{valid_rows:,}", "Con predicción generada", "#1F8A70")
    with metric_cols[2]:
        ui.metric_card("Filas omitidas", f"{report.invalid_rows:,}", "Datos no numéricos o vacíos", "#C0392B")
    with metric_cols[3]:
        ui.metric_card("Riesgo alto", f"{high_count:,}", f"{high_count / valid_rows:.1%} del lote", "#E0A100")

    if report.missing_values:
        detail = ", ".join(
            f"{FEATURE_LABELS.get(key, key)}: {value}"
            for key, value in report.missing_values.items()
        )
        st.warning(
            f"Se imputaron valores faltantes en {report.imputed_rows} fila(s) usando "
            f"medianas del entrenamiento. Detalle: {detail}."
        )
    if report.out_of_range_values:
        detail = ", ".join(
            f"{FEATURE_LABELS.get(key, key)}: {value}"
            for key, value in report.out_of_range_values.items()
        )
        st.warning(
            f"Se limitaron valores fuera de rango en {report.limited_rows} fila(s). "
            f"Detalle: {detail}. Revisa la columna 'valores_limitados'."
        )
    if report.invalid_rows:
        st.error(
            f"Se omitieron {report.invalid_rows} fila(s) con valores no numéricos o "
            "sin ningún dato de entrada."
        )
        with st.expander("Ver filas omitidas"):
            st.dataframe(report.invalid_detail, hide_index=True, width="stretch")

    chart_col, dist_col = st.columns([1.2, 1])
    with chart_col, st.container(border=True):
        st.markdown("##### Probabilidades del lote")
        st.pyplot(
            ui.probability_histogram(
                result["prob_regresion_logistica"].to_numpy(),
                result["prob_red_neuronal"].to_numpy(),
            ),
            width="stretch",
        )
    with dist_col, st.container(border=True):
        st.markdown("##### Riesgo consensuado")
        counts = result["nivel_riesgo_consenso"].value_counts()
        for level in RISK_LEVELS:
            ui.priority_bar(level, int(counts.get(level, 0)), valid_rows)

    display_columns = [
        "fila_original",
        *FEATURE_COLUMNS,
        "prob_regresion_logistica",
        "prob_red_neuronal",
        "prob_consenso",
        "nivel_riesgo_consenso",
        "prioridad",
        "modelos_coinciden",
        "datos_imputados",
        "valores_limitados",
        "recomendacion_principal",
    ]
    if TARGET_COLUMN in result.columns:
        display_columns.insert(-2, TARGET_COLUMN)
    table = result[display_columns].copy()
    for probability_column in (
        "prob_regresion_logistica",
        "prob_red_neuronal",
        "prob_consenso",
    ):
        table[probability_column] = table[probability_column] * 100
    rename_map = {
        "fila_original": "Fila original",
        TARGET_COLUMN: "Riesgo alto real",
        "prob_regresion_logistica": "Prob. logística (%)",
        "prob_red_neuronal": "Prob. red neuronal (%)",
        "prob_consenso": "Prob. consenso (%)",
        "nivel_riesgo_consenso": "Riesgo",
        "prioridad": "Prioridad",
        "modelos_coinciden": "Modelos coinciden",
        "datos_imputados": "Datos imputados",
        "valores_limitados": "Valores limitados",
        "recomendacion_principal": "Recomendación principal",
    }
    rename_map.update(
        {feature: FEATURE_SHORT_LABELS[feature] for feature in FEATURE_COLUMNS}
    )
    table = table.rename(columns=rename_map)

    st.markdown("#### Tabla de predicciones")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=430,
        column_config={
            "Prob. logística (%)": st.column_config.ProgressColumn(
                min_value=0.0, max_value=100.0, format="%.1f%%"
            ),
            "Prob. red neuronal (%)": st.column_config.ProgressColumn(
                min_value=0.0, max_value=100.0, format="%.1f%%"
            ),
            "Prob. consenso (%)": st.column_config.ProgressColumn(
                min_value=0.0, max_value=100.0, format="%.1f%%"
            ),
            "Recomendación principal": st.column_config.TextColumn(width="large"),
        },
    )

    csv_data = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar resultados completos (CSV)",
        data=csv_data,
        file_name=f"ecocanal_predicciones_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("#### Evaluación con etiquetas reales")
    if metrics:
        n_eval = int(
            metrics.get("logistica", metrics.get("red_neuronal", {})).get(
                "n_samples", 0
            )
        )
        st.info(
            f"Se encontró la columna `{TARGET_COLUMN}`. Las métricas se calcularon "
            f"únicamente con **{n_eval}** etiquetas reales válidas del archivo; "
            "nunca se usaron las predicciones como etiquetas."
        )
        render_metrics_table(metrics)
        matrix_cols = st.columns(3)
        for column, key in zip(matrix_cols, ["logistica", "red_neuronal", "consenso"]):
            if key not in metrics:
                continue
            label, _, cmap = MODEL_BATCH_COLUMNS[key]
            with column:
                st.pyplot(
                    ui.confusion_matrix_heatmap(
                        metrics[key]["confusion_matrix"],
                        f"Matriz de confusión · {label}",
                        color=cmap,
                    ),
                    width="stretch",
                )
    else:
        st.info(
            f"El archivo no contiene la columna objetivo `{TARGET_COLUMN}` con "
            "etiquetas válidas. No se calculan Accuracy, Precision, Recall, F1 ni "
            "matriz de confusión. A continuación se muestran estadísticas descriptivas."
        )
        descriptive = result[list(FEATURE_COLUMNS)].describe().T
        descriptive = descriptive[["mean", "std", "min", "50%", "max"]].rename(
            columns={
                "mean": "Media",
                "std": "Desv. est.",
                "min": "Mínimo",
                "50%": "Mediana",
                "max": "Máximo",
            }
        )
        descriptive.index = [
            FEATURE_LABELS.get(index, index) for index in descriptive.index
        ]
        st.dataframe(descriptive.round(2), width="stretch")


def render_batch() -> None:
    ui.page_header(
        "Predicción por lotes",
        "Procesa varios canales desde CSV o Excel",
        (
            "Carga un archivo con las seis variables obligatorias. El sistema valida "
            "columnas, valores faltantes y rangos; luego predice con ambos modelos y "
            "permite descargar los resultados."
        ),
    )

    template = make_template_dataframe()
    template_bytes = template.to_csv(index=False).encode("utf-8-sig")
    example_path = EXAMPLES_DIR / "ejemplo_lote_canales.csv"
    example_label_path = EXAMPLES_DIR / "ejemplo_lote_etiquetado.csv"

    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            "Descargar plantilla CSV",
            data=template_bytes,
            file_name="plantilla_ecocanal.csv",
            mime="text/csv",
            width="stretch",
        )
    with download_cols[1]:
        if example_path.exists():
            st.download_button(
                "Descargar lote de ejemplo",
                data=example_path.read_bytes(),
                file_name=example_path.name,
                mime="text/csv",
                width="stretch",
            )
        else:
            st.caption("Lote de ejemplo no disponible.")
    with download_cols[2]:
        if example_label_path.exists():
            st.download_button(
                "Descargar ejemplo etiquetado",
                data=example_label_path.read_bytes(),
                file_name=example_label_path.name,
                mime="text/csv",
                width="stretch",
            )
        else:
            st.caption("Ejemplo etiquetado no disponible.")

    source_mode = st.radio(
        "Origen de los datos",
        ["Subir archivo", "Usar lote de ejemplo"],
        horizontal=True,
        key="batch_source_mode",
    )
    uploaded = None
    source_name = ""
    source: Any = None

    if source_mode == "Subir archivo":
        uploaded = st.file_uploader(
            "Archivo de canales",
            type=["csv", "xlsx", "xls"],
            help="Debe incluir las columnas del dataset. Máximo 50 000 filas.",
        )
        if uploaded is not None:
            if uploaded.size > 25 * 1024 * 1024:
                st.error("El archivo supera el límite de 25 MB.")
            else:
                source = uploaded
                source_name = uploaded.name
    else:
        use_labels = st.checkbox(
            "Usar la versión del ejemplo con etiquetas reales (para probar métricas)",
            value=False,
        )
        chosen_example = example_label_path if use_labels else example_path
        if chosen_example.exists():
            source = chosen_example
            source_name = chosen_example.name
            if use_labels:
                st.info(
                    "Se usará el lote de ejemplo con la columna objetivo `riesgo_alto`. "
                    "Así podrás comprobar Accuracy, Precision, Recall, F1 y las matrices "
                    "de confusión."
                )
            else:
                st.info(
                    "Se usará un lote sintético de demostración sin columna objetivo. "
                    "Marca la casilla para probar también las métricas con etiquetas reales."
                )
        else:
            st.error("No se encontró el lote de ejemplo en examples/.")

    must_have = ", ".join(f"`{column}`" for column in FEATURE_COLUMNS)
    st.caption(f"Columnas obligatorias: {must_have}. Objetivo opcional: `{TARGET_COLUMN}`.")

    if st.button(
        "PROCESAR LOTE",
        type="primary",
        width="stretch",
        disabled=source is None,
    ):
        try:
            raw = read_tabular_file(source, filename=source_name)
            valid, report = prepare_batch(raw, artifacts_medians())
            prediction = predict_batch(
                valid,
                get_artifacts(),
                target_column=TARGET_COLUMN if TARGET_COLUMN in valid.columns else None,
            )
            metrics = evaluate_batch_results(prediction, TARGET_COLUMN)
        except DataValidationError as exc:
            st.error(f"Validación del archivo: {exc}")
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - protección de la interfaz
            st.error(f"No fue posible procesar el archivo: {exc}")
        else:
            st.session_state["batch_result"] = prediction
            st.session_state["batch_report"] = report
            st.session_state["batch_metrics"] = metrics
            st.session_state["batch_source_name"] = source_name

    if "batch_result" in st.session_state:
        render_batch_results(
            st.session_state["batch_result"],
            st.session_state["batch_report"],
            st.session_state.get("batch_metrics"),
            st.session_state.get("batch_source_name", "archivo"),
        )


def artifacts_medians() -> dict[str, float]:
    """Medianas del preprocesador para imputar lotes."""
    return get_artifacts().medians


def render_model_comparison(artifacts: Artifacts) -> None:
    ui.page_header(
        "Comparación de modelos",
        "Regresión Logística frente a Red Neuronal Artificial",
        (
            "Ambos modelos comparten las mismas variables, el mismo escalador "
            "ajustado en entrenamiento y el mismo conjunto de prueba estratificado. "
            "Las métricas provienen de registros no vistos durante el entrenamiento."
        ),
    )
    metadata = artifacts.metadata
    models = metadata.get("models", {})
    split = metadata.get("split", {})
    if not models:
        st.warning(
            "No se encontraron metadatos de evaluación. Reentrena los modelos desde "
            "la barra lateral."
        )
        return

    best_key = metadata.get("best_model_by_f1", "logistica")
    best = models[best_key]
    logistic = models["logistica"]
    neural = models["red_neuronal"]
    f1_gap = abs(logistic["f1"] - neural["f1"])

    cols = st.columns(4)
    with cols[0]:
        ui.metric_card(
            "Mejor modelo por F1",
            MODEL_LABELS[best_key],
            f"F1 = {best['f1']:.3f}",
            color="#0B3D2E",
        )
    with cols[1]:
        ui.metric_card(
            "Brecha de F1",
            f"{f1_gap:.3f}",
            "Diferencia absoluta entre modelos",
            color="#1F8A70",
        )
    with cols[2]:
        ui.metric_card(
            "Registros de prueba",
            f"{int(split.get('test_rows', 0)):,}",
            "Mismo conjunto para ambos modelos",
            color="#1E6F9F",
        )
    with cols[3]:
        ui.metric_card(
            "Umbral de alto riesgo",
            f"{metadata.get('risk_bands', {}).get('classification_threshold', 0.5):.2f}",
            "Clasificación binaria de prueba",
            color="#E0A100",
        )

    st.info(
        f"Partición estratificada: {int(split.get('train_rows', 0)):,} registros de "
        f"entrenamiento y {int(split.get('test_rows', 0)):,} de prueba "
        f"(random_state={split.get('random_state', 42)}). El escalador se ajustó solo "
        "con el conjunto de entrenamiento para evitar fuga de información."
    )

    st.markdown("#### Métricas sobre el conjunto de prueba")
    comparison_rows = []
    for key, label in (("logistica", "Regresión Logística"), ("red_neuronal", "Red Neuronal")):
        item = models[key]
        comparison_rows.append(
            {
                "Modelo": label,
                "Accuracy": item["accuracy"],
                "Precision": item["precision"],
                "Recall": item["recall"],
                "F1-score": item["f1"],
                "ROC-AUC": item.get("roc_auc"),
                "Tiempo de entrenamiento (s)": item["training_time_seconds"],
                "Falsos positivos": item["confusion_matrix"][0][1],
                "Falsos negativos": item["confusion_matrix"][1][0],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1-score": st.column_config.NumberColumn(format="%.3f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Tiempo de entrenamiento (s)": st.column_config.NumberColumn(format="%.3f"),
            "Falsos positivos": st.column_config.NumberColumn(format="%d"),
            "Falsos negativos": st.column_config.NumberColumn(format="%d"),
        },
    )

    chart_col, matrix_col = st.columns([1.2, 1])
    with chart_col, st.container(border=True):
        st.pyplot(ui.metrics_comparison_chart(metadata), width="stretch")
    with matrix_col:
        matrix_left, matrix_right = st.columns(2)
        with matrix_left:
            st.pyplot(
                ui.confusion_matrix_heatmap(
                    logistic["confusion_matrix"],
                    "Logística · prueba",
                    color="Greens",
                ),
                width="stretch",
            )
        with matrix_right:
            st.pyplot(
                ui.confusion_matrix_heatmap(
                    neural["confusion_matrix"],
                    "Red neuronal · prueba",
                    color="Blues",
                ),
                width="stretch",
            )

    st.markdown("#### Interpretación de resultados")
    logistic_fp = logistic["confusion_matrix"][0][1]
    logistic_fn = logistic["confusion_matrix"][1][0]
    neural_fp = neural["confusion_matrix"][0][1]
    neural_fn = neural["confusion_matrix"][1][0]
    st.markdown(
        f"""
        - **Mejor F1-score:** **{MODEL_LABELS[best_key]}** con **{best['f1']:.3f}**.
          La diferencia entre ambos modelos es de **{f1_gap:.3f}** puntos.
        - **Regresión Logística:** accuracy {logistic['accuracy']:.3f}, precision
          {logistic['precision']:.3f}, recall {logistic['recall']:.3f}, ROC-AUC
          {logistic.get('roc_auc') or 0:.3f}. Cometió {logistic_fp} falsos positivos y
          {logistic_fn} falsos negativos en {int(logistic['n_samples'])} registros.
        - **Red Neuronal:** accuracy {neural['accuracy']:.3f}, precision
          {neural['precision']:.3f}, recall {neural['recall']:.3f}, ROC-AUC
          {neural.get('roc_auc') or 0:.3f}. Cometió {neural_fp} falsos positivos y
          {neural_fn} falsos negativos en {int(neural['n_samples'])} registros.
        - **Costo operativo:** la Regresión Logística entrenó en
          **{fmt_seconds(logistic['training_time_seconds'])}**, mientras que la Red
          Neuronal requirió **{fmt_seconds(neural['training_time_seconds'])}** en CPU.
        - **Lectura preventiva:** un falso negativo implica un canal de riesgo alto que
          no se priorizaría. Por eso, además del F1, conviene revisar el recall al
          definir el umbral operativo de alerta.
        """
    )

    with st.expander("Detalle técnico: arquitectura, historial y coeficientes"):
        tab1, tab2 = st.tabs(["Entrenamiento de la red", "Coeficientes logísticos"])
        with tab1:
            st.pyplot(ui.training_history_chart(metadata), width="stretch")
            st.caption(
                f"Arquitectura: {' → '.join(neural.get('architecture', []))}. "
                f"Épocas ejecutadas: {neural.get('epochs_run', 'N/A')}. "
                f"Mejor pérdida de validación: {neural.get('best_val_loss', 0):.4f}."
            )
        with tab2:
            st.pyplot(ui.logistic_coefficients_chart(metadata), width="stretch")
            st.caption(
                "Los coeficientes corresponden a variables estandarizadas. Un valor "
                "positivo aumenta la probabilidad estimada de riesgo alto; uno negativo "
                "la reduce."
            )
        st.caption(
            "La red neuronal usa una partición interna estratificada del conjunto de "
            "entrenamiento para early stopping. El conjunto de prueba final no participa "
            "en el ajuste de ninguno de los dos modelos."
        )

    st.caption(APP_DISCLAIMER)


def render_recommendations(artifacts: Artifacts, analysis: pd.DataFrame | None) -> None:
    ui.page_header(
        "Recomendaciones ambientales",
        "De la predicción a un plan de acción preventivo",
        (
            "Las recomendaciones se generan a partir del nivel de riesgo consensuado y "
            "de condiciones específicas como obstrucción, días sin limpieza, volumen de "
            "plásticos o frecuencia de lluvias."
        ),
    )
    policy_cols = st.columns(3)
    policies = [
        (
            "Bajo",
            (
                "Mantener el monitoreo periódico y conservar el calendario habitual de "
                "mantenimiento. Registrar observaciones para detectar cambios."
            ),
            "Monitoreo periódico",
        ),
        (
            "Medio",
            (
                "Programar una inspección y una limpieza preventiva en el corto plazo. "
                "Revisar puntos de acumulación y reforzar la recolección."
            ),
            "Inspección y limpieza preventiva",
        ),
        (
            "Alto",
            (
                "Priorizar la limpieza y el retiro de residuos antes de que puedan llegar "
                "al río. Verificar obstrucciones y coordinar una inspección de campo."
            ),
            "Limpieza prioritaria",
        ),
    ]
    for column, (level, description, action) in zip(policy_cols, policies):
        with column:
            ui.policy_card(level, description, action)
            st.markdown("")

    if analysis is None:
        dataset_unavailable("no hay datos analíticos para priorizar canales")
        return

    counts = risk_counts(analysis)
    total = len(analysis)
    left, right = st.columns([1, 1.35])
    with left, st.container(border=True):
        st.markdown("#### Priorización del dataset simulado")
        for level in RISK_LEVELS:
            ui.priority_bar(level, counts[level], total)
        st.caption(
            "El riesgo consensuado promedia las probabilidades de ambos modelos."
        )
    with right, st.container(border=True):
        st.markdown("#### Señales de atención en el dataset")
        signal_rows = [
            (
                "Obstrucción ≥ 60 %",
                int((analysis["obstruccion"] >= 60).sum()),
            ),
            (
                "Días sin limpieza ≥ 180",
                int((analysis["dias_sin_limpieza"] >= 180).sum()),
            ),
            (
                "Basura acumulada ≥ 50 %",
                int((analysis["basura_acumulada"] >= 50).sum()),
            ),
            (
                "Plásticos ≥ 2 000 kg",
                int((analysis["residuos_plasticos"] >= 2000).sum()),
            ),
            (
                "Lluvias ≥ 20 días/mes",
                int((analysis["frecuencia_lluvias"] >= 20).sum()),
            ),
        ]
        for signal, count in signal_rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:.43rem 0;border-bottom:1px solid #E7EFEB;font-size:.84rem;'>"
                f"<span>{signal}</span><strong style='color:#0B3D2E;'>{count:,}</strong></div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Estas señales son conteos sobre datos sintéticos; no son mediciones de campo."
        )

    st.markdown("#### Canales simulados con prioridad P1")
    high = (
        analysis[analysis["nivel_riesgo_consenso"] == "Alto"]
        .sort_values("prob_consenso", ascending=False)
        .head(12)
        .copy()
    )
    if high.empty:
        st.info("No hay canales clasificados como riesgo alto con los datos actuales.")
    else:
        high.insert(
            0,
            "Canal simulado",
            high["fila_original"].map(lambda value: f"SIM-{int(value):04d}"),
        )
        high["prob_consenso_mostrada"] = high["prob_consenso"] * 100
        plan = high[
            [
                "Canal simulado",
                "prob_consenso_mostrada",
                "prioridad",
                "basura_acumulada",
                "obstruccion",
                "flujo_agua",
                "dias_sin_limpieza",
                "residuos_plasticos",
                "recomendacion_principal",
            ]
        ].rename(
            columns={
                "prob_consenso_mostrada": "Prob. consenso (%)",
                "prioridad": "Prioridad",
                "basura_acumulada": "Basura (%)",
                "obstruccion": "Obstrucción (%)",
                "flujo_agua": "Flujo",
                "dias_sin_limpieza": "Días sin limpieza",
                "residuos_plasticos": "Plásticos (kg)",
                "recomendacion_principal": "Acción inmediata",
            }
        )
        st.dataframe(
            plan,
            hide_index=True,
            width="stretch",
            height=380,
            column_config={
                "Prob. consenso (%)": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "Acción inmediata": st.column_config.TextColumn(width="large"),
            },
        )
        plan_csv = high.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar plan de priorización (CSV)",
            data=plan_csv,
            file_name=f"ecocanal_plan_priorizacion_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("Reglas de recomendación utilizadas"):
        st.markdown(
            """
            | Condición detectada | Acción sugerida |
            |---|---|
            | Días sin limpieza ≥ 180 | Recuperar calendario de mantenimiento periódico |
            | Obstrucción ≥ 60 % | Revisión hidráulica y desobstrucción |
            | Basura acumulada ≥ 50 % | Reforzar recolección y segregación |
            | Plásticos ≥ 2 000 kg | Jornada de recuperación y valorización |
            | Flujo de agua ≤ 30 | Evaluar estancamiento y focos de contaminación |
            | Lluvias ≥ 20 días/mes | Inspección previa a eventos de lluvia |

            La primera recomendación siempre corresponde al nivel de riesgo
            (bajo, medio o alto). Si los modelos no coinciden, se añade una
            recomendación de validación en campo.
            """
        )


def render_documentation() -> None:
    ui.page_header(
        "Metodología y documentación",
        "Cómo funciona ECOCANAL IA",
        (
            "Resumen técnico del dataset, los modelos, la clasificación de riesgo y "
            "las limitaciones del prototipo académico."
        ),
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            ### Problemática
            Los residuos sólidos que se acumulan en canales urbanos pueden ser
            arrastrados por la lluvia hacia ríos y cuerpos de agua. La limpieza
            reactiva suele llegar después del evento de contaminación. Este prototipo
            explora cómo dos modelos de clasificación pueden apoyar la priorización
            preventiva a partir de condiciones operativas del canal.

            ### Objetivo
            Estimar la probabilidad de riesgo alto de acumulación y comparar el
            desempeño de una Regresión Logística y una Red Neuronal Artificial bajo
            un mismo protocolo de entrenamiento y evaluación.

            ### Flujo metodológico
            1. Generación reproducible del dataset sintético.
            2. Validación de columnas, rangos y valores faltantes.
            3. División estratificada 80/20 en entrenamiento y prueba.
            4. Estandarización ajustada solo con entrenamiento.
            5. Entrenamiento de ambos modelos.
            6. Evaluación con etiquetas reales del conjunto de prueba.
            7. Persistencia de modelos, escalador y metadatos.
            """
        )
    with right:
        st.markdown(
            f"""
            ### Clasificación de riesgo
            | Nivel | Probabilidad estimada |
            |---|---|
            | Bajo | < {RISK_LOW_MAX:.2f} |
            | Medio | {RISK_LOW_MAX:.2f} – {RISK_HIGH_MIN:.2f} |
            | Alto | ≥ {RISK_HIGH_MIN:.2f} |

            El umbral binario de la variable objetivo (`riesgo_alto`) es
            **{CLASSIFICATION_THRESHOLD:.2f}**, por lo que la banda alta coincide con
            la clase positiva evaluada en las métricas.

            ### Métricas
            - **Accuracy:** proporción de clasificaciones correctas.
            - **Precision:** de los canales marcados como alto riesgo, cuántos lo eran.
            - **Recall:** de los canales realmente de alto riesgo, cuántos se detectaron.
            - **F1-score:** media armónica de precision y recall.
            - **ROC-AUC:** capacidad de ordenar positivos por encima de negativos.
            - **Matriz de confusión:** verdaderos/falsos positivos y negativos.

            ### Limitaciones
            - Los datos son sintéticos y no representan mediciones de campo.
            - No se detecta basura mediante cámaras ni sensores.
            - Las variables son estimaciones operativas dentro de rangos definidos.
            - Un modelo predictivo no reemplaza inspecciones ni laboratorios.
            - Antes de uso real se requiere validación con datos ambientales locales.
            """
        )

    st.markdown("#### Información del proyecto")
    info_cols = st.columns(3)
    with info_cols[0]:
        ui.status_card(
            "Dataset",
            "1 800 registros sintéticos, 6 variables y una etiqueta binaria. Semilla 2026.",
            color="#1F8A70",
        )
    with info_cols[1]:
        ui.status_card(
            "Modelos",
            "Regresión Logística con estandarización y red neuronal densa 24-12-1 con dropout.",
            color="#1E6F9F",
        )
    with info_cols[2]:
        ui.status_card(
            "Trazabilidad",
            "Métricas, partición, tiempos y hashes se guardan en model_metadata.json.",
            color="#0B3D2E",
        )

    st.markdown("#### Ejecución local en Windows")
    st.code(
        """py -3.12 -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.generate_dataset
python -m src.train_models
streamlit run app.py""",
        language="powershell",
    )
    st.caption(
        "Si los modelos ya están incluidos en models/, puedes ejecutar directamente "
        "`streamlit run app.py`. Consulta el README.md para el detalle completo."
    )

    download_cols = st.columns(3)
    readme_path = BASE_DIR / "README.md"
    if readme_path.exists():
        with download_cols[0]:
            st.download_button(
                "Descargar README.md",
                data=readme_path.read_bytes(),
                file_name="README_ECOCANAL_IA.md",
                mime="text/markdown",
                width="stretch",
            )
    with download_cols[1]:
        if DATASET_PATH.exists():
            st.download_button(
                "Descargar dataset sintético",
                data=DATASET_PATH.read_bytes(),
                file_name="dataset_canales.csv",
                mime="text/csv",
                width="stretch",
            )
    with download_cols[2]:
        metadata_path = BASE_DIR / "models" / "model_metadata.json"
        if metadata_path.exists():
            st.download_button(
                "Descargar metadatos de modelos",
                data=metadata_path.read_bytes(),
                file_name="model_metadata.json",
                mime="application/json",
                width="stretch",
            )
    st.warning(APP_DISCLAIMER)


def main() -> None:
    ui.inject_css()

    artifacts: Artifacts | None
    artifact_error: str | None = None
    try:
        artifacts = get_artifacts()
    except Exception as exc:  # noqa: BLE001  # artefactos ausentes o corruptos
        artifacts = None
        artifact_error = str(exc)

    dataset: pd.DataFrame | None = None
    dataset_error: str | None = None
    try:
        dataset = get_dataset()
    except Exception as exc:  # noqa: BLE001
        dataset_error = str(exc)

    analysis: pd.DataFrame | None = None
    if artifacts is not None and dataset is not None:
        try:
            analysis = get_analysis(artifacts, dataset_version())
        except Exception as exc:  # noqa: BLE001
            dataset_error = str(exc)

    page = sidebar(artifacts, len(dataset) if dataset is not None else None, dataset_error)

    flash_success = st.session_state.pop("flash_success", None)
    if flash_success:
        st.success(flash_success)

    if artifacts is None:
        render_artifact_error(artifact_error or "Artefactos no disponibles.")
        return

    if page == "Panel general":
        render_dashboard(artifacts, dataset, analysis)
    elif page == "Predicción individual":
        render_individual(artifacts)
    elif page == "Predicción por lotes":
        render_batch()
    elif page == "Comparación de modelos":
        render_model_comparison(artifacts)
    elif page == "Recomendaciones ambientales":
        render_recommendations(artifacts, analysis)
    elif page == "Metodología y documentación":
        render_documentation()

    ui.footer()


if __name__ == "__main__":
    main()
