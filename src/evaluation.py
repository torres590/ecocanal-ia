"""Métricas de evaluación y utilidades de serialización."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def to_jsonable(value: Any) -> Any:
    """Convierte tipos de NumPy a tipos serializables por JSON."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def confusion_matrix_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """Matriz de confusión con etiquetas fijas [0, 1]."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return matrix.astype(int).tolist()


def compute_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Calcula métricas binarias usando etiquetas reales externas.

    ``y_true`` debe provenir del dataset o de un archivo etiquetado. Nunca se
    deben pasar predicciones como etiquetas reales.
    """
    y_true_array = np.asarray(y_true, dtype=int).ravel()
    y_pred_array = np.asarray(y_pred, dtype=int).ravel()
    if y_true_array.size != y_pred_array.size:
        raise ValueError("y_true y y_pred deben tener la misma longitud.")
    if y_true_array.size == 0:
        raise ValueError("No hay observaciones para evaluar.")

    labels = sorted(set(y_true_array.tolist()) | set(y_pred_array.tolist()))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(y_true_array, y_pred_array, zero_division=0)
        ),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "n_samples": int(y_true_array.size),
        "support_positive": int(y_true_array.sum()),
        "confusion_matrix": confusion_matrix_2x2(y_true_array, y_pred_array),
        "classification_report": to_jsonable(
            classification_report(
                y_true_array,
                y_pred_array,
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            )
        ),
    }
    if y_prob is not None and len(set(labels)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def metrics_frame(metrics_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convierte métricas de varios modelos en filas para tablas."""
    rows: list[dict[str, Any]] = []
    for model_key, metrics in metrics_by_model.items():
        rows.append(
            {
                "modelo": model_key,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
            }
        )
    return rows


def evaluate_batch_results(
    result: pd.DataFrame,
    target_column: str,
) -> dict[str, dict[str, Any]] | None:
    """Evalúa un lote etiquetado usando únicamente etiquetas reales.

    Devuelve ``None`` cuando el archivo no incluye la columna objetivo o cuando
    no quedan etiquetas válidas. Las predicciones nunca se reutilizan como
    etiquetas reales.
    """
    if target_column not in result.columns:
        return None
    target = pd.to_numeric(result[target_column], errors="coerce")
    valid_mask = target.isin([0, 1])
    if not bool(valid_mask.any()):
        return None

    model_specs = {
        "logistica": ("prediccion_logistica", "prob_regresion_logistica"),
        "red_neuronal": ("prediccion_red_neuronal", "prob_red_neuronal"),
        "consenso": ("prediccion_consenso", "prob_consenso"),
    }
    metrics: dict[str, dict[str, Any]] = {}
    for key, (prediction_column, probability_column) in model_specs.items():
        if prediction_column not in result.columns:
            continue
        mask = valid_mask & result[prediction_column].notna()
        if not bool(mask.any()):
            continue
        metrics[key] = compute_metrics(
            target.loc[mask].to_numpy(dtype=int),
            result.loc[mask, prediction_column].to_numpy(dtype=int),
            result.loc[mask, probability_column].to_numpy(dtype=float),
        )
    return metrics or None
