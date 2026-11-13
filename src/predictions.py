"""Carga de artefactos y predicciones individuales y por lotes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    CLASSIFICATION_THRESHOLD,
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    LOGISTIC_MODEL_PATH,
    METADATA_PATH,
    MODELS_DIR,
    NEURAL_MODEL_PATH,
    PREPROCESSOR_PATH,
    classify_risk,
)
from .data_processing import DataValidationError
from .recommendations import generate_recommendations, priority_label


class ModelArtifactError(RuntimeError):
    """Error al cargar los modelos persistidos."""


@dataclass
class Artifacts:
    """Contenedor de modelos, preprocesador y metadatos cargados."""

    logistic_model: Any
    neural_model: Any
    preprocessor: dict[str, Any]
    metadata: dict[str, Any]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    @property
    def scaler(self) -> Any:
        return self.preprocessor["scaler"]

    @property
    def medians(self) -> dict[str, float]:
        return self.preprocessor.get("medians", {})


def load_metadata(models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    path = Path(models_dir) / METADATA_PATH.name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_artifacts(models_dir: Path = MODELS_DIR) -> Artifacts:
    """Carga modelos y preprocesador desde disco sin reentrenar."""
    models_dir = Path(models_dir)
    paths = {
        "Regresión Logística": models_dir / LOGISTIC_MODEL_PATH.name,
        "Red Neuronal": models_dir / NEURAL_MODEL_PATH.name,
        "Preprocesador": models_dir / PREPROCESSOR_PATH.name,
    }
    missing = [f"{label}: {path.name}" for label, path in paths.items() if not path.exists()]
    if missing:
        raise ModelArtifactError(
            "No se encontraron todos los artefactos del modelo. "
            "Ejecuta `python -m src.train_models`. Faltan: " + ", ".join(missing)
        )

    try:
        logistic_model = joblib.load(paths["Regresión Logística"])
        preprocessor = joblib.load(paths["Preprocesador"])
        from tensorflow import keras

        neural_model = keras.models.load_model(
            paths["Red Neuronal"], compile=False
        )
    except Exception as exc:  # pragma: no cover - depende del entorno
        raise ModelArtifactError(
            f"No fue posible cargar los modelos guardados: {exc}"
        ) from exc

    if not isinstance(preprocessor, dict) or "scaler" not in preprocessor:
        raise ModelArtifactError(
            "El preprocesador guardado no tiene el formato esperado."
        )
    feature_columns = tuple(
        preprocessor.get("feature_columns", FEATURE_COLUMNS)
    )
    if feature_columns != FEATURE_COLUMNS:
        raise ModelArtifactError(
            "Las variables del preprocesador no coinciden con la configuración."
        )

    return Artifacts(
        logistic_model=logistic_model,
        neural_model=neural_model,
        preprocessor=preprocessor,
        metadata=load_metadata(models_dir),
        feature_columns=feature_columns,
    )


def validate_single_features(values: Mapping[str, Any]) -> dict[str, float]:
    """Valida un registro individual y devuelve valores numéricos."""
    missing = [column for column in FEATURE_COLUMNS if column not in values]
    if missing:
        raise DataValidationError(
            "Faltan campos obligatorios: " + ", ".join(missing)
        )
    validated: dict[str, float] = {}
    errors: list[str] = []
    for column in FEATURE_COLUMNS:
        try:
            value = float(values[column])
        except (TypeError, ValueError):
            errors.append(f"{column} no es numérico")
            continue
        if not math.isfinite(value):
            errors.append(f"{column} no es un número finito")
            continue
        low, high = FEATURE_RANGES[column]
        if not low <= value <= high:
            errors.append(f"{column} debe estar entre {low:g} y {high:g}")
            continue
        validated[column] = value
    if errors:
        raise DataValidationError("; ".join(errors))
    return validated


def _probability_columns(
    frame: pd.DataFrame, artifacts: Artifacts
) -> tuple[np.ndarray, np.ndarray]:
    features = frame[list(artifacts.feature_columns)].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise DataValidationError(
            "Las variables de entrada contienen valores faltantes o infinitos."
        )
    scaled = artifacts.scaler.transform(features)
    logistic_probability = artifacts.logistic_model.predict_proba(scaled)[:, 1]
    neural_probability = np.asarray(
        artifacts.neural_model.predict(scaled, verbose=0)
    ).ravel()
    return logistic_probability, neural_probability


def _model_result(probability: float) -> dict[str, Any]:
    probability = float(probability)
    return {
        "probabilidad": probability,
        "prediccion": int(probability >= CLASSIFICATION_THRESHOLD),
        "nivel_riesgo": classify_risk(probability),
        "etiqueta_prediccion": (
            "Riesgo alto"
            if probability >= CLASSIFICATION_THRESHOLD
            else "Sin riesgo alto"
        ),
    }


def predict_single(
    values: Mapping[str, Any], artifacts: Artifacts
) -> dict[str, Any]:
    """Ejecuta ambos modelos para un canal y genera recomendaciones."""
    features = validate_single_features(values)
    frame = pd.DataFrame([features], columns=list(FEATURE_COLUMNS))
    logistic_probability, neural_probability = _probability_columns(frame, artifacts)

    logistic_result = _model_result(float(logistic_probability[0]))
    neural_result = _model_result(float(neural_probability[0]))
    consensus_probability = (
        logistic_result["probabilidad"] + neural_result["probabilidad"]
    ) / 2.0
    agree = logistic_result["prediccion"] == neural_result["prediccion"]
    consensus_risk = classify_risk(consensus_probability)
    recommendations = generate_recommendations(
        features,
        consensus_risk,
        models_agree=agree,
    )

    return {
        "entrada": features,
        "logistica": logistic_result,
        "red_neuronal": neural_result,
        "consenso": {
            "probabilidad": float(consensus_probability),
            "prediccion": int(consensus_probability >= CLASSIFICATION_THRESHOLD),
            "nivel_riesgo": consensus_risk,
            "modelos_coinciden": bool(agree),
            "prioridad": priority_label(consensus_risk),
        },
        "recomendaciones": recommendations,
        "recomendacion_principal": recommendations[0],
    }


def predict_batch(
    frame: pd.DataFrame,
    artifacts: Artifacts,
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Añade predicciones y recomendaciones a un lote ya validado.

    Se espera que las variables obligatorias estén presentes, sin faltantes y
    dentro de rango. La app construye ese lote con
    :func:`src.data_processing.prepare_batch`.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(
            "Faltan columnas obligatorias para predecir: " + ", ".join(missing)
        )
    if frame.empty:
        raise DataValidationError("El lote no contiene filas válidas para predecir.")

    working = frame.reset_index(drop=False).rename(columns={"index": "_indice_original"})
    original_index = working.pop("_indice_original")
    if pd.api.types.is_integer_dtype(original_index):
        original_rows = original_index.astype(int) + 2
    else:
        original_rows = pd.Series(
            [str(value) for value in original_index], index=working.index
        )

    logistic_probability, neural_probability = _probability_columns(
        working, artifacts
    )
    consensus_probability = (logistic_probability + neural_probability) / 2.0
    logistic_prediction = (logistic_probability >= CLASSIFICATION_THRESHOLD).astype(int)
    neural_prediction = (neural_probability >= CLASSIFICATION_THRESHOLD).astype(int)
    consensus_prediction = (
        consensus_probability >= CLASSIFICATION_THRESHOLD
    ).astype(int)
    agree = pd.Series(
        logistic_prediction == neural_prediction, index=working.index
    )

    result = working.copy()
    result.insert(0, "fila_original", original_rows.to_numpy())
    result["prob_regresion_logistica"] = np.round(logistic_probability, 6)
    result["prob_red_neuronal"] = np.round(neural_probability, 6)
    result["prob_consenso"] = np.round(consensus_probability, 6)
    result["prediccion_logistica"] = logistic_prediction
    result["prediccion_red_neuronal"] = neural_prediction
    result["prediccion_consenso"] = consensus_prediction
    result["nivel_riesgo_logistica"] = [
        classify_risk(value) for value in logistic_probability
    ]
    result["nivel_riesgo_red_neuronal"] = [
        classify_risk(value) for value in neural_probability
    ]
    result["nivel_riesgo_consenso"] = [
        classify_risk(value) for value in consensus_probability
    ]
    result["modelos_coinciden"] = agree

    recommendations: list[str] = []
    detailed: list[str] = []
    priorities: list[str] = []
    for position, (_, row) in enumerate(result.iterrows()):
        risk_level = result.iloc[position]["nivel_riesgo_consenso"]
        features = {column: float(row[column]) for column in FEATURE_COLUMNS}
        items = generate_recommendations(
            features,
            risk_level,
            models_agree=bool(agree.iloc[position]),
        )
        recommendations.append(items[0])
        detailed.append(" | ".join(items))
        priorities.append(priority_label(risk_level))
    result["prioridad"] = priorities
    result["recomendacion_principal"] = recommendations
    result["recomendaciones_detalladas"] = detailed

    if target_column and target_column in result.columns:
        result[target_column] = pd.to_numeric(
            result[target_column], errors="coerce"
        )
    return result
