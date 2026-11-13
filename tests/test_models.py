"""Pruebas de modelos persistidos, predicciones y evaluación."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LOGISTIC_MODEL_PATH,
    NEURAL_MODEL_PATH,
    PREPROCESSOR_PATH,
    RISK_LEVELS,
    TARGET_COLUMN,
)
from src.data_processing import DataValidationError, prepare_batch
from src.evaluation import compute_metrics, evaluate_batch_results
from src.predictions import (
    load_artifacts,
    predict_batch,
    predict_single,
    validate_single_features,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="session")
def artifacts():
    return load_artifacts()


@pytest.fixture(scope="session")
def dataset():
    return pd.read_csv(DATASET_PATH)


def test_saved_artifacts_exist() -> None:
    assert LOGISTIC_MODEL_PATH.exists()
    assert NEURAL_MODEL_PATH.exists()
    assert PREPROCESSOR_PATH.exists()


def test_artifacts_load_with_expected_features(artifacts) -> None:
    assert artifacts.feature_columns == FEATURE_COLUMNS
    assert artifacts.scaler.n_features_in_ == len(FEATURE_COLUMNS)
    assert artifacts.metadata["dataset"]["rows"] >= 1000
    assert artifacts.metadata["split"]["test_rows"] > 0


def test_metadata_is_consistent(artifacts) -> None:
    metadata = artifacts.metadata
    split = metadata["split"]
    models = metadata["models"]

    assert split["train_rows"] + split["test_rows"] == metadata["dataset"]["rows"]
    assert set(models) == {"logistica", "red_neuronal"}
    assert metadata["best_model_by_f1"] in models

    for model in models.values():
        matrix = model["confusion_matrix"]
        assert np.asarray(matrix).shape == (2, 2)
        assert sum(map(sum, matrix)) == model["n_samples"] == split["test_rows"]
        accuracy_from_matrix = (matrix[0][0] + matrix[1][1]) / model["n_samples"]
        assert accuracy_from_matrix == pytest.approx(model["accuracy"], abs=1e-9)
        for metric in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= model[metric] <= 1.0
        assert model["training_time_seconds"] >= 0


def test_preprocessor_transforms_training_ranges(artifacts) -> None:
    sample = pd.read_csv(DATASET_PATH)[list(FEATURE_COLUMNS)].head(20).to_numpy()
    transformed = artifacts.scaler.transform(sample)
    assert transformed.shape == (20, len(FEATURE_COLUMNS))
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed.mean(axis=0), 0, atol=1.5)


def test_single_predictions_follow_expected_gradient(artifacts) -> None:
    low = {
        "basura_acumulada": 8,
        "obstruccion": 5,
        "flujo_agua": 92,
        "frecuencia_lluvias": 5,
        "dias_sin_limpieza": 10,
        "residuos_plasticos": 50,
    }
    high = {
        "basura_acumulada": 92,
        "obstruccion": 94,
        "flujo_agua": 10,
        "frecuencia_lluvias": 28,
        "dias_sin_limpieza": 500,
        "residuos_plasticos": 4200,
    }
    low_result = predict_single(low, artifacts)
    high_result = predict_single(high, artifacts)

    assert low_result["consenso"]["probabilidad"] < high_result["consenso"]["probabilidad"]
    assert low_result["consenso"]["nivel_riesgo"] == "Bajo"
    assert high_result["consenso"]["nivel_riesgo"] == "Alto"
    assert low_result["logistica"]["prediccion"] == 0
    assert high_result["red_neuronal"]["prediccion"] == 1
    for key in ("logistica", "red_neuronal"):
        assert 0 <= low_result[key]["probabilidad"] <= 1
        assert 0 <= high_result[key]["probabilidad"] <= 1
        assert high_result[key]["nivel_riesgo"] in RISK_LEVELS
    assert low_result["recomendaciones"]
    assert high_result["recomendaciones"]
    assert low_result["recomendacion_principal"] != high_result["recomendacion_principal"]


@pytest.mark.parametrize(
    "values,message",
    [
        ({"basura_acumulada": 10}, "Faltan campos"),
        (
            {
                "basura_acumulada": 101,
                "obstruccion": 30,
                "flujo_agua": 30,
                "frecuencia_lluvias": 10,
                "dias_sin_limpieza": 10,
                "residuos_plasticos": 100,
            },
            "fuera de rango|debe estar entre",
        ),
    ],
)
def test_single_input_validation(values, message) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_single_features(values)


def test_batch_predictions_without_target(artifacts) -> None:
    raw = pd.read_csv(EXAMPLES_DIR / "ejemplo_lote_canales.csv")
    valid, report = prepare_batch(raw, artifacts.medians)
    result = predict_batch(valid, artifacts)

    assert report.valid_rows == len(raw)
    assert len(result) == len(raw)
    expected_columns = {
        "fila_original",
        "prob_regresion_logistica",
        "prob_red_neuronal",
        "prob_consenso",
        "nivel_riesgo_consenso",
        "recomendacion_principal",
        "recomendaciones_detalladas",
    }
    assert expected_columns.issubset(result.columns)
    assert result["prob_consenso"].between(0, 1).all()
    assert result["nivel_riesgo_consenso"].isin(RISK_LEVELS).all()
    assert result["recomendacion_principal"].str.len().gt(0).all()
    assert evaluate_batch_results(result, TARGET_COLUMN) is None


def test_batch_predictions_with_target_and_metrics(artifacts) -> None:
    raw = pd.read_csv(EXAMPLES_DIR / "ejemplo_lote_etiquetado.csv")
    valid, report = prepare_batch(raw, artifacts.medians)
    result = predict_batch(valid, artifacts, target_column=TARGET_COLUMN)
    metrics = evaluate_batch_results(result, TARGET_COLUMN)

    assert report.target_available
    assert metrics is not None
    assert set(metrics) == {"logistica", "red_neuronal", "consenso"}
    for item in metrics.values():
        assert item["n_samples"] == len(result)
        assert sum(map(sum, item["confusion_matrix"])) == len(result)
        assert 0 <= item["accuracy"] <= 1


def test_batch_metrics_use_real_labels_not_predictions() -> None:
    # Si la función usara las predicciones como etiquetas, accuracy sería 1.
    frame = pd.DataFrame(
        {
            "riesgo_alto": [1, 1, 1, 0],
            "prediccion_logistica": [0, 0, 0, 1],
            "prob_regresion_logistica": [0.1, 0.2, 0.3, 0.9],
            "prediccion_red_neuronal": [0, 0, 0, 1],
            "prob_red_neuronal": [0.2, 0.1, 0.2, 0.8],
            "prediccion_consenso": [0, 0, 0, 1],
            "prob_consenso": [0.15, 0.15, 0.25, 0.85],
        }
    )
    metrics = evaluate_batch_results(frame, "riesgo_alto")
    assert metrics is not None
    assert metrics["logistica"]["accuracy"] == 0.0
    assert metrics["logistica"]["recall"] == 0.0


def test_compute_metrics_confusion_matrix() -> None:
    metrics = compute_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.6, 0.4, 0.9])
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
