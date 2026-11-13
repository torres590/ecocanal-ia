"""Entrenamiento, evaluación y persistencia de los modelos de ECOCANAL IA.

El mismo conjunto de prueba y el mismo preprocesamiento se usan para la
Regresión Logística y la Red Neuronal. El escalador se ajusta únicamente con el
conjunto de entrenamiento para evitar fuga de información.

Uso:
    python -m src.train_models
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    APP_NAME,
    BASE_DIR,
    CLASSIFICATION_THRESHOLD,
    DATASET_PATH,
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    LOGISTIC_MODEL_PATH,
    METADATA_PATH,
    MODEL_LABELS,
    MODELS_DIR,
    NEURAL_MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    RISK_HIGH_MIN,
    RISK_LOW_MAX,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data_processing import get_training_medians, load_dataset
from .evaluation import compute_metrics, to_jsonable


@dataclass
class TrainingOutput:
    metadata: dict[str, Any]
    logistic_path: Path
    neural_path: Path
    preprocessor_path: Path
    metadata_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_tensorflow_seed(seed: int) -> Any:
    """Configura semillas y devuelve el módulo TensorFlow."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    with contextlib.suppress(Exception):
        tf.config.experimental.enable_op_determinism()
    return tf


def build_neural_network(input_dim: int, seed: int = RANDOM_STATE) -> Any:
    """Red neuronal densa binaria, pequeña para evitar sobreajuste."""
    _set_tensorflow_seed(seed)
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,), name="condiciones_canal"),
            layers.Dense(
                24,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
                name="capa_oculta_1",
            ),
            layers.Dropout(0.15, name="dropout_1"),
            layers.Dense(
                12,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
                name="capa_oculta_2",
            ),
            layers.Dense(1, activation="sigmoid", name="probabilidad_riesgo_alto"),
        ],
        name="ecocanal_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _train_logistic(
    x_train: np.ndarray, y_train: np.ndarray, seed: int
) -> tuple[LogisticRegression, float]:
    started = time.perf_counter()
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
        class_weight=None,
    )
    model.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    return model, elapsed


def _train_neural_network(
    x_train: np.ndarray, y_train: np.ndarray, seed: int
) -> tuple[Any, float, dict[str, Any]]:
    from tensorflow import keras

    # Partición interna estratificada para early stopping. El conjunto de
    # prueba final no participa en ninguna decisión de entrenamiento.
    x_sub, x_val, y_sub, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=seed,
        stratify=y_train,
    )
    model = build_neural_network(x_train.shape[1], seed=seed)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=12,
            min_delta=1e-4,
        ),
    ]
    started = time.perf_counter()
    history = model.fit(
        x_sub,
        y_sub,
        validation_data=(x_val, y_val),
        epochs=250,
        batch_size=32,
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )
    elapsed = time.perf_counter() - started
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    details = {
        "epochs_requested": 250,
        "epochs_run": len(history_data.get("loss", [])),
        "internal_train_rows": len(x_sub),
        "internal_validation_rows": len(x_val),
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss": "binary_crossentropy",
        "architecture": [
            "Dense(24, ReLU, L2=1e-4)",
            "Dropout(0.15)",
            "Dense(12, ReLU, L2=1e-4)",
            "Dense(1, Sigmoid)",
        ],
        "final_train_loss": history_data["loss"][-1],
        "best_val_loss": float(min(history_data["val_loss"])),
        "history": history_data,
    }
    return model, elapsed, details


def train_and_save(
    dataset_path: Path = DATASET_PATH,
    models_dir: Path = MODELS_DIR,
    *,
    seed: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    verbose: bool = True,
) -> TrainingOutput:
    """Entrena ambos modelos, los evalúa y guarda todos los artefactos."""
    models_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_path)

    if verbose:
        print(f"Cargando dataset: {dataset_path} ({len(dataset)} filas)")
        print(f"Prevalencia de riesgo alto: {dataset[TARGET_COLUMN].mean():.2%}")

    x = dataset[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = dataset[TARGET_COLUMN].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    if verbose:
        print(
            f"Entrenamiento: {len(x_train)} | Prueba: {len(x_test)} "
            f"(estratificado, test_size={test_size})"
        )

    preprocessor_fit = StandardScaler()
    x_train_scaled = preprocessor_fit.fit_transform(x_train)
    x_test_scaled = preprocessor_fit.transform(x_test)

    train_frame = pd.DataFrame(x_train, columns=list(FEATURE_COLUMNS))
    medians = get_training_medians(train_frame)

    if verbose:
        print("Entrenando Regresión Logística...")
    logistic_model, logistic_time = _train_logistic(x_train_scaled, y_train, seed)
    logistic_prob = logistic_model.predict_proba(x_test_scaled)[:, 1]
    logistic_pred = (logistic_prob >= CLASSIFICATION_THRESHOLD).astype(int)
    logistic_metrics = compute_metrics(y_test, logistic_pred, logistic_prob)
    logistic_metrics.update(
        {
            "training_time_seconds": logistic_time,
            "prediction_threshold": CLASSIFICATION_THRESHOLD,
            "hyperparameters": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "class_weight": None,
            },
        }
    )

    if verbose:
        print("Entrenando Red Neuronal Artificial...")
    neural_model, neural_time, neural_details = _train_neural_network(
        x_train_scaled, y_train, seed
    )
    neural_prob = neural_model.predict(x_test_scaled, verbose=0).ravel()
    neural_pred = (neural_prob >= CLASSIFICATION_THRESHOLD).astype(int)
    neural_metrics = compute_metrics(y_test, neural_pred, neural_prob)
    neural_metrics.update(
        {
            "training_time_seconds": neural_time,
            "prediction_threshold": CLASSIFICATION_THRESHOLD,
            **neural_details,
        }
    )

    if verbose:
        print(
            "Regresión Logística -> "
            f"accuracy={logistic_metrics['accuracy']:.4f} | "
            f"f1={logistic_metrics['f1']:.4f}"
        )
        print(
            "Red Neuronal       -> "
            f"accuracy={neural_metrics['accuracy']:.4f} | "
            f"f1={neural_metrics['f1']:.4f}"
        )

    preprocessor = {
        "scaler": preprocessor_fit,
        "medians": medians,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_ranges": {key: list(value) for key, value in FEATURE_RANGES.items()},
        "version": 1,
        "fitted_on": "train_split_only",
    }

    logistic_path = models_dir / LOGISTIC_MODEL_PATH.name
    neural_path = models_dir / NEURAL_MODEL_PATH.name
    preprocessor_path = models_dir / PREPROCESSOR_PATH.name
    metadata_path = models_dir / METADATA_PATH.name

    joblib.dump(logistic_model, logistic_path)
    joblib.dump(preprocessor, preprocessor_path)
    neural_model.save(neural_path)

    coefficients = {
        feature: {
            "coeficiente": float(coef),
            "odds_ratio": float(np.exp(coef)),
        }
        for feature, coef in zip(FEATURE_COLUMNS, logistic_model.coef_[0])
    }
    metrics_by_key = {
        "logistica": logistic_metrics,
        "red_neuronal": neural_metrics,
    }
    best_model = max(metrics_by_key, key=lambda key: metrics_by_key[key]["f1"])

    metadata: dict[str, Any] = {
        "project": APP_NAME,
        "version": "1.0.0",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "synthetic_data": True,
        "dataset": {
            "file": str(dataset_path.relative_to(BASE_DIR))
            if dataset_path.is_relative_to(BASE_DIR)
            else str(dataset_path),
            "rows": len(dataset),
            "features": list(FEATURE_COLUMNS),
            "target": TARGET_COLUMN,
            "prevalence": float(dataset[TARGET_COLUMN].mean()),
            "sha256": _sha256(dataset_path),
        },
        "split": {
            "test_size": test_size,
            "random_state": seed,
            "stratified": True,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "train_prevalence": float(y_train.mean()),
            "test_prevalence": float(y_test.mean()),
        },
        "preprocessing": {
            "scaler": "StandardScaler",
            "fit_on": "train_split_only",
            "imputation": "medianas del conjunto de entrenamiento",
            "feature_columns": list(FEATURE_COLUMNS),
            "medians": {key: float(value) for key, value in medians.items()},
        },
        "risk_bands": {
            "bajo": f"probabilidad < {RISK_LOW_MAX:.2f}",
            "medio": f"{RISK_LOW_MAX:.2f} <= probabilidad < {RISK_HIGH_MIN:.2f}",
            "alto": f"probabilidad >= {RISK_HIGH_MIN:.2f}",
            "classification_threshold": CLASSIFICATION_THRESHOLD,
        },
        "models": {
            "logistica": {
                "label": MODEL_LABELS["logistica"],
                **logistic_metrics,
                "coefficients": coefficients,
            },
            "red_neuronal": {
                "label": MODEL_LABELS["red_neuronal"],
                **neural_metrics,
            },
        },
        "best_model_by_f1": best_model,
        "artifacts": {
            "logistic_model": logistic_path.name,
            "neural_network": neural_path.name,
            "preprocessor": preprocessor_path.name,
        },
    }

    metadata_path.write_text(
        json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if verbose:
        print(f"Modelos guardados en: {models_dir}")
        print(f"Metadatos: {metadata_path}")
        print(f"Mejor modelo por F1: {MODEL_LABELS[best_model]}")

    return TrainingOutput(
        metadata=metadata,
        logistic_path=logistic_path,
        neural_path=neural_path,
        preprocessor_path=preprocessor_path,
        metadata_path=metadata_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos de ECOCANAL IA")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()
    train_and_save(
        dataset_path=args.dataset,
        models_dir=args.models_dir,
        seed=args.seed,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
