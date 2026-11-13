"""Pruebas de validación, lectura y preparación de datos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import DATASET_PATH, FEATURE_COLUMNS, FEATURE_RANGES, TARGET_COLUMN
from src.data_processing import (
    DataValidationError,
    make_template_dataframe,
    normalize_columns,
    prepare_batch,
    read_tabular_file,
    validate_required_columns,
)

MEDIANS = {
    "basura_acumulada": 34.0,
    "obstruccion": 33.0,
    "flujo_agua": 75.0,
    "frecuencia_lluvias": 14.0,
    "dias_sin_limpieza": 115.0,
    "residuos_plasticos": 840.0,
}


def test_normalize_columns_handles_accents_and_spaces() -> None:
    frame = pd.DataFrame(
        {
            "Basura acumulada": [10],
            "Obstrucción": [20],
            "FLUJO_AGUA": [80],
            "Frecuencia  lluvias": [10],
            "dias sin limpieza": [30],
            "residuos-plasticos": [100],
        }
    )
    normalized = normalize_columns(frame)
    for column in FEATURE_COLUMNS:
        assert column in normalized.columns


def test_validate_required_columns_reports_missing() -> None:
    with pytest.raises(DataValidationError, match="Faltan columnas"):
        validate_required_columns(pd.DataFrame({"basura_acumulada": [1]}))


def test_validate_required_columns_includes_target_when_requested() -> None:
    frame = pd.DataFrame({column: [1.0] for column in FEATURE_COLUMNS})
    validate_required_columns(frame)
    with pytest.raises(DataValidationError, match=TARGET_COLUMN):
        validate_required_columns(frame, require_target=True)


def test_prepare_batch_imputes_limits_and_excludes_invalid_rows() -> None:
    raw = pd.DataFrame(
        {
            "basura_acumulada": [None, "texto", 150, 20],
            "obstruccion": [60, 50, -5, 30],
            "flujo_agua": [30, 40, 110, 70],
            "frecuencia_lluvias": [20, 10, 5, 12],
            "dias_sin_limpieza": [200, 100, 30, 50],
            "residuos_plasticos": [1500, 800, 100, 400],
            "comentario": ["a", "b", "c", "d"],
        }
    )
    valid, report = prepare_batch(raw, MEDIANS)

    assert report.total_rows == 4
    assert report.valid_rows == 3
    assert report.invalid_rows == 1
    assert report.imputed_rows == 1
    assert report.limited_rows == 1
    assert report.missing_values["basura_acumulada"] == 1
    assert report.out_of_range_values["basura_acumulada"] == 1
    assert report.out_of_range_values["obstruccion"] == 1
    assert valid["datos_imputados"].sum() == 1
    assert valid["valores_limitados"].sum() == 1
    assert valid["basura_acumulada"].max() == 100
    assert valid["obstruccion"].min() == 0
    assert "comentario" in valid.columns


def test_prepare_batch_raises_when_no_valid_rows() -> None:
    raw = pd.DataFrame(
        {
            "basura_acumulada": [None],
            "obstruccion": [None],
            "flujo_agua": [None],
            "frecuencia_lluvias": [None],
            "dias_sin_limpieza": [None],
            "residuos_plasticos": [None],
        }
    )
    with pytest.raises(DataValidationError, match="Ninguna fila"):
        prepare_batch(raw, MEDIANS)


def test_read_csv_and_excel(tmp_path: Path) -> None:
    frame = make_template_dataframe()
    csv_path = tmp_path / "lote.csv"
    frame.to_csv(csv_path, index=False)
    loaded_csv = read_tabular_file(csv_path)
    assert list(loaded_csv.columns) == list(FEATURE_COLUMNS)
    assert len(loaded_csv) == 3

    excel_path = tmp_path / "lote.xlsx"
    frame.to_excel(excel_path, index=False)
    loaded_excel = read_tabular_file(excel_path)
    assert list(loaded_excel.columns) == list(FEATURE_COLUMNS)
    assert len(loaded_excel) == 3


def test_read_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "archivo.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Formato no soportado"):
        read_tabular_file(path)


def test_dataset_is_valid_and_within_ranges() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    assert list(dataset.columns) == list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    assert len(dataset) >= 1000
    assert not dataset.isna().any().any()
    assert set(dataset[TARGET_COLUMN].unique()) == {0, 1}
    assert 0.25 <= dataset[TARGET_COLUMN].mean() <= 0.65
    for column, (low, high) in FEATURE_RANGES.items():
        assert dataset[column].between(low, high).all()


def test_template_has_required_columns() -> None:
    template = make_template_dataframe()
    assert list(template.columns) == list(FEATURE_COLUMNS)
    assert not template.isna().any().any()
