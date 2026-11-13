"""Carga, validación y preparación de datos para ECOCANAL IA.

El módulo es independiente de Streamlit para que pueda reutilizarse desde
scripts, notebooks y pruebas automatizadas.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, FEATURE_RANGES, TARGET_COLUMN


class DataValidationError(ValueError):
    """Error controlado para datos inválidos o incompletos."""


@dataclass
class BatchReport:
    """Resumen legible del proceso de validación de un lote."""

    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    imputed_rows: int = 0
    limited_rows: int = 0
    missing_columns: list[str] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    out_of_range_values: dict[str, int] = field(default_factory=dict)
    invalid_detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_available: bool = False

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "imputed_rows": self.imputed_rows,
            "limited_rows": self.limited_rows,
            "missing_columns": self.missing_columns,
            "ignored_columns": self.ignored_columns,
            "missing_values": self.missing_values,
            "out_of_range_values": self.out_of_range_values,
            "target_available": self.target_available,
        }


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_column_name(column: object) -> str:
    """Normaliza encabezados: minúsculas, sin acentos y con guiones bajos."""
    text = _strip_accents(str(column).strip().lower())
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con encabezados normalizados y sin duplicados."""
    result = df.copy()
    normalized = [normalize_column_name(col) for col in result.columns]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise DataValidationError(
            "Hay columnas duplicadas después de normalizar los encabezados: "
            + ", ".join(duplicates)
        )
    result.columns = normalized
    return result


def validate_required_columns(df: pd.DataFrame, require_target: bool = False) -> None:
    """Comprueba que estén todas las columnas necesarias."""
    required = list(FEATURE_COLUMNS)
    if require_target:
        required.append(TARGET_COLUMN)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(
            "Faltan columnas obligatorias: " + ", ".join(missing)
        )


def _read_csv_safely(source: str | Path | BinaryIO | IO[bytes]) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        payload: str | Path | BytesIO = Path(source)
    else:
        raw = source.read()
        payload = BytesIO(raw)
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if isinstance(payload, BytesIO):
                payload.seek(0)
            return pd.read_csv(
                payload,
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            errors.append(str(exc))
        except pd.errors.ParserError as exc:
            errors.append(f"Error de formato CSV: {exc}")
            break
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - mensajes de bibliotecas
            errors.append(str(exc))
            break
    raise DataValidationError(
        "No fue posible leer el archivo CSV. " + " | ".join(errors[-1:])
    )


def read_tabular_file(
    source: str | Path | BinaryIO | IO[bytes], filename: str | None = None
) -> pd.DataFrame:
    """Lee un CSV o Excel y devuelve un DataFrame sin modificar.

    ``source`` puede ser una ruta o un archivo cargado en memoria. Se valida la
    extensión cuando se proporciona ``filename``.
    """
    name = filename
    if name is None and isinstance(source, (str, Path)):
        name = str(source)
    suffix = Path(name).suffix.lower() if name else ""

    if suffix in {".csv", ".txt"} or not suffix:
        df = _read_csv_safely(source)
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(source)
        except ImportError as exc:
            raise DataValidationError(
                "Falta un motor de Excel. Instala openpyxl (y xlrd para .xls)."
            ) from exc
        except Exception as exc:
            raise DataValidationError(
                f"No fue posible leer el archivo Excel: {exc}"
            ) from exc
    else:
        raise DataValidationError(
            "Formato no soportado. Carga un archivo .csv, .xlsx o .xls."
        )

    if df.empty:
        raise DataValidationError("El archivo no contiene filas de datos.")
    if df.shape[0] > 50_000:
        raise DataValidationError(
            "El archivo supera el límite de 50 000 filas del prototipo."
        )
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga y valida el dataset de entrenamiento."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataValidationError(
            f"No se encontró el dataset en {path}. Ejecuta el generador de datos."
        ) from exc
    df = normalize_columns(df)
    validate_required_columns(df, require_target=True)

    missing = df[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].isna().sum()
    if int(missing.sum()) > 0:
        detail = ", ".join(f"{column}: {int(count)}" for column, count in missing.items() if count)
        raise DataValidationError(f"El dataset contiene valores faltantes ({detail}).")

    numeric = df.copy()
    for column in list(FEATURE_COLUMNS) + [TARGET_COLUMN]:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    if numeric[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].isna().any().any():
        raise DataValidationError(
            "El dataset contiene valores no numéricos en las columnas del modelo."
        )

    out_of_range: list[str] = []
    for column, (low, high) in FEATURE_RANGES.items():
        invalid = ~numeric[column].between(low, high)
        if invalid.any():
            out_of_range.append(f"{column} ({int(invalid.sum())})")
    if out_of_range:
        raise DataValidationError(
            "El dataset contiene valores fuera de rango: " + ", ".join(out_of_range)
        )

    if not set(numeric[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise DataValidationError(
            f"La columna {TARGET_COLUMN} debe contener únicamente 0 o 1."
        )
    return numeric


def get_training_medians(df: pd.DataFrame) -> dict[str, float]:
    """Calcula medianas de entrenamiento para imputar lotes incompletos."""
    return {
        column: float(df[column].median())
        for column in FEATURE_COLUMNS
    }


def prepare_batch(
    df: pd.DataFrame,
    medians: dict[str, float],
    *,
    keep_extra_columns: bool = True,
) -> tuple[pd.DataFrame, BatchReport]:
    """Valida un lote, imputa faltantes y limita valores fuera de rango.

    Las filas con texto no numérico se omiten y se reportan. Los valores
    faltantes se imputan con medianas del conjunto de entrenamiento y los
    valores extremos se limitan al rango operativo, dejando trazabilidad en las
    columnas ``datos_imputados`` y ``valores_limitados``.
    """
    normalized = normalize_columns(df)
    validate_required_columns(normalized)
    report = BatchReport(total_rows=len(normalized))

    extra = [
        column
        for column in normalized.columns
        if column not in FEATURE_COLUMNS and column != TARGET_COLUMN
    ]
    report.ignored_columns = [] if keep_extra_columns else extra

    working = normalized.copy()
    target_series: pd.Series | None = None
    if TARGET_COLUMN in working.columns:
        report.target_available = True
        target_series = pd.to_numeric(working[TARGET_COLUMN], errors="coerce")

    non_numeric_mask = pd.Series(False, index=working.index)
    for column in FEATURE_COLUMNS:
        original = working[column]
        converted = pd.to_numeric(original, errors="coerce")
        text_values = original.notna() & converted.isna()
        non_numeric_mask |= text_values
        working[column] = converted

    all_missing_mask = working[list(FEATURE_COLUMNS)].isna().all(axis=1)
    invalid_mask = non_numeric_mask | all_missing_mask
    invalid_df = working.loc[invalid_mask, list(FEATURE_COLUMNS)].copy()
    if not invalid_df.empty:
        reasons: list[str] = []
        for idx in invalid_df.index:
            row_reasons: list[str] = []
            if non_numeric_mask.loc[idx]:
                row_reasons.append("valores no numéricos")
            if all_missing_mask.loc[idx]:
                row_reasons.append("todos los campos vacíos")
            reasons.append("; ".join(row_reasons))
        invalid_df.insert(0, "motivo", reasons)
        invalid_df.insert(0, "fila_original", [int(i) + 2 for i in invalid_df.index])
    report.invalid_detail = invalid_df

    valid = working.loc[~invalid_mask].copy()
    report.invalid_rows = int(invalid_mask.sum())
    if valid.empty:
        raise DataValidationError(
            "Ninguna fila pudo procesarse. Revisa que las columnas contengan valores numéricos."
        )

    report.missing_values = {
        column: int(valid[column].isna().sum())
        for column in FEATURE_COLUMNS
        if valid[column].isna().any()
    }
    imputed_mask = valid[list(FEATURE_COLUMNS)].isna().any(axis=1)
    report.imputed_rows = int(imputed_mask.sum())
    for column in FEATURE_COLUMNS:
        valid[column] = valid[column].fillna(float(medians[column]))

    limited_flags = pd.Series(False, index=valid.index)
    for column, (low, high) in FEATURE_RANGES.items():
        outside = ~valid[column].between(low, high)
        count = int(outside.sum())
        if count:
            report.out_of_range_values[column] = count
            limited_flags |= outside
        valid[column] = valid[column].clip(lower=low, upper=high)
    report.limited_rows = int(limited_flags.sum())

    valid["datos_imputados"] = imputed_mask.astype(bool)
    valid["valores_limitados"] = limited_flags.astype(bool)
    if target_series is not None:
        valid[TARGET_COLUMN] = target_series.loc[valid.index]
        valid.loc[~valid[TARGET_COLUMN].isin([0, 1]), TARGET_COLUMN] = np.nan

    report.valid_rows = len(valid)
    if not keep_extra_columns:
        keep = list(FEATURE_COLUMNS) + (
            [TARGET_COLUMN] if TARGET_COLUMN in valid.columns else []
        )
        valid = valid[keep]
    return valid, report


def make_template_dataframe() -> pd.DataFrame:
    """Plantilla de ejemplo para predicción por lotes (sin variable objetivo)."""
    return pd.DataFrame(
        [
            {
                "basura_acumulada": 12.0,
                "obstruccion": 18.0,
                "flujo_agua": 82.0,
                "frecuencia_lluvias": 8.0,
                "dias_sin_limpieza": 21.0,
                "residuos_plasticos": 120.0,
            },
            {
                "basura_acumulada": 48.0,
                "obstruccion": 55.0,
                "flujo_agua": 46.0,
                "frecuencia_lluvias": 17.0,
                "dias_sin_limpieza": 180.0,
                "residuos_plasticos": 1250.0,
            },
            {
                "basura_acumulada": 82.0,
                "obstruccion": 88.0,
                "flujo_agua": 14.0,
                "frecuencia_lluvias": 26.0,
                "dias_sin_limpieza": 410.0,
                "residuos_plasticos": 3600.0,
            },
        ]
    )
