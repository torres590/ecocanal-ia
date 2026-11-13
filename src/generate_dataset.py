"""Generador reproducible del dataset sintético de ECOCANAL IA.

Los datos NO representan mediciones ambientales reales. Se construyen con
relaciones plausibles entre acumulación de residuos, obstrucción, flujo,
lluvias, tiempo sin limpieza y plásticos, más ruido aleatorio. La variable
objetivo ``riesgo_alto`` se muestrea a partir de una probabilidad logística
latente, lo que permite entrenar y comparar modelos de clasificación con fines
académicos.

Uso:
    python -m src.generate_dataset
    python -m src.generate_dataset --rows 2500 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BASE_DIR,
    DATASET_PATH,
    DATASET_ROWS,
    DATASET_SEED,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

DATASET_DOC_PATH = BASE_DIR / "data" / "README_dataset.md"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_synthetic_dataset(
    rows: int = DATASET_ROWS, seed: int = DATASET_SEED
) -> pd.DataFrame:
    """Crea el dataset con distribuciones y relaciones documentadas.

    La generación es determinista para una misma semilla. No se utilizan
    etiquetas para construir las variables predictoras ni se introducen
    identificadores que puedan filtrar información.
    """
    rng = np.random.default_rng(seed)
    n = int(rows)
    if n < 200:
        raise ValueError("Se requieren al menos 200 filas para un entrenamiento estable.")

    # Factor latente de abandono/contaminación del canal. No se guarda en el
    # dataset: solo se usa para inducir correlaciones realistas.
    abandono = rng.gamma(shape=2.0, scale=0.5, size=n)

    basura = np.clip(8 + 30 * abandono + rng.normal(0, 10, n), 0, 100)
    dias_sin_limpieza = np.clip(
        5 + 120 * abandono + rng.normal(0, 60, n), 0, 730
    )
    residuos_plasticos = np.clip(
        30 + 700 * abandono + 6 * basura + rng.normal(0, 300, n), 0, 5000
    )
    obstruccion = np.clip(
        0.50 * basura + 0.15 * dias_sin_limpieza + rng.normal(0, 8, n), 0, 100
    )
    flujo_agua = np.clip(
        100 - 0.60 * obstruccion - 0.12 * basura + rng.normal(0, 8, n), 0, 100
    )
    frecuencia_lluvias = np.clip(rng.normal(14, 6, n), 0, 30)

    # Relación latente documentada. Los coeficientes son decisiones de diseño
    # académicas, no estimaciones ambientales publicadas.
    logit = (
        -2.30
        + 0.8
        * (
            0.035 * basura
            + 0.045 * obstruccion
            - 0.030 * flujo_agua
            + 0.045 * frecuencia_lluvias
            + 0.0030 * dias_sin_limpieza
            + 0.0005 * residuos_plasticos
            + 0.00006 * basura * obstruccion
        )
        + rng.normal(0, 1.2, n)
    )
    probabilidad_latente = _sigmoid(logit)
    riesgo_alto = rng.binomial(1, probabilidad_latente, size=n).astype(int)

    dataset = pd.DataFrame(
        {
            "basura_acumulada": np.round(basura, 1),
            "obstruccion": np.round(obstruccion, 1),
            "flujo_agua": np.round(flujo_agua, 1),
            "frecuencia_lluvias": np.round(frecuencia_lluvias, 1),
            "dias_sin_limpieza": np.round(dias_sin_limpieza).astype(int),
            "residuos_plasticos": np.round(residuos_plasticos, 1),
            TARGET_COLUMN: riesgo_alto,
        }
    )
    return dataset[list(FEATURE_COLUMNS) + [TARGET_COLUMN]]


def _build_documentation(dataset: pd.DataFrame, rows: int, seed: int) -> str:
    prevalence = float(dataset[TARGET_COLUMN].mean())
    descriptives = dataset[list(FEATURE_COLUMNS)].describe().T
    table = descriptives[["mean", "std", "min", "50%", "max"]].round(2)
    table_lines = [
        "| Variable | Media | Desv. est. | Mínimo | Mediana | Máximo |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature in FEATURE_COLUMNS:
        row = table.loc[feature]
        table_lines.append(
            f"| `{feature}` | {row['mean']:.2f} | {row['std']:.2f} | "
            f"{row['min']:.2f} | {row['50%']:.2f} | {row['max']:.2f} |"
        )
    table_md = "\n".join(table_lines)
    return f"""# Dataset sintético de ECOCANAL IA

> **Aviso académico:** este archivo contiene datos simulados. No proviene de
> mediciones ambientales reales ni de registros oficiales de canales. Su único
> propósito es demostrar el flujo de entrenamiento y evaluación de modelos.

## Reproducibilidad

- Generador: `src/generate_dataset.py`
- Semilla: `{seed}`
- Filas: `{rows}`
- Comando: `python -m src.generate_dataset --rows {rows} --seed {seed}`
- Prevalencia de `riesgo_alto = 1`: **{prevalence:.2%}**

## Proceso de generación

1. Se simula un factor latente de abandono con distribución Gamma (no se guarda).
2. Las variables de basura, días sin limpieza y plásticos crecen con ese factor
   y con ruido gaussiano.
3. La obstrucción depende de la basura y de los días sin limpieza; el flujo de
   agua se reduce cuando aumenta la obstrucción.
4. La frecuencia de lluvias se simula de forma independiente con una normal
   centrada en 14 días/mes.
5. Se calcula una probabilidad latente con una función logística que combina
   todas las señales, incluyendo una interacción basura × obstrucción y ruido.
6. La etiqueta binaria se muestrea con esa probabilidad, por lo que existen
   casos difíciles y errores inevitables incluso para modelos bien entrenados.

Los coeficientes son decisiones de diseño académicas y **no** deben
interpretarse como relaciones ambientales validadas científicamente.

## Variables

| Variable | Descripción | Unidad / rango |
|---|---|---|
| `basura_acumulada` | Residuos sólidos visibles en el tramo | % (0-100) |
| `obstruccion` | Sección hidráulica obstruida | % (0-100) |
| `flujo_agua` | Índice relativo de circulación | 0-100 |
| `frecuencia_lluvias` | Días con lluvia al mes | 0-30 |
| `dias_sin_limpieza` | Días desde la última limpieza | 0-730 |
| `residuos_plasticos` | Masa estimada de plásticos | kg (0-5000) |
| `riesgo_alto` | Objetivo binario | 0 = no alto, 1 = alto |

## Estadísticas descriptivas

{table_md}

## Uso responsable

- No presentar estos datos como mediciones reales.
- No usar el prototipo para decisiones ambientales críticas.
- Reentrenar con datos de campo validados antes de cualquier uso operativo.
"""


def save_dataset(
    dataset: pd.DataFrame,
    path: Path = DATASET_PATH,
    *,
    rows: int = DATASET_ROWS,
    seed: int = DATASET_SEED,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    DATASET_DOC_PATH.write_text(
        _build_documentation(dataset, rows=rows, seed=seed), encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el dataset sintético ECOCANAL IA")
    parser.add_argument("--rows", type=int, default=DATASET_ROWS)
    parser.add_argument("--seed", type=int, default=DATASET_SEED)
    args = parser.parse_args()

    dataset = generate_synthetic_dataset(rows=args.rows, seed=args.seed)
    path = save_dataset(dataset, rows=args.rows, seed=args.seed)
    prevalence = dataset[TARGET_COLUMN].mean()
    print(f"Dataset guardado en: {path}")
    print(f"Filas: {len(dataset)} | Riesgo alto: {prevalence:.2%}")
    print(f"Documentación: {DATASET_DOC_PATH}")


if __name__ == "__main__":
    main()
