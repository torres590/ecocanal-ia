"""Configuración central y constantes del proyecto ECOCANAL IA.

Este módulo concentra rutas, columnas, rangos válidos, umbrales de riesgo y
parámetros compartidos por el entrenamiento, la evaluación y la interfaz.
Mantener una única fuente de verdad evita inconsistencias entre la app y los
modelos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"
EXAMPLES_DIR: Final[Path] = BASE_DIR / "examples"

DATASET_PATH: Final[Path] = DATA_DIR / "dataset_canales.csv"
LOGISTIC_MODEL_PATH: Final[Path] = MODELS_DIR / "logistic_model.pkl"
NEURAL_MODEL_PATH: Final[Path] = MODELS_DIR / "neural_network.keras"
PREPROCESSOR_PATH: Final[Path] = MODELS_DIR / "preprocessor.pkl"
METADATA_PATH: Final[Path] = MODELS_DIR / "model_metadata.json"
LOGO_PATH: Final[Path] = ASSETS_DIR / "logo.png"

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "basura_acumulada",
    "obstruccion",
    "flujo_agua",
    "frecuencia_lluvias",
    "dias_sin_limpieza",
    "residuos_plasticos",
)

TARGET_COLUMN: Final[str] = "riesgo_alto"

FEATURE_LABELS: Final[dict[str, str]] = {
    "basura_acumulada": "Basura acumulada (%)",
    "obstruccion": "Obstrucción del canal (%)",
    "flujo_agua": "Nivel de flujo de agua (índice 0-100)",
    "frecuencia_lluvias": "Frecuencia de lluvias (días/mes)",
    "dias_sin_limpieza": "Días desde la última limpieza",
    "residuos_plasticos": "Residuos plásticos (kg)",
}

FEATURE_SHORT_LABELS: Final[dict[str, str]] = {
    "basura_acumulada": "Basura",
    "obstruccion": "Obstrucción",
    "flujo_agua": "Flujo de agua",
    "frecuencia_lluvias": "Lluvias",
    "dias_sin_limpieza": "Días sin limpieza",
    "residuos_plasticos": "Plásticos",
}

FEATURE_HELP: Final[dict[str, str]] = {
    "basura_acumulada": "Porcentaje estimado de residuos sólidos visibles en el tramo analizado.",
    "obstruccion": "Porcentaje de la sección hidráulica que se encuentra obstruida.",
    "flujo_agua": "Índice relativo de circulación del agua: 0 = flujo casi nulo y 100 = flujo alto.",
    "frecuencia_lluvias": "Número promedio de días con lluvia al mes en la zona del canal.",
    "dias_sin_limpieza": "Días transcurridos desde la última jornada de limpieza registrada.",
    "residuos_plasticos": "Masa estimada de residuos plásticos presentes en el tramo (kg).",
}

# Rangos usados para validar formularios y archivos. Los límites son
# operativos del prototipo académico, no mediciones ambientales oficiales.
FEATURE_RANGES: Final[dict[str, tuple[float, float]]] = {
    "basura_acumulada": (0.0, 100.0),
    "obstruccion": (0.0, 100.0),
    "flujo_agua": (0.0, 100.0),
    "frecuencia_lluvias": (0.0, 30.0),
    "dias_sin_limpieza": (0.0, 730.0),
    "residuos_plasticos": (0.0, 5000.0),
}

FEATURE_STEPS: Final[dict[str, float]] = {
    "basura_acumulada": 1.0,
    "obstruccion": 1.0,
    "flujo_agua": 1.0,
    "frecuencia_lluvias": 1.0,
    "dias_sin_limpieza": 1.0,
    "residuos_plasticos": 10.0,
}

# Clasificación operativa de riesgo a partir de la probabilidad estimada.
# Alto coincide con el umbral binario de la variable objetivo (0.50).
RISK_LOW_MAX: Final[float] = 0.35
RISK_HIGH_MIN: Final[float] = 0.50
CLASSIFICATION_THRESHOLD: Final[float] = 0.50

RISK_LEVELS: Final[tuple[str, ...]] = ("Bajo", "Medio", "Alto")
RISK_COLORS: Final[dict[str, str]] = {
    "Bajo": "#1F8A70",
    "Medio": "#E0A100",
    "Alto": "#C0392B",
}

# Paleta visual institucional.
COLORS: Final[dict[str, str]] = {
    "verde_oscuro": "#0B3D2E",
    "verde": "#1F8A70",
    "verde_claro": "#46B98E",
    "agua": "#1E6F9F",
    "agua_clara": "#63B7D8",
    "blanco": "#FFFFFF",
    "gris_claro": "#F2F6F4",
    "gris": "#DDE7E2",
    "texto": "#173B30",
}

APP_NAME: Final[str] = "ECOCANAL IA"
APP_SLOGAN: Final[str] = "Inteligencia artificial para proteger nuestros canales y ríos"
APP_DISCLAIMER: Final[str] = (
    "Prototipo académico basado en un dataset sintético. No realiza mediciones "
    "ambientales reales ni detecta residuos mediante cámaras. Sus predicciones "
    "son estimaciones orientativas para apoyar la priorización de mantenimiento."
)

MODEL_KEYS: Final[tuple[str, ...]] = ("logistica", "red_neuronal")
MODEL_LABELS: Final[dict[str, str]] = {
    "logistica": "Regresión Logística",
    "red_neuronal": "Red Neuronal Artificial",
}

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20

# Parámetros documentados de generación del dataset sintético.
DATASET_SEED: Final[int] = 2026
DATASET_ROWS: Final[int] = 1800


def classify_risk(probability: float) -> str:
    """Convierte una probabilidad de riesgo alto en una etiqueta operativa.

    Bajo: p < 0.35; Medio: 0.35 <= p < 0.50; Alto: p >= 0.50.
    """
    probability = float(probability)
    if probability >= RISK_HIGH_MIN:
        return "Alto"
    if probability >= RISK_LOW_MAX:
        return "Medio"
    return "Bajo"
