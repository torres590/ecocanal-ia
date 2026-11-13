"""Recomendaciones ambientales y preventivas basadas en el riesgo estimado."""

from __future__ import annotations

from collections.abc import Mapping

BASE_RECOMMENDATIONS: dict[str, str] = {
    "Bajo": (
        "Mantener monitoreo periódico del tramo y continuar con el calendario "
        "habitual de limpieza preventiva."
    ),
    "Medio": (
        "Programar una inspección de campo y una limpieza preventiva en el corto "
        "plazo para evitar que la acumulación aumente."
    ),
    "Alto": (
        "Priorizar la limpieza del canal y el retiro de residuos antes de que "
        "puedan ser arrastrados hacia el río."
    ),
}

PRIORITY_BY_RISK: dict[str, str] = {
    "Bajo": "P3 - Seguimiento",
    "Medio": "P2 - Preventiva",
    "Alto": "P1 - Prioritaria",
}


def _condition_recommendations(features: Mapping[str, float]) -> list[str]:
    """Reglas interpretables que complementan la recomendación principal."""
    notes: list[str] = []
    basura = float(features.get("basura_acumulada", 0))
    obstruccion = float(features.get("obstruccion", 0))
    flujo = float(features.get("flujo_agua", 100))
    lluvias = float(features.get("frecuencia_lluvias", 0))
    dias = float(features.get("dias_sin_limpieza", 0))
    plasticos = float(features.get("residuos_plasticos", 0))

    if dias >= 180:
        notes.append(
            "El tiempo desde la última limpieza es elevado: establecer o recuperar "
            "un calendario de mantenimiento periódico."
        )
    if obstruccion >= 60:
        notes.append(
            "La obstrucción supera el 60 %: realizar una revisión hidráulica y "
            "despejar el tramo para recuperar la circulación."
        )
    if basura >= 50:
        notes.append(
            "La basura acumulada es alta: reforzar la recolección y disponer "
            "puntos de segregación cercanos al canal."
        )
    if plasticos >= 2000:
        notes.append(
            "La masa de plásticos es considerable: organizar una jornada específica "
            "de recuperación y valorización de plásticos."
        )
    if flujo <= 30:
        notes.append(
            "El flujo es bajo: evaluar zonas de estancamiento y posibles focos de "
            "contaminación asociados al agua retenida."
        )
    if lluvias >= 20:
        notes.append(
            "La frecuencia de lluvias es alta: inspeccionar y limpiar antes de los "
            "eventos de lluvia para reducir el arrastre de residuos."
        )
    return notes


def generate_recommendations(
    features: Mapping[str, float],
    risk_level: str,
    *,
    models_agree: bool = True,
    max_items: int = 4,
) -> list[str]:
    """Genera recomendaciones ordenadas por relevancia.

    La primera recomendación siempre corresponde al nivel de riesgo global. Las
    demás se activan según condiciones específicas del canal.
    """
    if risk_level not in BASE_RECOMMENDATIONS:
        raise ValueError(f"Nivel de riesgo desconocido: {risk_level}")

    recommendations = [BASE_RECOMMENDATIONS[risk_level]]
    recommendations.extend(_condition_recommendations(features))
    if not models_agree:
        recommendations.append(
            "Los dos modelos no coinciden en la clasificación binaria: validar el "
            "caso con una inspección de campo antes de tomar decisiones."
        )
    if risk_level == "Bajo" and len(recommendations) == 1:
        recommendations.append(
            "No se detectan condiciones críticas en los datos ingresados; registrar "
            "la observación para conservar la trazabilidad del tramo."
        )
    return recommendations[:max_items]


def priority_label(risk_level: str) -> str:
    return PRIORITY_BY_RISK.get(risk_level, "Sin clasificar")
