"""Pruebas de humo de la interfaz Streamlit y sus flujos principales."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

PAGES = [
    "Panel general",
    "Predicción individual",
    "Predicción por lotes",
    "Comparación de modelos",
    "Recomendaciones ambientales",
    "Metodología y documentación",
]


def _text_content(app: AppTest) -> str:
    return "\n".join(
        str(element.value)
        for element in (list(app.markdown) + list(app.info) + list(app.warning))
        if getattr(element, "value", None) is not None
    )


def test_all_pages_render_without_exceptions() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=240)
    app.run()
    assert not app.exception
    for page in PAGES:
        app.radio(key="navigation").set_value(page).run()
        assert not app.exception, f"Excepción en {page}: {[str(e.value) for e in app.exception]}"
        assert len(app.markdown) > 3


def test_individual_prediction_flow() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=240)
    app.run()
    app.radio(key="navigation").set_value("Predicción individual").run()
    app.button(key="demo_Alto").click().run()
    assert not app.exception

    submit = next(button for button in app.button if button.label == "PREDECIR RIESGO")
    submit.click().run()
    assert not app.exception

    content = _text_content(app)
    assert "Nivel de riesgo consensuado" in content
    assert "Priorizar la limpieza" in content
    assert "Regresión Logística" in content
    assert "Red Neuronal Artificial" in content


def test_batch_example_with_labels_flow() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=240)
    app.run()
    app.radio(key="navigation").set_value("Predicción por lotes").run()
    app.radio(key="batch_source_mode").set_value("Usar lote de ejemplo").run()
    app.checkbox[0].set_value(True).run()

    process = next(button for button in app.button if button.label == "PROCESAR LOTE")
    process.click().run()
    assert not app.exception
    assert len(app.dataframe) >= 2
    content = _text_content(app)
    assert "Evaluación con etiquetas reales" in content
    assert "nunca se usaron las predicciones como etiquetas" in content


def test_batch_without_labels_does_not_show_metrics() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=240)
    app.run()
    app.radio(key="navigation").set_value("Predicción por lotes").run()
    app.radio(key="batch_source_mode").set_value("Usar lote de ejemplo").run()
    process = next(button for button in app.button if button.label == "PROCESAR LOTE")
    process.click().run()
    assert not app.exception
    content = _text_content(app)
    assert "No se calculan Accuracy" in content
