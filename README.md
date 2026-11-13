# ECOCANAL IA

**Sistema académico de apoyo a la priorización del riesgo de residuos en canales de agua.**

ECOCANAL IA es una aplicación web desarrollada con **Python**, **Streamlit**, **scikit-learn** y **TensorFlow/Keras**. Estima la probabilidad de que un canal presente un riesgo alto de acumulación de residuos, clasifica el riesgo en tres niveles, genera recomendaciones preventivas y compara dos modelos de clasificación binaria: una **Regresión Logística** y una **Red Neuronal Artificial**.

> **Aviso importante:** el dataset es sintético. El sistema no mide contaminación real, no utiliza sensores ni cámaras y no sustituye una inspección ambiental de campo. Sus resultados son demostrativos y académicos.

---

## Tabla de contenido

1. [Problemática](#problemática)
2. [Objetivos](#objetivos)
3. [Funcionalidades](#funcionalidades)
4. [Tecnologías](#tecnologías)
5. [Modelos de clasificación](#modelos-de-clasificación)
6. [Dataset](#dataset)
7. [Resultados reales](#resultados-reales)
8. [Estructura del proyecto](#estructura-del-proyecto)
9. [Instalación en Windows](#instalación-en-windows)
10. [Ejecución](#ejecución)
11. [Uso de la aplicación](#uso-de-la-aplicación)
12. [Formato de archivos por lotes](#formato-de-archivos-por-lotes)
13. [Métricas y evaluación](#métricas-y-evaluación)
14. [Pruebas y verificación](#pruebas-y-verificación)
15. [Google Colab](#google-colab)
16. [Limitaciones y uso responsable](#limitaciones-y-uso-responsable)
17. [Solución de problemas](#solución-de-problemas)

---

## Problemática

La acumulación de residuos en canales, drenajes y cuerpos de agua puede provocar obstrucciones, inundaciones, deterioro de la calidad del agua y afectación a la salud pública. La revisión de todos los canales con la misma frecuencia no siempre es viable, por lo que resulta útil priorizar los puntos que presentan mayor riesgo.

El proyecto plantea un escenario académico: a partir de variables operativas simuladas (basura acumulada, obstrucción, flujo de agua, lluvias, días sin limpieza y residuos plásticos), estimar el nivel de riesgo y ofrecer una recomendación preventiva. El objetivo no es predecir con precisión un canal real, sino demostrar un flujo completo de ciencia de datos: generación y documentación de datos, preprocesamiento, entrenamiento, evaluación, interpretación y despliegue en una interfaz.

## Objetivos

### Objetivo general

Desarrollar una aplicación web que estime el riesgo de acumulación de residuos en canales de agua mediante modelos de clasificación supervisada, presentando resultados comprensibles y recomendaciones de acción.

### Objetivos específicos

- Generar un dataset sintético reproducible y documentado.
- Implementar un preprocesamiento sin fuga de información.
- Entrenar una Regresión Logística y una Red Neuronal Artificial.
- Evaluar ambos modelos con las mismas muestras de prueba.
- Comparar métricas de clasificación e interpretar sus resultados.
- Permitir predicciones individuales y por lotes desde CSV o Excel.
- Generar recomendaciones según el nivel de riesgo.
- Verificar el funcionamiento de la interfaz y de los módulos con pruebas automatizadas.

## Funcionalidades

- **Inicio / Dashboard:** resumen del dataset, prevalencia, métricas principales, distribución del riesgo y canales simulados prioritarios.
- **Predicción individual:** formulario con validación de rangos, tres casos de ejemplo y probabilidad de ambos modelos.
- **Predicción por lotes:** carga de CSV/Excel, validación, imputación controlada, descarga de resultados y evaluación opcional si existe una columna objetivo real.
- **Comparación de modelos:** tabla de métricas, matrices de confusión, curvas ROC e interpretación de resultados.
- **Recomendaciones:** reglas preventivas según el nivel de riesgo y las señales del registro.
- **Documentación:** explicación del dataset, variables, modelo, métricas, limitaciones y uso responsable.
- **Modelo de consenso:** promedio de las probabilidades de ambos modelos.
- **Tema visual personalizado:** diseño responsive, logotipo propio y componentes visuales.

## Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Interfaz web | Streamlit |
| Manipulación de datos | pandas, NumPy |
| Modelos | scikit-learn, TensorFlow/Keras |
| Visualización | Matplotlib, Seaborn |
| Persistencia | joblib, formato `.keras` |
| Pruebas | pytest, Streamlit AppTest |
| Calidad de código | Ruff |
| Datos de ejemplo | CSV y Excel |

Versiones verificadas en el entorno de desarrollo:

- Python 3.12.16
- Streamlit 1.67.0
- pandas 3.1.1
- NumPy 2.5.4
- scikit-learn 1.9.1
- TensorFlow 2.21.0 (`tensorflow-cpu` en Linux)
- Matplotlib 3.11.2
- Seaborn 0.13.2
- pytest 9.2.0

> **Requisito importante:** TensorFlow no ofrece soporte para Python 3.14 en el momento de este proyecto. Use **Python 3.11 o 3.12**. En Windows puede comprobar la versión con `py -3.12 --version`.

## Modelos de clasificación

### Variable objetivo

`riesgo_alto` es una variable binaria:

- `0`: riesgo no alto.
- `1`: riesgo alto.

El umbral de decisión es `0.50`.

### Niveles de riesgo para la interfaz

| Nivel | Probabilidad consensuada | Prioridad |
|---|---:|---|
| Bajo | `p < 0.35` | Monitoreo periódico |
| Medio | `0.35 <= p < 0.50` | Inspección y limpieza preventiva |
| Alto | `p >= 0.50` | Limpieza prioritaria |

El **consenso** es el promedio simple de las probabilidades de la Regresión Logística y de la Red Neuronal. El nivel se calcula a partir de ese promedio.

### Regresión Logística

Modelo lineal para clasificación binaria. Las variables se estandarizan con `StandardScaler`; luego se estima la probabilidad mediante la función sigmoide. Se usa regularización L2 con `C=1.0`, solver `lbfgs` y `max_iter=2000`.

Ventajas en este proyecto:

- Entrenamiento muy rápido.
- Coeficientes interpretables.
- Buen desempeño con relaciones aproximadamente lineales.
- Menor costo computacional.

### Red Neuronal Artificial

Red densa implementada con Keras:

```text
6 variables de entrada
        ↓
24 neuronas · ReLU · regularización L2 (1e-4)
        ↓
Dropout 0.15
        ↓
12 neuronas · ReLU · regularización L2 (1e-4)
        ↓
1 neurona · activación sigmoide
```

El entrenamiento usa `binary_crossentropy`, optimizador Adam (`learning_rate=0.001`), parada temprana, reducción de la tasa de aprendizaje y un 15% interno del conjunto de entrenamiento para validación. El conjunto de prueba no participa en el ajuste de pesos ni en la selección de la época.

### Prevención de fuga de información

- La división entrenamiento/prueba es estratificada, `test_size=0.20`, `random_state=42`.
- El escalador se ajusta **solo** con entrenamiento.
- Ambos modelos usan el mismo preprocesamiento y el mismo conjunto de prueba.
- Las medianas usadas para imputar lotes también provienen del entrenamiento.
- La red neuronal usa un subconjunto interno del entrenamiento para validación.
- El conjunto de prueba se evalúa una vez al final.
- En lotes sin columna objetivo no se calculan métricas supervisadas.

## Dataset

El archivo principal es `data/dataset_canales.csv`.

- **Registros:** 1800
- **Variables de entrada:** 6
- **Variable objetivo:** `riesgo_alto`
- **Semilla:** `2026`
- **Prevalencia de riesgo alto:** 42.83%
- **División:** 1440 entrenamiento / 360 prueba

### Variables

| Variable | Unidad | Rango operativo | Descripción |
|---|---|---:|---|
| `basura_acumulada` | % | 0–100 | Porcentaje de basura acumulada en el canal. |
| `obstruccion` | % | 0–100 | Grado de obstrucción del flujo. |
| `flujo_agua` | índice | 0–100 | Nivel relativo de flujo de agua; valores mayores indican mejor circulación. |
| `frecuencia_lluvias` | días/mes | 0–30 | Días con lluvia en el mes. |
| `dias_sin_limpieza` | días | 0–730 | Tiempo desde la última limpieza. |
| `residuos_plasticos` | kg | 0–5000 | Residuos plásticos estimados. |

La generación de la variable objetivo combina reglas ambientales y un componente aleatorio para producir solapamiento entre clases. Esto evita un problema trivial y permite observar errores reales de clasificación. La documentación detallada está en [`data/README_dataset.md`](data/README_dataset.md).

Para regenerar el dataset:

```bash
python -m src.generate_dataset
```

La regeneración es reproducible con la semilla documentada. El hash del dataset se registra en `models/model_metadata.json` al entrenar.

## Resultados reales

Resultados obtenidos en el conjunto de prueba de **360 registros** con los artefactos incluidos en `models/`:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Tiempo de entrenamiento |
|---|---:|---:|---:|---:|---:|---:|
| Regresión Logística | **79.17%** | **81.60%** | **66.23%** | **73.12%** | **84.50%** | **0.054 s** |
| Red Neuronal | 77.78% | 79.37% | 64.94% | 71.43% | 84.29% | 15.662 s |

**Mejor modelo por F1:** Regresión Logística.

### Matrices de confusión

**Regresión Logística**

|  | Predicción 0 | Predicción 1 |
|---|---:|---:|
| **Real 0** | 183 (TN) | 23 (FP) |
| **Real 1** | 52 (FN) | 102 (TP) |

**Red Neuronal**

|  | Predicción 0 | Predicción 1 |
|---|---:|---:|
| **Real 0** | 180 (TN) | 26 (FP) |
| **Real 1** | 54 (FN) | 100 (TP) |

### Interpretación

- La Regresión Logística obtuvo el mejor F1 y la mejor accuracy, además de un tiempo de entrenamiento mucho menor.
- La Red Neuronal quedó ligeramente por debajo en F1 y recall, aunque con un ROC-AUC cercano.
- Ambos modelos tienen un recall moderado: detectan aproximadamente dos tercios de los casos de riesgo alto. En un escenario real esto implicaría que una parte de los canales de riesgo no sería priorizada automáticamente.
- La precision es relativamente alta: cuando el modelo indica riesgo alto, suele acertar, pero deja pasar algunos positivos.
- El desempeño es razonable para un dataset sintético con solapamiento, pero **no debe interpretarse como evidencia de funcionamiento en canales reales**.

Los resultados se recalculan y guardan al ejecutar:

```bash
python -m src.train_models
```

## Estructura del proyecto

```text
ECOCANAL_IA/
├── app.py                         # Aplicación Streamlit y navegación
├── requirements.txt               # Dependencias del proyecto
├── pytest.ini                     # Configuración de pruebas
├── README.md                      # Documentación principal
├── .streamlit/
│   └── config.toml                # Tema y configuración de Streamlit
├── assets/
│   └── logo.png                   # Logotipo de la aplicación
├── data/
│   ├── dataset_canales.csv        # Dataset sintético
│   └── README_dataset.md          # Documentación del dataset
├── examples/
│   ├── ejemplo_lote_canales.csv   # Lote de ejemplo sin etiquetas
│   └── ejemplo_lote_etiquetado.csv# Lote de ejemplo con objetivo
├── models/
│   ├── logistic_model.pkl         # Regresión Logística entrenada
│   ├── neural_network.keras       # Red Neuronal entrenada
│   ├── preprocessor.pkl           # Escalador y medianas
│   └── model_metadata.json        # Métricas, split y configuración
├── notebooks/
│   └── ECOCANAL_IA_Colab.ipynb    # Guía reproducible para Google Colab
├── scripts/
│   └── generate_logo.py           # Generación del logotipo
├── src/
│   ├── config.py                  # Rutas, columnas, rangos y umbrales
│   ├── data_processing.py         # Carga, validación e imputación
│   ├── evaluation.py              # Métricas y evaluación por lote
│   ├── generate_dataset.py        # Generador sintético reproducible
│   ├── predictions.py             # Carga de artefactos y predicciones
│   ├── recommendations.py         # Reglas de recomendación
│   ├── train_models.py            # Entrenamiento y persistencia
│   └── ui.py                      # Componentes visuales y CSS
└── tests/
    ├── test_app.py                # Renderizado y flujos de la app
    ├── test_data_processing.py    # Validación e imputación
    └── test_models.py             # Predicciones y artefactos
```

## Instalación en Windows

### 1. Verificar Python

Abra **PowerShell** o **CMD** y ejecute:

```powershell
py -3.12 --version
```

Si no está instalado, descargue Python 3.12 desde <https://www.python.org/downloads/> y marque la opción **Add Python to PATH**.

> No use Python 3.14 para este proyecto si TensorFlow no ofrece una versión compatible. Puede tener varias versiones instaladas y elegir explícitamente `py -3.12`.

### 2. Entrar al proyecto

```powershell
cd C:\ruta\donde\esta\ECOCANAL_IA
```

### 3. Crear y activar el entorno virtual

**PowerShell:**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**CMD:**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

Sabrá que el entorno está activo porque el prompt muestra `(.venv)`.

### 4. Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Generar datos y entrenar

El repositorio ya incluye dataset y modelos. Si desea reproducir todo desde cero:

```powershell
python -m src.generate_dataset
python -m src.train_models
```

### 6. Ejecutar la aplicación

```powershell
streamlit run app.py
```

Se abrirá el navegador en una dirección similar a `http://localhost:8501`.

## Ejecución

Comandos principales:

```bash
# Generar el dataset sintético
python -m src.generate_dataset

# Entrenar ambos modelos y actualizar metadatos
python -m src.train_models

# Ejecutar la aplicación
streamlit run app.py

# Ejecutar pruebas
python -m pytest

# Revisar calidad del código
ruff check .
```

En Linux/macOS, si el entorno está en `.venv`:

```bash
.venv/bin/python -m src.generate_dataset
.venv/bin/python -m src.train_models
.venv/bin/streamlit run app.py
```

## Uso de la aplicación

### Predicción individual

1. Abra la sección **Predicción individual**.
2. Ingrese los seis valores del canal o seleccione un caso de ejemplo.
3. Presione **Analizar riesgo**.
4. Revise la probabilidad de la Regresión Logística, la Red Neuronal y el consenso.
5. Consulte el nivel de riesgo y las recomendaciones generadas.

Los valores fuera de rango se limitan al rango operativo y se informa al usuario. Los campos vacíos se imputan con las medianas del entrenamiento.

### Predicción por lotes

1. Abra la sección **Predicción por lotes**.
2. Descargue un archivo de ejemplo o cargue su propio CSV/Excel.
3. Revise el resumen de validación.
4. Descargue los resultados con probabilidades, nivel de riesgo, prioridad y recomendación.

Puede usar:

- `examples/ejemplo_lote_canales.csv`: sin columna objetivo; no calcula métricas.
- `examples/ejemplo_lote_etiquetado.csv`: con `riesgo_alto`; calcula métricas reales.

### Comparación de modelos

Muestra las métricas de ambos modelos, matrices de confusión, curvas ROC y una interpretación breve. Las métricas provienen del mismo conjunto de prueba.

### Recomendaciones

Presenta reglas preventivas según el nivel de riesgo, la obstrucción, los días sin limpieza, la basura acumulada, los plásticos y las lluvias.

## Formato de archivos por lotes

El archivo debe contener las seis variables de entrada:

```text
basura_acumulada,obstruccion,flujo_agua,frecuencia_lluvias,dias_sin_limpieza,residuos_plasticos
62,48,22.5,14,210,1800
85,91,9.2,26,520,4100
...
```

La columna objetivo es **opcional**:

```text
...,residuos_plasticos,riesgo_alto
...,1800,0
...,4100,1
```

Reglas de validación:

- Se aceptan CSV y Excel (`.xlsx`, `.xls`).
- Las filas no numéricas o completamente vacías se omiten.
- Los valores faltantes se imputan con medianas del entrenamiento.
- Los valores fuera de rango se limitan y se marcan.
- Las columnas `datos_imputados` y `valores_limitados` indican ajustes realizados.
- Si no existe `riesgo_alto`, solo se muestran estadísticas descriptivas.
- Las predicciones nunca se usan como etiquetas reales para calcular métricas.

## Métricas y evaluación

- **Accuracy:** proporción de clasificaciones correctas.
- **Precision:** de los casos predichos como riesgo alto, cuántos eran realmente de riesgo alto.
- **Recall:** de los casos realmente de riesgo alto, cuántos fueron detectados.
- **F1:** media armónica entre precision y recall.
- **ROC-AUC:** capacidad de ordenar casos de riesgo alto por encima de los de riesgo no alto.
- **Matriz de confusión:** verdaderos negativos, falsos positivos, falsos negativos y verdaderos positivos.

En este problema, el **recall** es especialmente relevante si el objetivo es no dejar canales de riesgo sin priorizar. La **precision** importa para evitar inspecciones innecesarias. El F1 resume el equilibrio entre ambos.

## Pruebas y verificación

El proyecto incluye **24 pruebas automatizadas**:

```bash
python -m pytest
```

Resultado verificado:

```text
24 passed
```

Cobertura funcional de las pruebas:

- Generación y estructura del dataset.
- Validación de columnas y rangos.
- División estratificada y ausencia de fuga de información.
- Carga de modelos y preprocesador.
- Predicciones individuales y coherencia de probabilidades.
- Predicción por lotes con y sin etiquetas.
- Verificación de que las predicciones no se usan como etiquetas reales.
- Renderizado de las secciones principales con `streamlit.testing.v1.AppTest`.
- Flujo de la interfaz sin excepciones.

Calidad de código:

```bash
ruff check . --exclude .venv
```

Resultado verificado:

```text
All checks passed!
```

## Google Colab

La libreta [`notebooks/ECOCANAL_IA_Colab.ipynb`](notebooks/ECOCANAL_IA_Colab.ipynb) contiene una guía para:

- Subir y descomprimir el proyecto.
- Instalar dependencias.
- Regenerar el dataset.
- Entrenar los modelos.
- Verificar métricas y una predicción.
- Abrir la interfaz con el proxy de Colab.

Para la entrega y uso normal se recomienda ejecutar Streamlit localmente. Colab puede ser útil para demostrar el entrenamiento, pero los túneles y la visualización web pueden ser menos estables.

## Limitaciones y uso responsable

- **Datos sintéticos:** no representan canales reales ni condiciones ambientales medidas.
- **Sin sensores:** no hay integración con cámaras, IoT, imágenes satelitales ni estaciones meteorológicas.
- **Sin validación temporal o espacial:** no se evalúa el desempeño en distintos ríos, estaciones o años.
- **Probabilidades no calibradas:** la salida debe interpretarse como un puntaje de apoyo, no como una probabilidad ambiental certificada.
- **Desempeño moderado:** el recall ronda dos tercios; el sistema puede omitir canales de riesgo.
- **Dependencia de rangos:** los límites operativos son supuestos académicos.
- **No es un sistema de alerta:** requiere revisión humana, inspección de campo y fuentes oficiales.
- **Sesgo del generador:** las reglas del dataset sintético incorporan supuestos del autor.
- **Uso académico:** no debe emplearse para decisiones ambientales, sanitarias o presupuestarias sin validación adicional.

## Solución de problemas

### TensorFlow no se instala

Verifique la versión de Python:

```powershell
py -3.12 --version
```

Cree el entorno con Python 3.12. Si el error persiste, actualice `pip`:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### No se puede activar el entorno en PowerShell

Ejecute solo para la sesión actual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

También puede usar CMD con `.venv\Scripts\activate.bat`.

### `streamlit` no se reconoce como comando

Asegúrese de tener el entorno activo o use:

```powershell
python -m streamlit run app.py
```

### El puerto 8501 está ocupado

```powershell
streamlit run app.py --server.port 8502
```

### Mensajes `E0000 ... use_unbounded_threadpool`

Son advertencias internas de TensorFlow al usar ciertas operaciones de carga de datos. No impiden el entrenamiento y no afectan las métricas. Se pueden ignorar en este proyecto.

### Los modelos no cargan

Regenere los artefactos:

```powershell
python -m src.train_models
```

No modifique manualmente el formato de los archivos de `models/`.

---

**Proyecto académico — ECOCANAL IA.**  
Los resultados incluidos corresponden a un dataset sintético y a un conjunto de prueba reservado de 360 registros.
