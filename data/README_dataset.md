# Dataset sintético de ECOCANAL IA

> **Aviso académico:** este archivo contiene datos simulados. No proviene de
> mediciones ambientales reales ni de registros oficiales de canales. Su único
> propósito es demostrar el flujo de entrenamiento y evaluación de modelos.

## Reproducibilidad

- Generador: `src/generate_dataset.py`
- Semilla: `2026`
- Filas: `1800`
- Comando: `python -m src.generate_dataset --rows 1800 --seed 2026`
- Prevalencia de `riesgo_alto = 1`: **42.83%**

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

| Variable | Media | Desv. est. | Mínimo | Mediana | Máximo |
|---|---:|---:|---:|---:|---:|
| `basura_acumulada` | 37.93 | 22.32 | 0.00 | 34.35 | 100.00 |
| `obstruccion` | 37.70 | 24.26 | 0.00 | 33.10 | 100.00 |
| `flujo_agua` | 72.61 | 18.33 | 7.30 | 75.35 | 100.00 |
| `frecuencia_lluvias` | 13.95 | 5.90 | 0.00 | 14.10 | 30.00 |
| `dias_sin_limpieza` | 127.53 | 97.90 | 0.00 | 115.00 | 633.00 |
| `residuos_plasticos` | 956.76 | 660.99 | 0.00 | 842.75 | 4057.00 |

## Uso responsable

- No presentar estos datos como mediciones reales.
- No usar el prototipo para decisiones ambientales críticas.
- Reentrenar con datos de campo validados antes de cualquier uso operativo.
