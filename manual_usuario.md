# Manual de Usuario: Pronóstico Multi-step-ahead vía ARIMA (Two-Phase OLS)

Este manual detalla los pasos necesarios para replicar el experimento de pronóstico de series de tiempo utilizando un modelo ARIMA mediante el método de estimación Two-Phase OLS. El sistema evalúa la estacionariedad, optimiza los hiperparámetros mediante búsqueda en rejilla (Grid Search) y genera pronósticos directos independientes para múltiples horizontes.

---

## 1. Requisitos Previos y Preparación

Antes de ejecutar los scripts, asegúrese de contar con el entorno adecuado:

- **Lenguaje:** Python 3.x
- **Librerías requeridas:** `numpy`, `pandas`, `matplotlib`. (No se utilizan librerías externas de machine learning o estadística para los modelos matemáticos).
- **Datos de entrada:** El archivo de la serie de tiempo (por ejemplo, `ts_taller2.csv` o `tseries.csv`) debe estar en la misma carpeta que los scripts. El archivo debe contener una única columna con valores numéricos y **no debe tener encabezados**.

Los scripts que componen este proyecto son:

1.  `utility.py`: Motor matemático (matrices OLS, métricas, recuperación binomial, ACF).
2.  `adf.py`: Análisis de estacionariedad (Test Dickey-Fuller Aumentado).
3.  `trn.py`: Entrenamiento y optimización de hiperparámetros (Grid Search).
4.  `tst.py`: Evaluación en el set de prueba y generación de gráficos.

---

## 2. Ejecución Paso a Paso

El experimento debe ejecutarse en un orden estricto, ya que cada script genera archivos de configuración (`.csv`) que alimentan a la siguiente fase.

### Paso 1: Análisis de Estacionariedad

Ejecute en su terminal:
`python adf.py`

- **¿Qué hace?** Toma el 80% de los datos (entrenamiento) y realiza el Test de Dickey-Fuller Aumentado (ADF) bajo tres casos determinísticos iterando el orden de integración (`d`).
- **Salida:** Genera el archivo `adf.csv` con los resultados del test y los criterios de información (AIC). Además, imprime en consola el valor óptimo de `d` necesario para alcanzar la estacionariedad, y genera dos gráficos iniciales:
  - `time_series.png`: Serie de tiempo original.
  - `acf_initial.png`: Autocorrelación muestral de la serie original.

### Paso 2: Entrenamiento y Optimización

Ejecute en su terminal:
`python trn.py`

- **¿Qué hace?** Lee el valor `d` desde `adf.csv`. Luego, ejecuta un Grid Search iterando los rezagos autorregresivos (`p`) y de media móvil (`q`) entre 0 y 10 para minimizar el criterio AIC usando el horizonte base (h=1). Finalmente, estima los coeficientes independientes para cada horizonte solicitado usando Two-Phase OLS.
- **Salida:** Genera el archivo clave `train.csv`.

#### 📌 ¿Dónde y cómo encontrar los valores óptimos de 'p' y 'q'?

El profesor requiere trazabilidad de los hiperparámetros ganadores. Estos valores se guardan de forma permanente y estructurada en el archivo **`train.csv`**.

Si abre `train.csv` (con Excel, un bloc de notas o visualizador de CSV), verá que tiene las columnas: `h`, `type`, `index`, `value`.
Para encontrar los valores de `p`, `d`, `q` y `m`, mire **las primeras 4 filas** del archivo. Están etiquetadas con el horizonte `h = 0` y el tipo `meta` (metadata). Se leen de la siguiente manera:

- Fila 1: `0, meta, p, [Valor de p]` -> Indica cuántos rezagos AR se usaron.
- Fila 2: `0, meta, d, [Valor de d]` -> Indica el orden de diferenciación.
- Fila 3: `0, meta, q, [Valor de q]` -> Indica cuántos rezagos MA se usaron.
- Fila 4: `0, meta, m, [Valor de m]` -> Indica la longitud del modelo AR de la Fase 1.

_Nota: El archivo `tst.py` lee automáticamente estas filas etiquetadas como `meta` para reconstruir las ecuaciones matemáticas sin necesidad de intervención manual._

### Paso 3: Evaluación y Generación de Resultados

Ejecute en su terminal:
`python tst.py`

- **¿Qué hace?** Lee el modelo óptimo desde `train.csv`. Evalúa las predicciones exclusivamente en el 20% de los datos no vistos (Test set) iterando día a día para evitar fuga de datos (Data Leakage). Realiza la recuperación binomial para llevar los pronósticos al dominio original y calcula las métricas.
- **Salida:** 1. Genera el archivo `test.csv` con las métricas mNSE, MAPE y el estadístico Jarque-Bera (JB) para cada horizonte. 2. Genera gráficos comparativos de la predicción directa vs. el valor real: `plot_h1.png`, `plot_h3.png`, `plot_h5.png`. (El título incluirá los valores `p,d,q` automáticamente). 3. Genera gráficos de validación de residuos (ACF) para cada horizonte: `acf_residuos_h1.png`, `acf_residuos_h3.png`, `acf_residuos_h5.png`.

---

## 3. Interpretación de Validaciones Finales

- **mNSE y MAPE:** Indican la calidad de la predicción. Se espera que el rendimiento disminuya (el mNSE baje y el MAPE suba) a medida que el horizonte `h` aumenta (por ejemplo, h=5 será menos preciso que h=1).
- **Test de Jarque-Bera:** Si el valor impreso en consola y en `test.csv` es **menor a 5.991**, se acepta la normalidad de los residuos (deseable).
- **Gráficos ACF de Residuos:** Verifique que en las imágenes `acf_residuos_hX.png`, las barras rojas (a excepción del lag 0) se mantengan dentro de las bandas horizontales azules. Esto indica que no hay autocorrelación en los errores (Ruido Blanco).
