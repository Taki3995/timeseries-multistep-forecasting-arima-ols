# Specification: Taller 2 - Multi-step-ahead Forecasting via ARIMA

## 1. Objetivo General

Implementar y evaluar un modelo de pronóstico multi-paso (Multi-step-ahead) para series de tiempo no-estacionarias mediante un enfoque directo y el método Two-Phase OLS.

## 2. Requerimientos de Software y Archivos

### Librerías Permitidas

- **Core:** `python`, `numpy` (as `np`), `pandas`, `matplotlib`.
- **PROHIBIDO:** `statsmodels`, `scikit-learn`, `scipy.stats`.

### Estructura de Fuentes (OBLIGATORIA)

- `adf.py`: Implementación del Test de Dickey-Fuller Aumentado para identificar raíces unitarias y determinar el orden de integración $d$.
- `trn.py`: Entrenamiento y estimación de parámetros $(\phi_h, \theta_h)$ mediante Two-Phase OLS.
- `tst.py`: Evaluación, predicción multi-paso y generación de métricas.
- `utility.py`: Funciones auxiliares, álgebra lineal manual y cálculos estadísticos.

## 3. Especificaciones Técnicas y Fórmulas

### Álgebra Lineal Manual (OBLIGATORIO)

- **Pseudo-Inversa via SVD:** No usar `np.linalg.pinv`. Implementar `pinv_svd(A)` usando:
  $$A^+ = V_h^T \cdot \text{diag}(1/S) \cdot U^T$$

### Fase 1: Estimación de Residuos

Estimar residuos $\hat{\epsilon}_t$ mediante un proceso $AR(m)$ de orden largo ($m \gg p, q$):
$$\hat{\epsilon}_t = z_t - \sum_{i=1}^{m} \phi_{t-i} z_{t-i}$$

### Fase 2: Predicción Multi-Step-Ahead

Estimar un modelo independiente para cada horizonte $h \in \{1, \dots, H\}$ usando los residuos como regresores
$$\hat{Y}_{t+h} = \hat{z}_{t+h} - \sum_{k=1}^{d} (-1)^k \binom{d}{k} Y_{t+h-k}$$

### Validación de Residuos (Test Jarque-Bera)

Cálculo manual de momentos para el estadístico $JB$:

- **Momentos:** $\hat{\mu}_k = \frac{1}{n} \sum_{t=1}^{n} (\eta_t - \overline{\eta})^k$
- **Asimetría ($S$) y Curtosis ($K$):** $S = \frac{\hat{\mu}_3}{\hat{\mu}_2^{3/2}}$, $K = \frac{\hat{\mu}_4}{\hat{\mu}_2^2}$
- **Estadístico:** $JB = \frac{n}{6} (S^2 + \frac{(K-3)^2}{4})$
- **Valor Crítico ($\alpha=0.05$):** $5.991$

## 4. Algoritmos y Procedimientos

1. **Identificación:** Determinar $d$ tal que $(1-L)^d Y_t$ sea estacionaria mediante ADF y Grid Search para minimizar AIC ($AIC = N \log(SSE) + 2p$).
2. **Estimación:** Aplicar Two-Phase OLS para obtener los coeficientes del modelo.
3. **Recuperación:** Para $d > 0$, recuperar el dominio original usando el Teorema del Binomio de Newton:
   $$\hat{Y}_{T+h} = \text{Recuperar}(\hat{y}_{T+h}, \Delta^{d-1} Y_T, \dots, Y_T)$$
4. **Evaluación:** Reportar curvas comparativas, mNSE y MAPE.

## 5. Entregables de Datos

- Archivos CSV de configuración: `adf.csv`, `train.csv`, `test.csv`.
- Resultados parciales de cada etapa en formato `.csv`
