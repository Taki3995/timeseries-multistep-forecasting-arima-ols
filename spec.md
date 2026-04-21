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

## 4. Split Train/Test (OBLIGATORIO)

- **Proporción cronológica:** 80% entrenamiento / 20% test
- **Fórmula:** $n_{train} = \text{int}(0.8 \times \text{len(series)})$, $n_{test} = \text{len(series)} - n_{train}$
- **Restricción:** Fase 1 y 2 de Two-Phase OLS exclusivamente sobre el conjunto TRAIN

## 5. Tabla de Valores Críticos Dickey-Fuller

| Nivel de Significancia | Valor Crítico |
| ---------------------- | ------------- |
| 1% (α=0.01)            | -3.43         |
| 5% (α=0.05)            | -2.86         |
| 10% (α=0.10)           | -2.57         |

**Criterio:** Si $\hat{\tau} < $ valor crítico → Rechazar $H_0$ (serie es estacionaria)

## 6. Algoritmos y Procedimientos

1. **Identificación (ADF iterativo):**
   - Ciclo: $d = 0, 1, 2, \dots$ hasta que $\hat{\tau} < -2.86$ (valor crítico al 5%)
   - Para cada $d$, calcular el estadístico ADF
   - Exportar resultados a `adf.csv` (d, estadístico, valor_crítico, estacionario)
2. **Estimación (Two-Phase OLS sobre TRAIN):**
   - Aplicar Fase 1 y 2 exclusivamente al conjunto de entrenamiento (80%)
   - Exportar coeficientes y métricas a `train.csv`
3. **Predicción Multi-Step (sobre TEST):**
   - Iterar $h$ veces sobre el conjunto de prueba (20%)
   - Para $d > 0$, recuperar el dominio original usando el Teorema del Binomio de Newton:
     $$\hat{Y}_{T+h} = \text{Recuperar}(\hat{y}_{T+h}, \Delta^{d-1} Y_T, \dots, Y_T)$$
4. **Evaluación:**
   - Calcular mNSE, MAPE y Jarque-Bera
   - Exportar predicciones y métricas a `test.csv` (h, y_real, y_pred, error, mNSE, MAPE, JB)

## 7. Entregables de Datos

- **`adf.csv`:** d, estadístico, valor_crítico, estacionario
- **`train.csv`:** h, p, q, coeficientes, SSE, AIC
- **`test.csv`:** h, y_real, y_pred, error, mNSE, MAPE, JB

## NO USAR COMENTARIOS. solo dejar el codigo sin ningún tipo de comentario en el
