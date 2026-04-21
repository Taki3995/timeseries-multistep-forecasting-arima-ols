# Manual de Usuario - Taller 2: Pronóstico Multi-paso ARIMA

## 1. Descripción General

Este proyecto implementa un pipeline completo de pronóstico de series de tiempo no-estacionarias mediante:
- **Fase ADF:** Identificación del orden de integración mediante test Dickey-Fuller Aumentado
- **Fase TRN:** Entrenamiento Two-Phase OLS con split cronológico 80/20
- **Fase TST:** Predicción multi-paso con recuperación de dominio original

## 2. Requisitos

- Python 3.7+
- Librerías: numpy, pandas, matplotlib
- Datos: serie temporal en `tseries.csv` (formato: una columna sin encabezado)

## 3. Ejecución del Pipeline

### Paso 1: Test ADF (Identificación de d)
```bash
python adf.py
```
**Salida:** `adf.csv` con resultados de diferenciación iterativa

### Paso 2: Entrenamiento
```bash
python trn.py
```
**Salida:** `train.csv` con coeficientes y métricas del modelo

### Paso 3: Predicción Multi-paso
```bash
python tst.py
```
**Salida:** `test.csv`, `prediccion.png` con resultados de predicción

## 4. Interpretación de Resultados

### adf.csv
- `d`: Orden de integración
- `adf_stat`: Estadístico τ̂ calculado
- `critical_value`: Valor crítico (α=0.05)
- `is_stationary`: True si la serie es estacionaria

### train.csv
- `h`: Horizonte de predicción
- `p`, `q`: Órdenes AR y MA
- `coeficientes`: Vectores de parámetros θ_h
- `SSE`, `AIC`: Métricas de ajuste

### test.csv
- `h`: Horizonte
- `y_real`, `y_pred`: Valores reales vs predichos
- `error`: Diferencia
- `mNSE`, `MAPE`, `JB`: Métricas globales

## 5. Restricciones Técnicas

✓ Solo librerías permitidas: numpy, pandas, matplotlib
✓ Pseudo-inversa vía SVD manual (NO np.linalg.pinv)
✓ Split 80/20 cronológico
✓ Entrenamiento exclusivo sobre TRAIN
✓ Prevención de data leakage en predicción
