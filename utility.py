import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PREPARACIÓN Y ESTACIONARIEDAD
# ==========================================

def schwert_rule(T):
    """
    Calcula el número máximo de rezagos (p_max) usando la regla de Schwert.
    """
    return int(np.floor(12 * (T / 100)**0.25))

def diff_series(y, d):
    """
    Aplica el operador de diferenciación delta iterativamente d veces.
    """
    y_diff = np.array(y)
    for _ in range(d):
        y_diff = np.diff(y_diff)
    return y_diff

def create_lag_matrix(y, p):
    """
    Construye las matrices de regresión para estimaciones OLS.
    Retorna la matriz de regresores X (rezagos) y el vector objetivo Y.
    """
    N = len(y)
    X = np.zeros((N - p, p))
    for i in range(p):
        X[:, i] = y[p - 1 - i : N - 1 - i]
    Y = y[p:]
    return X, Y

# ==========================================
# 2. PRONÓSTICO Y RECUPERACIÓN
# ==========================================

def factorial(n):
    """Función auxiliar para calcular factoriales sin importar librerías extra."""
    if n == 0: return 1
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res

def binom_coeff(n, k):
    """Calcula el coeficiente binomial nCk."""
    return factorial(n) // (factorial(k) * factorial(n - k))

def recover_prediction(z_pred, y_hist, d):
    """
    Recupera la predicción al dominio original usando el Teorema del Binomio de Newton.
    z_pred : Predicción en el dominio diferenciado (z_{t+h})
    y_hist : Array con el historial reciente de la serie original. 
             Debe terminar cronológicamente en Y_{t+h-1}.
    d      : Orden de integración.
    """
    if d == 0:
        return z_pred
        
    y_recup = z_pred
    for k in range(1, d + 1):
        coeff = (-1)**k * binom_coeff(d, k)
        y_recup -= coeff * y_hist[-k]
        
    return y_recup

# ==========================================
# 3. MÉTRICAS Y CRITERIOS DE INFORMACIÓN
# ==========================================

def calc_mnse(real, pred):
    """Modified Nash-Sutcliffe Efficiency (mNSE)."""
    real = np.array(real)
    pred = np.array(pred)
    mean_real = np.mean(real)
    num = np.sum(np.abs(real - pred))
    den = np.sum(np.abs(real - mean_real))
    return 1 - (num / den) if den != 0 else np.nan

def calc_mape(real, pred):
    """Mean Absolute Percentage Error (MAPE)."""
    real = np.array(real)
    pred = np.array(pred)
    valid = real != 0 # Prevenir división por cero
    return np.mean(np.abs((real[valid] - pred[valid]) / real[valid]))

def calc_rmse(real, pred):
    """Root Mean Square Error (RMSE)."""
    real = np.array(real)
    pred = np.array(pred)
    return np.sqrt(np.mean((real - pred)**2))

def calc_aic(sse, N, p):
    """Criterio de Información de Akaike (AIC)."""
    if sse <= 0: return np.nan
    return np.log(sse / N) + (2 * p / N)

def calc_bic(sse, N, p):
    """Criterio de Información Bayesiano (BIC)."""
    if sse <= 0: return np.nan
    return np.log(sse / N) + (p * np.log(N) / N)

# ==========================================
# 4. VALIDACIÓN DE RESIDUOS
# ==========================================

def jarque_bera_test(residuals):
    """
    Realiza el test de Jarque-Bera calculando los momentos centrales muestrales.
    Retorna el estadístico JB, Asimetría (S) y Curtosis (K).
    """
    res = np.array(residuals)
    n = len(res)
    mean_res = np.mean(res)
    
    # Momentos centrales (mu_k)
    mu2 = np.mean((res - mean_res)**2)
    mu3 = np.mean((res - mean_res)**3)
    mu4 = np.mean((res - mean_res)**4)
    
    if mu2 == 0:
        return np.nan, np.nan, np.nan
        
    # Asimetría y Curtosis
    S = mu3 / (mu2**(1.5))
    K = mu4 / (mu2**2)
    
    # Estadístico JB
    JB = (n / 6) * (S**2 + ((K - 3)**2) / 4)
    return JB, S, K

# ==========================================
# 5. GRÁFICOS Y AUTOCORRELACIÓN
# ==========================================

def calc_acf(y, max_lag):
    """Calcula la Función de Autocorrelación Muestral (ACF)."""
    y = np.array(y)
    N = len(y)
    mean_y = np.mean(y)
    var_y = np.sum((y - mean_y)**2)
    
    if var_y == 0:
        return np.zeros(max_lag + 1)
        
    acf = [1.0] # El rezago 0 siempre es 1
    for k in range(1, max_lag + 1):
        cov = np.sum((y[:N-k] - mean_y) * (y[k:] - mean_y))
        acf.append(cov / var_y)
    return np.array(acf)

def plot_acf(y, max_lag=20, title="Sample Autocorrelation Function", filename="acf.png"):
    """Genera el gráfico ACF estilo 'lollipop' con bandas de confianza al 95%."""
    acf_vals = calc_acf(y, max_lag)
    N = len(y)
    conf_int = 1.96 / np.sqrt(N) # Banda de confianza del 95%
    
    lags = np.arange(max_lag + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Crear el gráfico tipo stem (lollipop) replicando el estilo visual
    markerline, stemlines, baseline = plt.stem(lags, acf_vals, basefmt="k-")
    plt.setp(markerline, color='orangered', marker='o', markersize=5)
    plt.setp(stemlines, color='coral', linewidth=1)
    plt.setp(baseline, linewidth=0.5)
    
    # Bandas de confianza horizontales
    plt.axhline(y=conf_int, color='steelblue', linestyle='-', linewidth=1)
    plt.axhline(y=-conf_int, color='steelblue', linestyle='-', linewidth=1)
    
    plt.title(title, fontweight='bold')
    plt.xlabel('Lag')
    plt.ylabel('Sample Autocorrelation')
    plt.grid(True, alpha=0.4)
    plt.xlim(0, max_lag + 0.5)
    
    plt.savefig(filename)
    plt.close()