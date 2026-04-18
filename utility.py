import numpy as np

def pinv_svd(A, tol=1e-15):
    r"""
    Calcula la pseudo-inversa de Moore-Penrose usando SVD.
    A^+ = V_h^T \cdot diag(1/S) \cdot U^T
    """
    m, n = A.shape
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    
    # Filtrar valores singulares pequeños
    S_inv = np.zeros_like(S)
    mask = S > tol
    S_inv[mask] = 1.0 / S[mask]
    
    # Construir inversa
    A_pinv = Vh.T @ np.diag(S_inv) @ U.T
    return A_pinv

def jarque_bera(x):
    """
    Calcula manualmente el estadístico de Jarque-Bera.
    """
    n = len(x)
    mu = np.mean(x)
    
    # Momentos centrales empíricos
    mu2 = np.mean((x - mu)**2)
    mu3 = np.mean((x - mu)**3)
    mu4 = np.mean((x - mu)**4)
    
    if mu2 == 0:
        return 0.0 # Caso donde todos los valores son iguales
        
    s = mu3 / (mu2**(3/2))
    k = mu4 / (mu2**2)
    
    jb = (n / 6.0) * (s**2 + ((k - 3)**2) / 4.0)
    return jb

def mnse(y_true, y_pred):
    """
    Calcula el Modified Nash-Sutcliffe Efficiency (mNSE).
    mNSE = 1 - (suma de errores absolutos / suma de desviaciones absolutas a la media)
    Si se usa la versión al cuadrado de especificación:
    mNSE = 1 - sum(e^2) / sum((x - mean(x))^2)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    numerador = np.sum((y_true - y_pred)**2)
    denominador = np.sum((y_true - np.mean(y_true))**2)
    
    if denominador == 0:
        return float('-inf') # Indefinido
        
    return 1 - (numerador / denominador)

def mape(y_true, y_pred):
    """
    Calcula el Mean Absolute Percentage Error (MAPE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Evitar división por cero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def recover_prediction(y_hat_diff, hist_Y, d):
    """
    Recuperación Newton: Vuelve al dominio original si d > 0.
    hat{Y}_{T+h} = hat{z}_{T+h} - sum_{k=1}^d (-1)^k binomial(d, k) Y_{t+h-k}
    
    Parámetros:
    y_hat_diff: Predicción en la serie diferenciada (hat{z}_{T+h})
    hist_Y: Arreglo cronológico extendido con los valores originales de Y necesarios (Y_{T+h-d} a Y_{T+h-1})
           Debe tener longitud 'd'.
    d: Orden de integración.
    """
    if d == 0:
        return y_hat_diff
        
    # Coeficientes binomiales: (-1)^k * C(d, k) para k=1..d
    # Usaremos una matriz de pascal/coeficientes que restaremos de la prediccion z
    
    import math
    
    Y_pred = y_hat_diff
    
    # k va de 1 a d.
    # hist_Y tiene longitud d y almacena:
    # hist_Y[0] = Y_{T+h-d}, hist_Y[1] = Y_{T+h-d+1}  ... hist_Y[d-1] = Y_{T+h-1}
    
    for k in range(1, d + 1):
        # Y_{T+h-k} corresponde al índice (d - k) en hist_Y
        y_val = hist_Y[d - k]
        
        coef_binomial = math.comb(d, k)
        signo = (-1)**k
        
        Y_pred -= signo * coef_binomial * y_val
        
    return Y_pred
