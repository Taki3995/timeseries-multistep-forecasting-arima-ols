import numpy as np
from utility import pinv_svd

def adf_test(y, max_lag=5):
    """
    Test Augmented Dickey-Fuller con grid search por AIC.
    Calcula todo manualmente (OLS con pinv_svd).
    Dy_t = c + beta*t + gamma*y_{t-1} + sum(delta_i * Dy_{t-i}) + e_t
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    dy = np.diff(y)  # dy[t] = y[t+1] - y[t] (longitud n-1)
    
    best_aic = float('inf')
    best_adf_stat = None
    best_p = 0
    
    for p in range(max_lag + 1):
        # Necesitamos datos alineados.
        # Elementos disponibles para Y de OLS es desde t=p hasta n-2 (índice max en dy es n-2)
        # N_obs número efectivo de observaciones para la regresión.
        n_obs = len(dy) - p
        
        # Y vector para regresión: Dy_t (desde t=p en dy)
        Y = dy[p:n_obs+p] 
        
        # Construimos X matriz de diseño: [const, trend, y_{t-1}, Dy_{t-1}, ... Dy_{t-p}]
        X = np.zeros((n_obs, 3 + p))
        
        # Constante
        X[:, 0] = 1.0
        # Tendencia temporal (t) adaptada a la venta utilizada
        X[:, 1] = np.arange(p + 1, n_obs + p + 1)
        # y_{t-1} = y desde t=p hasta n_obs+p-1 (recordar y tiene índice +1 respecto a dy)
        X[:, 2] = y[p:n_obs+p]
        
        # Rezagos Dy_{t-i}
        for i in range(1, p + 1):
            X[:, 2 + i] = dy[p - i : n_obs + p - i]
            
        # OLS estimation: theta_hat = X^+ * Y
        X_pinv = pinv_svd(X)
        theta_hat = X_pinv @ Y
        
        # Gamma está en el índice 2 (tercera columna: y_{t-1})
        gamma_hat = theta_hat[2]
        
        # Cálculo de errores OLS
        Y_pred = X @ theta_hat
        residuals = Y - Y_pred
        sse = np.sum(residuals**2)
        
        # Matriz de covarianza y errores estándar
        k = X.shape[1] # número de regresores
        sigma2 = sse / (n_obs - k)
        
        cov_matrix = sigma2 * (X_pinv @ X_pinv.T) # (X^T X)^{-1} es pinv_svd(X) * pinv_svd(X)^T
        se_gamma = np.sqrt(cov_matrix[2, 2])
        
        # Estadístico ADF
        adf_stat = gamma_hat / se_gamma
        
        # Akaike Information Criterion (AIC)
        # AIC = N * log(SSE) + 2*p
        aic = n_obs * np.log(sse) + 2 * p
        
        if aic < best_aic:
            best_aic = aic
            best_adf_stat = adf_stat
            best_p = p
            
    return best_adf_stat, best_p, best_aic
