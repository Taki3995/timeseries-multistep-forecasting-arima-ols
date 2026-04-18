import numpy as np
from utility import pinv_svd

def estimate_residuals(y, m=20):
    """
    Fase 1: Estima los residuos e_t ajustando un modelo AR(m) largo.
    Retorna un arreglo de la misma longitud que y, con los primeros m valores en 0.
    """
    n = len(y)
    if n <= m:
        raise ValueError(f"La longitud de la serie ({n}) debe ser mayor a m ({m}).")
        
    n_obs = n - m
    
    # Y vector: desde t=m hasta n-1
    Y = y[m:]
    
    # Matriz X: constante + m rezagos
    X = np.zeros((n_obs, 1 + m))
    X[:, 0] = 1.0 # Intersección
    
    for i in range(1, m + 1):
        # y_{t-i}
        X[:, i] = y[m - i : n - i]
        
    # Estimación OLS: theta = X^+ * Y
    theta = pinv_svd(X) @ Y
    
    # Predecir y calcular residuos
    Y_pred = X @ theta
    eps_hat = Y - Y_pred
    
    # Rellenar los primeros m valores con 0 para mantener la estructura temporal
    residuals = np.zeros(n)
    residuals[m:] = eps_hat
    
    return residuals

def estimate_two_phase_model(y, eps_hat, p, q, H):
    """
    Fase 2: Estima un modelo OLS independiente para cada horizonte h in {1 ... H}.
    Usa p rezagos autorregresivos y q rezagos de media móvil (usando eps_hat).
    """
    n = len(y)
    models = {} # Guardaremos los coeficientes por horizonte h
    
    # El máximo rezago necesario para tener datos completos en cada fila de atributos
    max_lag = max(p, q)
    
    # Iteramos sobre cada paso
    for h in range(1, H + 1):
        # Para predecir Y_{t+h}, necesitamos conocer Y y eps en t, t-1,...
        # El tiempo "t" disponible comienza en max_lag (para poder mirar hacia atrás max_lag steps, es decir t-max_lag >= 0).
        # Además, como el target es t+h, t no puede exceder n - 1 - h (para que t+h <= n-1).
        
        t_start = max_lag
        t_end = n - h # para que el slice max de t sea n - h - 1, y Y target sea hasta n-1
        
        if t_end <= t_start:
            raise ValueError(f"No hay suficientes datos para el horizonte {h} con lags p={p}, q={q}.")
            
        n_obs = t_end - t_start
        
        # Objetivo Y: y_{t+h}
        Y_target = y[t_start + h : t_end + h]
        
        # Diseñar Matriz X
        X = np.zeros((n_obs, 1 + p + q))
        X[:, 0] = 1.0 # Intersección
        
        # Rezagos AR de 'y': y_t, y_{t-1}, ... y_{t-p+1}
        for i in range(p):
            X[:, 1 + i] = y[t_start - i : t_end - i]
            
        # Rezagos MA de 'eps_hat': eps_t, eps_{t-1}, ... eps_{t-q+1}
        for j in range(q):
            X[:, 1 + p + j] = eps_hat[t_start - j : t_end - j]
            
        # Estimación OLS
        theta_h = pinv_svd(X) @ Y_target
        models[h] = theta_h
        
    return models

def grid_search_arima(y, p_max, q_max, H, m=20):
    """
    Búsqueda grid de (p, q) minimizando el AIC del modelo OLS para h=1.
    """
    eps_hat = estimate_residuals(y, m)
    
    best_aic = float('inf')
    best_p = 0
    best_q = 0
    best_models = None
    
    n = len(y)
    
    for p in range(1, p_max + 1):
        for q in range(q_max + 1):
            max_lag_pq = max(p, q)
            h = 1 # Evaluaremos AIC usando solo predecir 1 paso adelante para decidir la estructura (p,q)
            
            t_start = max_lag_pq
            t_end = n - h
            
            if t_end <= t_start:
                continue # Evitar configuraciones sin datos suficientes
                
            n_obs = t_end - t_start
            
            # Matriz X para h=1
            Y_target = y[t_start + h : t_end + h]
            X = np.zeros((n_obs, 1 + p + q))
            X[:, 0] = 1.0
            
            for i in range(p):
                X[:, 1 + i] = y[t_start - i : t_end - i]
            for j in range(q):
                X[:, 1 + p + j] = eps_hat[t_start - j : t_end - j]
                
            theta_h1 = pinv_svd(X) @ Y_target
            Y_pred = X @ theta_h1
            
            sse = np.sum((Y_target - Y_pred)**2)
            
            if sse <= 0: # Evitar log(0)
                continue
                
            # AIC = n * ln(SSE) + 2(p + q + 1)
            # Aunque la versión estricta usa (SSE/n_obs), esto es equivalente para optimizar
            aic = n_obs * np.log(sse) + 2 * (1 + p + q)
            
            if aic < best_aic:
                best_aic = aic
                best_p = p
                best_q = q
                
    # Una vez encontrados los mejores p, q, entrenamos para todos los h
    best_models = estimate_two_phase_model(y, eps_hat, best_p, best_q, H)
    
    return best_p, best_q, best_models, eps_hat

def train_model(y, p_max=5, q_max=5, H=12, m=20):
    """
    Función principal de entrenamiento. Delega en GridSearchCV para obtener el mejor (p,q).
    """
    y = np.asarray(y, dtype=float)
    best_p, best_q, models, residuals = grid_search_arima(y, p_max, q_max, H, m)
    
    return {
        'p': best_p,
        'q': best_q,
        'models': models, # dictionary of h: theta_h
        'residuals': residuals
    }
