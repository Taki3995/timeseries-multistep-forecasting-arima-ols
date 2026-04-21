import numpy as np
import pandas as pd
from utility import pinv_svd, read_d_from_adf, apply_differencing

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
    X[:, 0] = 1.0  # Intersección
    
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

def estimate_two_phase_model(z, eps_hat, p, q, H):
    """
    Fase 2: Estima un modelo OLS independiente para cada horizonte h in {1 ... H}.
    Usa p rezagos autorregresivos y q rezagos de media móvil (usando eps_hat).
    Entrena sobre la serie diferenciada z (estacionaria).
    """
    n = len(z)
    models = {}  # Guardaremos los coeficientes por horizonte h
    
    # El máximo rezago necesario para tener datos completos en cada fila de atributos
    max_lag = max(p, q) if p > 0 or q > 0 else 1
    
    # Iteramos sobre cada paso
    for h in range(1, H + 1):
        # Para predecir z_{t+h}, necesitamos conocer z y eps en t, t-1,...
        t_start = max_lag
        t_end = n - h
        
        if t_end <= t_start:
            raise ValueError(f"No hay suficientes datos para el horizonte {h} con lags p={p}, q={q}.")
            
        n_obs = t_end - t_start
        
        # Objetivo: z_{t+h} (serie diferenciada)
        Y_target = z[t_start + h : t_end + h]
        
        # Diseñar Matriz X
        X = np.zeros((n_obs, 1 + p + q))
        X[:, 0] = 1.0  # Intersección
        
        # Rezagos AR de 'z': z_t, z_{t-1}, ... z_{t-p+1}
        for i in range(p):
            X[:, 1 + i] = z[t_start - i : t_end - i]
            
        # Rezagos MA de 'eps_hat': eps_t, eps_{t-1}, ... eps_{t-q+1}
        for j in range(q):
            X[:, 1 + p + j] = eps_hat[t_start - j : t_end - j]
            
        # Estimación OLS
        theta_h = pinv_svd(X) @ Y_target
        
        # Calcular métricas
        Y_pred = X @ theta_h
        residuals = Y_target - Y_pred
        sse = np.sum(residuals**2)
        aic = n_obs * np.log(sse) + 2 * (1 + p + q)
        
        models[h] = {
            'theta': theta_h,
            'sse': sse,
            'aic': aic
        }
        
    return models

def grid_search_arima(z_train, p_max, q_max, H, m=20):
    """
    Búsqueda grid de (p, q) minimizando el AIC del modelo OLS para h=1.
    Realizado EXCLUSIVAMENTE sobre la serie diferenciada z_train.
    """
    eps_hat = estimate_residuals(z_train, m)
    
    best_aic = float('inf')
    best_p = 0
    best_q = 0
    best_models = None
    
    n = len(z_train)
    
    for p in range(1, p_max + 1):
        for q in range(q_max + 1):
            max_lag_pq = max(p, q) if p > 0 or q > 0 else 1
            h = 1  # Evaluaremos AIC usando solo predecir 1 paso adelante
            
            t_start = max_lag_pq
            t_end = n - h
            
            if t_end <= t_start:
                continue  # Evitar configuraciones sin datos suficientes
                
            n_obs = t_end - t_start
            
            # Matriz X para h=1
            Y_target = z_train[t_start + h : t_end + h]
            X = np.zeros((n_obs, 1 + p + q))
            X[:, 0] = 1.0
            
            for i in range(p):
                X[:, 1 + i] = z_train[t_start - i : t_end - i]
            for j in range(q):
                X[:, 1 + p + j] = eps_hat[t_start - j : t_end - j]
                
            theta_h1 = pinv_svd(X) @ Y_target
            Y_pred = X @ theta_h1
            
            sse = np.sum((Y_target - Y_pred)**2)
            
            if sse <= 0:  # Evitar log(0)
                continue
                
            # AIC = n * ln(SSE) + 2(p + q + 1)
            aic = n_obs * np.log(sse) + 2 * (1 + p + q)
            
            if aic < best_aic:
                best_aic = aic
                best_p = p
                best_q = q
                
    # Una vez encontrados los mejores p, q, entrenamos para todos los h
    best_models = estimate_two_phase_model(z_train, eps_hat, best_p, best_q, H)
    
    return best_p, best_q, best_models, eps_hat, best_aic

def export_train_results(p, q, models, filepath='train.csv'):
    """
    Exporta coeficientes y métricas del entrenamiento a CSV.
    Formato: h, p, q, coeficientes (space-separated), SSE, AIC
    """
    rows = []
    for h, model_data in sorted(models.items()):
        theta = model_data['theta']
        sse = model_data['sse']
        aic = model_data['aic']
        
        # Convertir coeficientes a string
        coef_str = ' '.join([f'{c:.6f}' for c in theta])
        
        rows.append({
            'h': h,
            'p': p,
            'q': q,
            'coeficientes': coef_str,
            'SSE': sse,
            'AIC': aic
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Resultados de entrenamiento exportados a {filepath}")
    return df

def train_model(y, p_max=5, q_max=5, H=12, m=20):
    """
    Función principal de entrenamiento con split 80/20 cronológico.
    Lee d desde adf.csv, diferencia y_train, y entrena sobre z_train.
    RESTRICCIÓN: Fase 1 y 2 se aplican EXCLUSIVAMENTE sobre z_train (serie estacionaria).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    
    # Split cronológico 80/20
    n_train = int(0.8 * n)
    y_train = y[:n_train]
    y_test = y[n_train:]
    
    print(f"Split Train/Test: {n_train} train (80%), {len(y_test)} test (20%)")
    
    # Leer d desde adf.csv
    d = read_d_from_adf('adf.csv')
    print(f"Orden de integración d={d} (desde adf.csv)")
    
    # Diferenciar y_train para obtener z_train (serie estacionaria)
    z_train = apply_differencing(y_train, d)
    print(f"Longitud de z_train después de d={d} diferenciaciones: {len(z_train)}")
    
    # Grid search y estimación EXCLUSIVAMENTE sobre z_train
    best_p, best_q, models, residuals, best_aic = grid_search_arima(
        z_train, p_max, q_max, H, m
    )
    
    print(f"Mejores parámetros encontrados: p={best_p}, q={best_q}, AIC={best_aic:.4f}")
    
    # Exportar resultados de entrenamiento
    export_train_results(best_p, best_q, models, 'train.csv')
    
    return {
        'p': best_p,
        'q': best_q,
        'models': models,
        'residuals': residuals,
        'y_train': y_train,
        'y_test': y_test,
        'z_train': z_train,
        'd': d,
        'n_train': n_train,
        'H': H,
        'm': m
    }

if __name__ == '__main__':
    # Cargar datos
    try:
        data = pd.read_csv('tseries.csv', header=None)
        y = data.values.flatten().astype(float)
        train_result = train_model(y, p_max=5, q_max=5, H=12, m=20)
        print("Entrenamiento completado.")
    except FileNotFoundError:
        print("Error: No se encontró tseries.csv")
