import numpy as np
import pandas as pd
from utility import pinv_svd, read_d_from_adf, apply_differencing

def estimate_residuals(y, m=20):
    n = len(y)
    if n <= m:
        raise ValueError(f"La longitud de la serie ({n}) debe ser mayor a m ({m}).")
    n_obs = n - m
    Y = y[m:]
    X = np.zeros((n_obs, 1 + m))
    X[:, 0] = 1.0
    for i in range(1, m + 1):
        X[:, i] = y[m - i : n - i]
    theta = pinv_svd(X) @ Y
    Y_pred = X @ theta
    eps_hat = Y - Y_pred
    residuals = np.zeros(n)
    residuals[m:] = eps_hat
    return residuals

def estimate_two_phase_model(z, eps_hat, p, q, H):
    n = len(z)
    models = {}
    max_lag = max(p, q) if p > 0 or q > 0 else 1
    for h in range(1, H + 1):
        t_start = max_lag
        t_end = n - h
        if t_end <= t_start:
            raise ValueError(f"No hay suficientes datos para el horizonte {h} con lags p={p}, q={q}.")
        n_obs = t_end - t_start
        Y_target = z[t_start + h : t_end + h]
        X = np.zeros((n_obs, 1 + p + q))
        X[:, 0] = 1.0
        for i in range(p):
            X[:, 1 + i] = z[t_start - i : t_end - i]
        for j in range(q):
            X[:, 1 + p + j] = eps_hat[t_start - j : t_end - j]
        theta_h = pinv_svd(X) @ Y_target
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

def grid_search_arima(z_train, p_max, q_max, H):
    best_aic = float('inf')
    best_p = 0
    best_q = 0
    best_models = None
    best_eps_hat = None
    n = len(z_train)
    for p in range(0, p_max + 1):
        for q in range(0, q_max + 1):
            m = max(1, (p + q) * 3)
            try:
                eps_hat = estimate_residuals(z_train, m)
            except:
                continue
            max_lag_pq = max(p, q) if p > 0 or q > 0 else 1
            h = 1
            t_start = max_lag_pq
            t_end = n - h
            if t_end <= t_start:
                continue
            n_obs = t_end - t_start
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
            if sse <= 0:
                continue
            aic = n_obs * np.log(sse) + 2 * (1 + p + q)
            if aic < best_aic:
                best_aic = aic
                best_p = p
                best_q = q
                best_eps_hat = eps_hat
    best_models = estimate_two_phase_model(z_train, best_eps_hat, best_p, best_q, H)
    return best_p, best_q, best_models, best_eps_hat, best_aic

def export_train_results(p, q, models, filepath='train.csv'):
    rows = []
    for h, model_data in sorted(models.items()):
        theta = model_data['theta']
        sse = model_data['sse']
        aic = model_data['aic']
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

def train_model(y, p_max=10, q_max=10, H=12, m=20):
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_train = int(0.8 * n)
    y_train = y[:n_train]
    y_test = y[n_train:]
    print(f"Split Train/Test: {n_train} train (80%), {len(y_test)} test (20%)")
    d = read_d_from_adf('adf.csv')
    print(f"Orden de integración d={d} (desde adf.csv)")
    z_train = apply_differencing(y_train, d)
    print(f"Longitud de z_train después de d={d} diferenciaciones: {len(z_train)}")
    best_p, best_q, models, residuals, best_aic = grid_search_arima(z_train, p_max, q_max, H)
    print(f"Mejores parámetros encontrados: p={best_p}, q={best_q}, AIC={best_aic:.4f}")
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
    try:
        data = pd.read_csv('tseries.csv', header=None)
        y = data.values.flatten().astype(float)
        train_result = train_model(y, p_max=10, q_max=10, H=12, m=20)
        print("Entrenamiento completado.")
    except FileNotFoundError:
        print("Error: No se encontró tseries.csv")
