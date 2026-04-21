import numpy as np
import pandas as pd
def pinv_svd(A, tol=1e-15):
    m, n = A.shape
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    S_inv = np.zeros_like(S)
    mask = S > tol
    S_inv[mask] = 1.0 / S[mask]
    A_pinv = Vh.T @ np.diag(S_inv) @ U.T
    return A_pinv
def jarque_bera(x):
    n = len(x)
    mu = np.mean(x)
    mu2 = np.mean((x - mu)**2)
    mu3 = np.mean((x - mu)**3)
    mu4 = np.mean((x - mu)**4)
    if mu2 == 0:
        return 0.0
    s = mu3 / (mu2**(3/2))
    k = mu4 / (mu2**2)
    jb = (n / 6.0) * (s**2 + ((k - 3)**2) / 4.0)
    return jb
def mnse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    numerador = np.sum((y_true - y_pred)**2)
    denominador = np.sum((y_true - np.mean(y_true))**2)
    if denominador == 0:
        return float('-inf')
    return 1 - (numerador / denominador)
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
def recover_prediction(y_hat_diff, hist_Y, d):
    if d == 0:
        return y_hat_diff
    import math
    Y_pred = y_hat_diff
    for k in range(1, d + 1):
        y_val = hist_Y[d - k]
        coef_binomial = math.comb(d, k)
        signo = (-1)**k
        Y_pred -= signo * coef_binomial * y_val
    return Y_pred
def export_csv_partial(data, columns, filepath):
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    if columns:
        df = df[columns]
    df.to_csv(filepath, index=False)
    return df
def read_d_from_adf(filepath='adf.csv'):
    try:
        df = pd.read_csv(filepath)
        stationary_rows = df[df['is_stationary'] == True]
        if len(stationary_rows) > 0:
            d_optimal = int(stationary_rows.iloc[0]['d'])
            return d_optimal
        else:
            return 0
    except FileNotFoundError:
        print(f"Aviso: {filepath} no encontrado. Usando d=0 por defecto.")
        return 0
    except Exception as e:
        print(f"Error al leer {filepath}: {e}. Usando d=0 por defecto.")
        return 0
def apply_differencing(y, d):
    if d <= 0:
        return y
    z = y.copy()
    for _ in range(d):
        z = np.diff(z)
    return z
