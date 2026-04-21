import numpy as np
import pandas as pd
from utility import pinv_svd
CRITICAL_VALUES = {
    0.01: -3.43,
    0.05: -2.86,
    0.10: -2.57
}
def adf_test_single(y, p):
    y = np.asarray(y, dtype=float)
    n = len(y)
    dy = np.diff(y)
    n_obs = len(dy) - p
    Y = dy[p:n_obs+p]
    X = np.zeros((n_obs, 2 + p))
    X[:, 0] = 1.0
    X[:, 1] = y[p:n_obs+p]
    for i in range(1, p + 1):
        X[:, 1 + i] = dy[p - i : n_obs + p - i]
    X_pinv = pinv_svd(X)
    theta_hat = X_pinv @ Y
    gamma_hat = theta_hat[1]
    Y_pred = X @ theta_hat
    residuals = Y - Y_pred
    sse = np.sum(residuals**2)
    k = X.shape[1]
    sigma2 = sse / (n_obs - k)
    cov_matrix = sigma2 * (X_pinv @ X_pinv.T)
    se_gamma = np.sqrt(cov_matrix[1, 1])
    adf_stat = gamma_hat / se_gamma
    aic = n_obs * np.log(sse) + 2 * p
    return adf_stat, sse, aic
def adf_test_iterative(y, max_d=3, max_lag=5, alpha=0.05):
    y = np.asarray(y, dtype=float)
    critical_value = CRITICAL_VALUES.get(alpha, -2.86)
    adf_results = []
    y_diff = y.copy()
    for d in range(max_d + 1):
        best_aic = float('inf')
        best_adf_stat = None
        best_p = 0
        for p in range(max_lag + 1):
            try:
                adf_stat, sse, aic = adf_test_single(y_diff, p)
                if aic < best_aic:
                    best_aic = aic
                    best_adf_stat = adf_stat
                    best_p = p
            except:
                continue
        is_stationary = best_adf_stat < critical_value if best_adf_stat is not None else False
        adf_results.append({
            'd': d,
            'adf_stat': best_adf_stat,
            'critical_value': critical_value,
            'is_stationary': is_stationary,
            'p_optimal': best_p
        })
        if is_stationary:
            return d, adf_results
        y_diff = np.diff(y_diff)
        if len(y_diff) < max_lag + 2:
            break
    return max_d, adf_results
def export_adf_results(adf_results, filepath='adf.csv'):
    df = pd.DataFrame(adf_results)
    df.to_csv(filepath, index=False)
    print(f"Resultados ADF exportados a {filepath}")
    return df
def main(data_path='tseries.csv', output_path='adf.csv', max_d=3, max_lag=5):
    try:
        data = pd.read_csv(data_path, header=None)
        y = data.values.flatten().astype(float)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {data_path}")
        return None
    print(f"Datos cargados: {len(y)} observaciones")
    d_optimal, adf_results = adf_test_iterative(y, max_d=max_d, max_lag=max_lag, alpha=0.05)
    print(f"\nResultados Test ADF Iterativo:")
    print("=" * 60)
    for result in adf_results:
        print(f"d={result['d']}: τ̂={result['adf_stat']:.4f}, "
              f"valor_crítico={result['critical_value']:.4f}, "
              f"estacionario={result['is_stationary']}, "
              f"p_optimal={result['p_optimal']}")
    print(f"\nOrden de integración óptimo: d={d_optimal}")
    print("=" * 60)
    df_results = export_adf_results(adf_results, output_path)
    return d_optimal, adf_results, y
if __name__ == '__main__':
    d_optimal, results, series = main()
