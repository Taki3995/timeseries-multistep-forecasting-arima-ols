import numpy as np
import pandas as pd
from utility import pinv_svd

# Tabla de valores críticos Dickey-Fuller (nivel de significancia del 5%)
CRITICAL_VALUES = {
    0.01: -3.43,
    0.05: -2.86,
    0.10: -2.57
}

def adf_test_single(y, p):
    """
    Test ADF para un orden de diferenciación específico con p rezagos.
    Calcula el estadístico ADF manualmente usando OLS con pinv_svd.
    Dy_t = c + beta*t + gamma*y_{t-1} + sum(delta_i * Dy_{t-i}) + e_t
    
    Retorna: (adf_stat, sse, aic)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    dy = np.diff(y)  # dy[t] = y[t+1] - y[t] (longitud n-1)
    
    # Elementos disponibles para la regresión desde t=p hasta n-2
    n_obs = len(dy) - p
    
    # Y vector para regresión: Dy_t (desde t=p en dy)
    Y = dy[p:n_obs+p] 
    
    # Construimos X matriz de diseño: [const, trend, y_{t-1}, Dy_{t-1}, ... Dy_{t-p}]
    X = np.zeros((n_obs, 3 + p))
    
    # Constante
    X[:, 0] = 1.0
    # Tendencia temporal
    X[:, 1] = np.arange(p + 1, n_obs + p + 1)
    # y_{t-1}
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
    k = X.shape[1]  # número de regresores
    sigma2 = sse / (n_obs - k)
    
    cov_matrix = sigma2 * (X_pinv @ X_pinv.T)
    se_gamma = np.sqrt(cov_matrix[2, 2])
    
    # Estadístico ADF
    adf_stat = gamma_hat / se_gamma
    
    # Akaike Information Criterion
    aic = n_obs * np.log(sse) + 2 * p
    
    return adf_stat, sse, aic

def adf_test_iterative(y, max_d=3, max_lag=5, alpha=0.05):
    """
    Test ADF iterativo: ciclo d=0,1,2,... hasta encontrar serie estacionaria.
    Para cada d, realiza grid search de p para minimizar AIC.
    
    Parámetros:
    - y: serie temporal
    - max_d: máximo orden de diferenciación a probar
    - max_lag: máximo número de rezagos para grid search
    - alpha: nivel de significancia (0.05 por defecto)
    
    Retorna: (d_optimal, adf_results_list) donde cada elemento es
             (d, adf_stat, critical_value, is_stationary)
    """
    y = np.asarray(y, dtype=float)
    critical_value = CRITICAL_VALUES.get(alpha, -2.86)
    
    adf_results = []
    y_diff = y.copy()
    
    for d in range(max_d + 1):
        # Grid search para encontrar mejor p
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
        
        # Determinar si la serie es estacionaria
        is_stationary = best_adf_stat < critical_value if best_adf_stat is not None else False
        
        adf_results.append({
            'd': d,
            'adf_stat': best_adf_stat,
            'critical_value': critical_value,
            'is_stationary': is_stationary,
            'p_optimal': best_p
        })
        
        # Si es estacionaria, detener el ciclo
        if is_stationary:
            return d, adf_results
        
        # Diferenciar para la siguiente iteración
        y_diff = np.diff(y_diff)
        
        # Validar que hay suficientes datos para continuar
        if len(y_diff) < max_lag + 2:
            break
    
    return max_d, adf_results

def export_adf_results(adf_results, filepath='adf.csv'):
    """
    Exporta resultados ADF a archivo CSV.
    """
    df = pd.DataFrame(adf_results)
    df.to_csv(filepath, index=False)
    print(f"Resultados ADF exportados a {filepath}")
    return df

def main(data_path='tseries.csv', output_path='adf.csv', max_d=3, max_lag=5):
    """
    Función principal: carga datos, ejecuta test ADF iterativo y exporta resultados.
    """
    # Cargar datos
    try:
        data = pd.read_csv(data_path, header=None)
        y = data.values.flatten().astype(float)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {data_path}")
        return None
    
    print(f"Datos cargados: {len(y)} observaciones")
    
    # Ejecutar test ADF iterativo
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
    
    # Exportar resultados
    df_results = export_adf_results(adf_results, output_path)
    
    return d_optimal, adf_results, y

if __name__ == '__main__':
    d_optimal, results, series = main()
