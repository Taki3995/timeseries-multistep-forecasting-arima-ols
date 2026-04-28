import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utility

# ==========================================
# 1. VALORES CRÍTICOS DE MACKINNON (5%)
# ==========================================

def mackinnon_cv(case, T):
    """
    Calcula el valor crítico de MacKinnon al 5% (alpha=0.05).
    Se utilizan los coeficientes asintóticos estándar para los 3 casos.
    """
    if case == 1:
        # Caso 1: Sin constante ni tendencia
        phi_inf, phi_1, phi_2 = -1.9410, -0.2686, -3.365
    elif case == 2:
        # Caso 2: Con constante (drift)
        phi_inf, phi_1, phi_2 = -2.8621, -2.7380, -8.360
    else: 
        # Caso 3: Con constante y tendencia
        phi_inf, phi_1, phi_2 = -3.4126, -4.0390, -17.830
        
    return phi_inf + (phi_1 / T) + (phi_2 / (T**2))

# ==========================================
# 2. CONSTRUCCIÓN DEL MODELO OLS PARA ADF
# ==========================================

def get_gamma_index(case):
    """Retorna el índice en el vector beta donde se encuentra gamma."""
    if case == 1: return 0  # gamma
    if case == 2: return 1  # alpha, gamma
    return 2                # alpha, beta, gamma

def build_adf_matrix(S, dS, p_max, p, case, M):
    """
    Construye la matriz de diseño X dependiendo del caso determinístico.
    M: Tamaño efectivo de la muestra (N - 1 - p_max) para que el AIC 
       sea comparable entre distintos valores de p.
    """
    cols = []
    
    if case in [2, 3]:
        cols.append(np.ones(M)) # Término constante (alpha)
    if case == 3:
        cols.append(np.arange(1, M + 1)) # Tendencia determinística (beta * t)
        
    cols.append(S[p_max : p_max + M]) # Rezago original: gamma * y_{t-1}
    
    # Rezagos de las diferencias: delta_i * dS_{t-i}
    for i in range(1, p + 1):
        cols.append(dS[p_max - i : p_max + M - i]) 
        
    return np.column_stack(cols)

# ==========================================
# 3. MOTOR PRINCIPAL TEST ADF
# ==========================================

def run_adf_test(data):
    """
    Ejecuta el test iterando el orden de integración d.
    Para cada d, evalúa los 3 casos e itera p para minimizar el AIC.
    """
    # 80% Entrenamiento según requerimientos
    train_size = int(len(data) * 0.8)
    y_train = data[:train_size]
    
    results = []
    d = 0
    is_stationary = False
    
    # Bucle de diferenciación: seguimos hasta encontrar estacionariedad
    while not is_stationary and d < 4: 
        # 1. Preparar las series diferenciadas
        S = utility.diff_series(y_train, d)
        N = len(S)
        p_max = utility.schwert_rule(N)
        dS = np.diff(S)
        
        # Fijar el vector Y para que todos los modelos tengan el mismo 'N' en su AIC
        M = N - 1 - p_max 
        Y = dS[p_max : p_max + M]
        
        d_stationary = False
        
        # 2. Evaluar los 3 Casos Determinísticos
        for case in [1, 2, 3]:
            best_aic = np.inf
            best_p = 0
            best_t_adf = 0
            
            # 3. Grid Search iterando rezagos (p) para minimizar AIC
            for p in range(p_max + 1):
                X = build_adf_matrix(S, dS, p_max, p, case, M)
                
                # Regresión OLS manual usando numpy
                try:
                    XTX_inv = np.linalg.inv(X.T @ X)
                except np.linalg.LinAlgError:
                    continue # Omite iteración si la matriz es singular
                    
                beta = XTX_inv @ X.T @ Y
                residuals = Y - X @ beta
                sse = np.sum(residuals**2)
                k = X.shape[1]
                
                aic = utility.calc_aic(sse, M, k)
                
                if aic < best_aic:
                    best_aic = aic
                    best_p = p
                    
                    # Cálculo del Error Estándar (SE) para gamma
                    sigma2 = sse / (M - k)
                    cov_beta = sigma2 * XTX_inv
                    gamma_idx = get_gamma_index(case)
                    gamma = beta[gamma_idx]
                    se_gamma = np.sqrt(cov_beta[gamma_idx, gamma_idx])
                    
                    # Estadístico t_ADF
                    best_t_adf = gamma / se_gamma
            
            # Evaluar significancia al 5%
            crit_val = mackinnon_cv(case, M)
            # El test es de cola izquierda: rechazamos H0 si t_ADF es menor al valor crítico
            stationary_case = bool(best_t_adf < crit_val)
            
            if stationary_case:
                d_stationary = True
                
            results.append({
                'd': d,
                'case': case,
                'opt_p': best_p,
                'aic': best_aic,
                't_adf': best_t_adf,
                'crit_val': crit_val,
                'is_stationary': stationary_case
            })
            
        if d_stationary:
            is_stationary = True
            print(f"\n[+] ÉXITO: La serie es estacionaria con orden de integración d = {d}.")
        else:
            print(f"[-] Nivel d={d} evaluado: No estacionaria. Diferenciando...")
            d += 1
            
    # Guardar en CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('adf.csv', index=False)
    
    # Retorna el d_optimo (el que frenó el bucle)
    return df_results, d if is_stationary else d-1

if __name__ == '__main__':
    # Cambia el archivo aquí según el que quieras evaluar
    file_path = 'ts_taller2.csv' 
    
    print(f"Cargando dataset: {file_path}")
    df = pd.read_csv(file_path, header=None)
    data = df[0].values
    
    # ==========================================
    # NUEVO: GENERACIÓN DE GRÁFICOS INICIALES
    # ==========================================
    
    # 1. Gráfico de la Serie de Tiempo Original
    plt.figure(figsize=(10, 6))
    plt.plot(data, color='steelblue', linewidth=1.2)
    plt.title('Time Series', fontweight='bold')
    plt.xlabel('time')
    plt.ylabel('mm')
    plt.grid(True, alpha=0.5)
    plt.xlim(0, len(data))
    plt.savefig('time_series.png')
    plt.close()
    print("[+] Gráfico 'time_series.png' generado.")
    
    # 2. Gráfico ACF Inicial (Serie original)
    utility.plot_acf(data, max_lag=20, title="Sample Autocorrelation Function", filename="acf_initial.png")
    print("[+] Gráfico 'acf_initial.png' generado.")
    
    # ==========================================
    
    print(f"Total datos: {len(data)}. Usando 80% para Training: {int(len(data)*0.8)}")
    res_df, d_opt = run_adf_test(data)
    
    print(f"\nResultados consolidados guardados en 'adf.csv':")
    print(res_df.to_string(index=False))