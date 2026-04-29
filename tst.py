import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utility

# ==========================================
# 1. FUNCIONES AUXILIARES (RECONSTRUCCIÓN)
# ==========================================

def build_phase1_matrix(Z, m):
    """Reconstruye las matrices del proceso AR(m) para Fase 1."""
    N = len(Z)
    X = np.zeros((N - m, m))
    for i in range(m):
        X[:, i] = Z[m - 1 - i : N - 1 - i]
    Y = Z[m:]
    return X, Y

def load_model_parameters(filename):
    """Carga los parámetros óptimos y coeficientes desde train.csv."""
    df = pd.read_csv(filename)
    
    # Extraer metadatos (p, d, q, m)
    meta = df[df['type'] == 'meta'].set_index('index')['value'].to_dict()
    p, d, q, m = int(meta['p']), int(meta['d']), int(meta['q']), int(meta['m'])
    
    # Extraer coeficientes para TODOS los horizontes (necesario para recuperación en cadena)
    coefs = {}
    for h in df['h'].unique():
        if h == 0: continue # El 0 guarda la metadata
        df_h = df[df['h'] == h]
        ar_vals = df_h[df_h['type'] == 'AR'].sort_values('index')['value'].values
        ma_vals = df_h[df_h['type'] == 'MA'].sort_values('index')['value'].values
        coefs[h] = {'AR': ar_vals, 'MA': ma_vals}
            
    return p, d, q, m, coefs

# ==========================================
# 2. EVALUACIÓN Y PRONÓSTICO (TESTING DIRECTO)
# ==========================================

def run_testing(file_path):
    # Cargar datos
    df_data = pd.read_csv(file_path, header=None)
    data = df_data[0].values
    
    N_total = len(data)
    train_idx = int(N_total * 0.8)
    
    # Cargar modelo entrenado
    try:
        p, d, q, m, coefs = load_model_parameters('train.csv')
    except FileNotFoundError:
        print("[-] Error: 'train.csv' no encontrado. Ejecuta trn.py primero.")
        return

    # Diferenciar toda la serie
    Z = utility.diff_series(data, d)
    
    # ---------------------------------------------------------
    # Recalcular Fase 1 (AR_m)
    # ---------------------------------------------------------
    Z_train = Z[:train_idx - d]
    X_p1_train, Y_p1_train = build_phase1_matrix(Z_train, m)
    beta_p1 = np.linalg.inv(X_p1_train.T @ X_p1_train) @ X_p1_train.T @ Y_p1_train
    
    X_p1_all, Y_p1_all = build_phase1_matrix(Z, m)
    residuals_all = Y_p1_all - X_p1_all @ beta_p1
    
    # Diccionarios para almacenar resultados por horizonte a reportar
    horizontes_reporte = [1, 3, 5]
    resultados_h = {h: {'real': [], 'pred': [], 'err': []} for h in horizontes_reporte}
    
    # Máximo horizonte que podemos predecir
    max_h = max(coefs.keys())
    
    # Definir la ventana estricta de evaluación para igualar el denominador del mNSE
    target_start = train_idx + max_h
    target_end = N_total
    
    # Iterar "día a día" asegurando que el origen permita predecir dentro de la ventana target
    for t in range(target_start - max_h, target_end - 1):
        # Índices ajustados por la diferenciación 'd'
        t_z = t - d
        
        # Historial original estricto: Todo lo que conoce el modelo hasta el día t, se mantiene FIJO
        y_hist_fijo = list(data[: t + 1])
        
        # Extraer variables explicativas (Solo información del pasado t_z)
        z_feats = [Z[t_z - i] for i in range(p)] if p > 0 else []
        eps_feats = [residuals_all[t_z - i - m] for i in range(q)] if q > 0 else []
        x_t = np.array(z_feats + eps_feats)
        
        # Cadena de predicción Multi-Step
        for h in range(1, max_h + 1):
            if h not in coefs: break
                
            weights = np.concatenate([coefs[h]['AR'], coefs[h]['MA']])
            
            # Predicción Directa en dominio diferenciado (z_t+h)
            z_pred = np.dot(x_t, weights)
            
            # Recuperación Binomial usando el historial FIJO (sin append posterior)
            y_pred_recup = utility.recover_prediction(z_pred, y_hist_fijo, d)
            
            # Índice real de la predicción
            target_T = t + h
            
            # GUARDADO RESTRINGIDO: Solo registrar si cae en la ventana estandarizada compartida
            if h in horizontes_reporte and target_start <= target_T < target_end:
                y_real = data[target_T]
                resultados_h[h]['real'].append(y_real)
                resultados_h[h]['pred'].append(y_pred_recup)
                resultados_h[h]['err'].append(y_real - y_pred_recup)
            
    # Cálculo de Métricas y Exportación
    resultados_finales = []
    
    for h in horizontes_reporte:
        y_real_list = np.array(resultados_h[h]['real'])
        y_pred_list = np.array(resultados_h[h]['pred'])
        errores_list = np.array(resultados_h[h]['err'])
        
        mnse = utility.calc_mnse(y_real_list, y_pred_list)
        mape = utility.calc_mape(y_real_list, y_pred_list)
        
        jb_stat, skew, kurt = utility.jarque_bera_test(errores_list)
        jb_critico = 5.991
        normalidad = "Rechazada (Residuos NO normales)" if jb_stat > jb_critico else "Aceptada (Residuos Normales)"
        
        print(f"\n--- Resultados Horizonte h = {h} ---")
        print(f"mNSE: {mnse * 100:.2f}%")
        print(f"MAPE: {mape * 100:.2f}%")
        print(f"Jarque-Bera (JB): {jb_stat:.4f} -> {normalidad}")
        
        resultados_finales.append({
            'h': h,
            'mNSE': mnse,
            'MAPE': mape,
            'JB_Stat': jb_stat,
            'Normalidad': normalidad
        })
        
        # Generar Gráficos Comparativos (Predicción)
        plt.figure(figsize=(10, 6))
        plt.plot(y_real_list, color='black', label='Real (t+h)')
        plt.plot(y_pred_list, color='crimson', linestyle='--', label='Predicción Directa')
        plt.suptitle(f'Test con 20% de Data: Pronóstico ARIMA({p},{d},{q})', fontweight='bold')
        plt.title(f'Test ARIMA - Horizonte h = {h}')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.xlim(left=0)
        plt.savefig(f'plot_h{h}.png')
        plt.close()
        
        # Generar Gráficos ACF (Residuos)
        utility.plot_acf(
            errores_list, 
            max_lag=20, 
            title="Sample Autocorrelation Function", 
            filename=f"acf_residuos_h{h}.png"
        )
        
    df_test = pd.DataFrame(resultados_finales)
    df_test.to_csv('test.csv', index=False)
    print("\n[+] Reporte de evaluación guardado en 'test.csv'.")
    print("[+] Gráficos generados y guardados.")

if __name__ == '__main__':
    archivo_datos = 'ts_taller2.csv'
    print(f"Iniciando evaluación sin Data Leakage en Set de Prueba (20%) para {archivo_datos}...")
    run_testing(archivo_datos)