import numpy as np
import pandas as pd
import utility

# ==========================================
# 1. TWO-PHASE OLS MATRICES
# ==========================================

def build_phase1_matrix(Z, m):
    """
    Construye las matrices para el proceso AR(m) largo.
    Retorna X (rezagos de Z) y Y (Z actual).
    """
    N = len(Z)
    X = np.zeros((N - m, m))
    for i in range(m):
        X[:, i] = Z[m - 1 - i : N - 1 - i]
    Y = Z[m:]
    return X, Y

def build_phase2_matrix(Z, residuals, m, p, q, h):
    """
    Construye las matrices para la estimación Directa Multi-Step.
    Target: Z_{t+h}
    Regresores: Z_{t}, ..., Z_{t-p+1} y eps_{t}, ..., eps_{t-q+1}
    """
    N = len(Z)
    
    # residuals[0] corresponde al tiempo t = m en la serie Z
    # Necesitamos al menos q-1 residuos pasados y p-1 valores de Z pasados
    t_start = m + max(0, q - 1)
    t_start = max(t_start, p - 1)
    
    # El tiempo t máximo que podemos usar para features es tal que t+h exista
    t_end = N - 1 - h
    
    if t_start > t_end:
        return np.array([]), np.array([]) # No hay suficientes datos
        
    X = []
    Y = []
    
    for t in range(t_start, t_end + 1):
        target = Z[t + h]
        
        ar_feats = [Z[t - i] for i in range(p)] if p > 0 else []
        ma_feats = [residuals[t - i - m] for i in range(q)] if q > 0 else []
        
        X.append(ar_feats + ma_feats)
        Y.append(target)
        
    return np.array(X), np.array(Y)

# ==========================================
# 2. MOTOR DE ENTRENAMIENTO
# ==========================================

def run_training(data, d):
    """Ejecuta el Grid Search y entrena modelos independientes por horizonte."""
    train_size = int(len(data) * 0.8)
    y_train = data[:train_size]
    
    # Serie diferenciada
    Z = utility.diff_series(y_train, d)
    N = len(Z)
    
    best_aic = np.inf
    best_p, best_q = 0, 0
    
    print("Iniciando Grid Search (p, q entre 0 y 10)... esto puede tardar un poco.")
    
    # ------------------------------------------
    # GRID SEARCH: Buscando los mejores p y q usando el modelo h=1
    # ------------------------------------------
    for p in range(11):
        for q in range(11):
            if p == 0 and q == 0:
                continue
                
            m = (p + q) * 3
            if m >= N - 5: # Prevenir desbordamiento de datos
                continue
                
            # Fase 1: Estimación de innovaciones
            X_p1, Y_p1 = build_phase1_matrix(Z, m)
            try:
                beta_p1 = np.linalg.inv(X_p1.T @ X_p1) @ X_p1.T @ Y_p1
                residuals = Y_p1 - X_p1 @ beta_p1
            except np.linalg.LinAlgError:
                continue
                
            # Fase 2: Evaluación OLS para h=1
            X_p2, Y_p2 = build_phase2_matrix(Z, residuals, m, p, q, h=1)
            if len(Y_p2) < 5:
                continue
                
            try:
                beta_p2 = np.linalg.inv(X_p2.T @ X_p2) @ X_p2.T @ Y_p2
                errors = Y_p2 - X_p2 @ beta_p2
                sse = np.sum(errors**2)
                
                aic = utility.calc_aic(sse, len(Y_p2), p + q)
                if aic < best_aic:
                    best_aic = aic
                    best_p = p
                    best_q = q
            except np.linalg.LinAlgError:
                continue
                
    print(f"[+] Grid Search completado. Óptimos: p={best_p}, q={best_q} (AIC: {best_aic:.4f})")
    
    # ------------------------------------------
    # ESTIMACIÓN FINAL: Modelos independientes para h={1, 2, 3, 4, 5}
    # ------------------------------------------
    m_opt = (best_p + best_q) * 3
    
    # Recalculamos Fase 1 con el óptimo
    X_p1, Y_p1 = build_phase1_matrix(Z, m_opt)
    beta_p1 = np.linalg.inv(X_p1.T @ X_p1) @ X_p1.T @ Y_p1
    residuals = Y_p1 - X_p1 @ beta_p1
    
    results = []
    
    # Guardar metadata
    results.extend([
        {'h': 0, 'type': 'meta', 'index': 'p', 'value': best_p},
        {'h': 0, 'type': 'meta', 'index': 'd', 'value': d},
        {'h': 0, 'type': 'meta', 'index': 'q', 'value': best_q},
        {'h': 0, 'type': 'meta', 'index': 'm', 'value': m_opt}
    ])
    
    # Iterar sobre horizontes (h)
    for h in range(1, 6):
        X_p2, Y_p2 = build_phase2_matrix(Z, residuals, m_opt, best_p, best_q, h)
        if len(Y_p2) == 0:
            print(f"[-] Advertencia: No hay suficientes datos para el horizonte h={h}")
            continue
            
        beta_h = np.linalg.inv(X_p2.T @ X_p2) @ X_p2.T @ Y_p2
        
        # Separar coeficientes AR y MA y guardarlos
        for i in range(best_p):
            results.append({'h': h, 'type': 'AR', 'index': i+1, 'value': beta_h[i]})
        for j in range(best_q):
            results.append({'h': h, 'type': 'MA', 'index': j+1, 'value': beta_h[best_p + j]})
            
    # Guardar la persistencia de los modelos
    df_train = pd.DataFrame(results)
    df_train.to_csv('train.csv', index=False)
    print("\nParámetros de los modelos guardados en 'train.csv'.")

if __name__ == '__main__':
    # 1. Cargar Data Original
    file_path = 'ts_taller2.csv' 
    df = pd.read_csv(file_path, header=None)
    data = df[0].values
    
    # 2. Cargar 'd' de la fase anterior
    try:
        df_adf = pd.read_csv('adf.csv')
        # Buscar la primera fila donde la serie sea estacionaria
        estacionarios = df_adf[df_adf['is_stationary'] == True]
        if not estacionarios.empty:
            d_opt = int(estacionarios.iloc[0]['d'])
        else:
            # Si ninguna lo fue, usamos el mayor evaluado
            d_opt = int(df_adf['d'].max()) 
    except FileNotFoundError:
        print("No se encontró adf.csv. Por favor, ejecuta adf.py primero.")
        exit()
        
    print(f"Usando orden de integración d={d_opt}")
    run_training(data, d_opt)