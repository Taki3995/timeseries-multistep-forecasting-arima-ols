import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import mnse, mape, recover_prediction, jarque_bera, read_d_from_adf, apply_differencing
from trn import train_model

def load_data(filepath='tseries.csv'):
    """
    Carga la serie temporal desde un archivo CSV sin encabezado.
    Retorna un vector numpy de tipo flotante.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        series = df.values.flatten().astype(float)
        return series
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}")
        return np.array([])

def predict_multi_step(train_result, d=None):
    """
    Ejecuta predicciones multi-step sobre el conjunto de TEST.
    
    Cambios clave:
    - Usa predicciones Z iterativas (sin data leakage de y_test)
    - Fija innovaciones MA a 0 para t > T (futuro inobservable)
    - Integra las predicciones z_pred a Y_pred usando recover_prediction
    
    Parámetros:
    - train_result: dict retornado por train_model()
    - d: orden de integración (si None, lo lee desde adf.csv)
    
    Retorna: dict con predicciones y métricas
    """
    models = train_result['models']
    y_train = train_result['y_train']
    y_test = train_result['y_test']
    z_train = train_result['z_train']
    residuals = train_result['residuals']
    H = train_result['H']
    n_train = train_result['n_train']
    
    # Leer d si no está especificado
    if d is None:
        d = read_d_from_adf('adf.csv')
    
    p = train_result['p']
    q = train_result['q']
    
    # Buffer de serie original Y para recuperación
    Y_buffer = y_train.tolist()
    
    # Buffer de serie diferenciada Z para predicciones iterativas
    Z_buffer = z_train.tolist()
    
    # Buffer de predicciones Z generadas
    Z_pred_buffer = []
    
    predictions = []
    
    # Para cada paso h en el horizonte
    for h in range(1, len(y_test) + 1):
        if h > H:  # Limitar a H pasos adelante
            break
        
        if h not in models:
            continue
        
        model_data = models[h]
        theta_h = model_data['theta']
        
        # Vector feature: [const, AR lags de Z, MA lags]
        x_h = np.zeros(1 + p + q)
        x_h[0] = 1.0  # Constante
        
        # AR lags: z_{t-i} para i=0..p-1
        # CRÍTICO: Usar predicciones Z iterativas para evitar data leak
        for i in range(p):
            # Índice relativo dentro del buffer
            idx_in_buffer = len(Z_buffer) + len(Z_pred_buffer) - 1 - i
            
            if idx_in_buffer >= 0 and idx_in_buffer < len(Z_buffer) + len(Z_pred_buffer):
                if idx_in_buffer < len(Z_buffer):
                    x_h[1 + i] = Z_buffer[idx_in_buffer]
                else:
                    # Usar predicción previa de Z_pred_buffer
                    x_h[1 + i] = Z_pred_buffer[idx_in_buffer - len(Z_buffer)]
        
        # MA lags: eps_{t-i} para i=0..q-1
        # CRÍTICO: Forzar a 0 si t-j > n_train (futuro inobservable)
        for j in range(q):
            # Índice temporal absoluto
            t_idx = n_train + h - 1 - j
            
            if t_idx < n_train and 0 <= t_idx < len(residuals):
                # Dentro del conjunto de entrenamiento
                x_h[1 + p + j] = residuals[t_idx]
            else:
                # Futuro (t_idx >= n_train): fijar a 0
                x_h[1 + p + j] = 0.0
        
        # Predicción en serie diferenciada: ẑ_{T+h}
        z_pred = np.dot(x_h, theta_h)
        Z_pred_buffer.append(z_pred)
        
        # Recuperación iterativa: integrar z_pred a Y_pred
        if d > 0:
            # Extraer últimos d valores de Y_buffer para recuperación
            hist_y = Y_buffer[-d:] if len(Y_buffer) >= d else Y_buffer + [0.0] * (d - len(Y_buffer))
            hist_y = hist_y[-d:]  # Asegurar exactamente d elementos
            
            # Recuperar predicción en dominio original
            y_pred = recover_prediction(z_pred, hist_y, d)
            
            # Actualizar Y_buffer para próximas iteraciones
            Y_buffer.append(y_pred)
        else:
            y_pred = z_pred
            Y_buffer.append(y_pred)
        
        # Valor real en el TEST set
        if h - 1 < len(y_test):
            y_real = y_test[h - 1]
            error = y_real - y_pred
        else:
            y_real = np.nan
            error = np.nan
        
        predictions.append({
            'h': h,
            'y_real': y_real,
            'y_pred': y_pred,
            'error': error,
            'z_pred': z_pred
        })
    
    return predictions

def evaluate_predictions(predictions, d=0):
    """
    Calcula métricas agregadas de predicción.
    """
    y_true = np.array([p['y_real'] for p in predictions if not np.isnan(p['y_real'])])
    y_pred = np.array([p['y_pred'] for p in predictions if not np.isnan(p['y_real'])])
    errors = y_true - y_pred
    
    if len(y_true) == 0:
        return None
    
    mNSE_val = mnse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    jb_stat = jarque_bera(errors)
    
    return {
        'mNSE': mNSE_val,
        'MAPE': mape_val,
        'JB': jb_stat,
        'n_predictions': len(y_true)
    }

def export_test_results(predictions, metrics, filepath='test.csv'):
    """
    Exporta predicciones y métricas del conjunto TEST a CSV.
    Formato: h, y_real, y_pred, error, mNSE, MAPE, JB
    """
    rows = []
    for pred in predictions:
        rows.append({
            'h': pred['h'],
            'y_real': pred['y_real'],
            'y_pred': pred['y_pred'],
            'error': pred['error'],
            'mNSE': metrics['mNSE'],
            'MAPE': metrics['MAPE'],
            'JB': metrics['JB']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Resultados de predicción exportados a {filepath}")
    return df

def plot_results(predictions, H):
    """
    Genera el gráfico comparativo de predicción versus valores reales.
    """
    h_vals = np.array([p['h'] for p in predictions if not np.isnan(p['y_real'])])
    y_true = np.array([p['y_real'] for p in predictions if not np.isnan(p['y_real'])])
    y_pred = np.array([p['y_pred'] for p in predictions if not np.isnan(p['y_real'])])
    
    if len(y_true) == 0:
        print("No hay predicciones válidas para graficar.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(h_vals, y_true, label='Serie Real (Test)', color='blue', marker='o', linewidth=2)
    plt.plot(h_vals, y_pred, label='Predicción Multi-step', color='red', linestyle='--', marker='x', linewidth=2)
    plt.title(f'Predicción Multi-step-ahead vs Realidad (H={H})')
    plt.xlabel('Horizonte de Predicción (h)')
    plt.ylabel('Valor de la Serie')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediccion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Gráfico comparativo guardado como 'prediccion.png'.")

def main():
    """
    Orquestador principal: integra ADF -> TRN -> TST para predicción completa.
    """
    print("="*70)
    print("PIPELINE COMPLETO: ADF -> TRN -> TST (Versión Corregida)")
    print("="*70)
    
    # Cargar datos
    y = load_data('tseries.csv')
    if len(y) == 0:
        return
    
    print(f"\n[1/3] Datos cargados: {len(y)} observaciones")
    
    # Fase 1: Entrenamiento con split 80/20 y diferenciación
    print("\n[2/3] Entrenamiento Two-Phase OLS (conjunto TRAIN 80%)...")
    train_result = train_model(y, p_max=5, q_max=5, H=12, m=20)
    
    p = train_result['p']
    q = train_result['q']
    d = train_result['d']
    print(f"  → Parámetros óptimos: p={p}, q={q}, d={d}")
    
    # Fase 2: Predicción multi-step sobre TEST (sin data leakage)
    print("\n[3/3] Predicción multi-step (conjunto TEST 20%, sin data leakage)...")
    predictions = predict_multi_step(train_result, d=d)
    print(f"  → {len(predictions)} predicciones generadas")
    
    # Evaluación
    metrics = evaluate_predictions(predictions, d=d)
    
    print("\n" + "="*70)
    print("MÉTRICAS DE EVALUACIÓN")
    print("="*70)
    print(f"Orden de integración (d): {d}")
    print(f"Número de predicciones: {metrics['n_predictions']}")
    print(f"mNSE: {metrics['mNSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"Jarque-Bera: {metrics['JB']:.4f}")
    
    # Criterio de normalidad (JB < 5.991 al 5%)
    jb_critical = 5.991
    if metrics['JB'] > jb_critical:
        print(f"  ⚠ JB > {jb_critical}: Se rechaza normalidad de residuos (α=0.05)")
    else:
        print(f"  ✓ JB ≤ {jb_critical}: No se rechaza normalidad de residuos (α=0.05)")
    
    print("="*70)
    
    # Exportar resultados
    export_test_results(predictions, metrics, 'test.csv')
    
    # Visualizar
    plot_results(predictions, train_result['H'])
    
    print("\n✓ Pipeline completado exitosamente (versión corregida).")
    print("  - adf.csv (generado)")
    print("  - train.csv (exportado)")
    print("  - test.csv (exportado)")
    print("  - prediccion.png (generado)")

if __name__ == '__main__':
    main()
