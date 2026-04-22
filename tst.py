import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import mnse, mape, recover_prediction, jarque_bera, read_d_from_adf, apply_differencing
from trn import train_model

def load_data():
    for fname in ['ts_taller2.csv', 'tseries.csv']:
        try:
            df = pd.read_csv(fname, header=None)
            series = df.values.flatten().astype(float)
            return series
        except FileNotFoundError:
            continue
    print("Error: No se encontró ningún archivo de serie de tiempo válido.")
    return np.array([])

def build_features_rolling(y_hist, eps_hist, p, q):
    x = np.zeros(1 + p + q)
    x[0] = 1.0
    for i in range(p):
        idx = len(y_hist) - 1 - i
        if idx >= 0:
            x[1 + i] = y_hist[idx]
    for j in range(q):
        idx = len(eps_hist) - 1 - j
        if idx >= 0:
            x[1 + p + j] = eps_hist[idx]
        else:
            x[1 + p + j] = 0.0
    return x

def rolling_forecast_metrics(train_result, d):
    models = train_result['models']
    y_train = train_result['y_train']
    y_test = train_result['y_test']
    z_train = train_result['z_train']
    p = train_result['p']
    q = train_result['q']
    results = []
    for h in [1, 3, 5]:
        if h not in models:
            continue
        theta_h = models[h]['theta']
        residuals_train = models[h]['residuals']
        theta_m = models[h]['theta_m']
        y_pred_h = []
        y_real_h = []
        max_t = len(y_test) - h
        for t in range(max_t):
            y_hist = list(y_train) + list(y_test[:t])
            if d > 0:
                z_hist = apply_differencing(np.array(y_hist, dtype=float), d).tolist()
            else:
                z_hist = list(y_hist)
            # Calcular innovación MA dinámica para el paso actual
            if q > 0:
                m = len(theta_m) - 1
                if len(z_hist) >= m:
                    X_ar = np.zeros(m + 1)
                    X_ar[0] = 1.0
                    for i in range(1, m + 1):
                        X_ar[i] = z_hist[-i]
                    innov = z_hist[-1] - np.dot(X_ar, theta_m)
                else:
                    innov = 0.0
                eps_hist = [innov]
            else:
                eps_hist = []
            x = build_features_rolling(z_hist, eps_hist, p, q)
            z_pred = np.dot(x, theta_h)
            if d > 0:
                hist_y = y_hist[-d:] if len(y_hist) >= d else y_hist + [0.0] * (d - len(y_hist))
                hist_y = hist_y[-d:]
                y_pred = recover_prediction(z_pred, hist_y, d)
            else:
                y_pred = z_pred
            y_real = y_test[t + h - 1]
            y_pred_h.append(y_pred)
            y_real_h.append(y_real)
        y_pred_h = np.array(y_pred_h, dtype=float)
        y_real_h = np.array(y_real_h, dtype=float)
        mNSE_h = mnse(y_real_h, y_pred_h)
        MAPE_h = mape(y_real_h, y_pred_h)
        jb_h = jarque_bera(residuals_train)
        results.append({
            'h': h,
            'mNSE_h': mNSE_h,
            'MAPE_h': MAPE_h,
            'JB_train_h': jb_h,
            'y_pred': y_pred_h,
            'y_real': y_real_h
        })
    return results

def export_test_results(results, filepath='test.csv'):
    rows = []
    for r in results:
        rows.append({
            'h': r['h'],
            'mNSE_h': r['mNSE_h'],
            'MAPE_h': r['MAPE_h'],
            'JB_train_h': r['JB_train_h']
        })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Resultados de predicción exportados a {filepath}")
    return df

def plot_results(results):
    if len(results) == 0:
        print("No hay predicciones válidas para graficar.")
        return
    h_vals = [r['h'] for r in results]
    plt.figure(figsize=(12, 6))
    for r in results:
        if len(r['y_real']) == 0:
            continue
        plt.plot(r['y_real'], label=f"Real h={r['h']}", linewidth=1)
        plt.plot(r['y_pred'], label=f"Pred h={r['h']}", linestyle='--', linewidth=1)
    plt.title('Rolling Forecast - Predicción vs Realidad')
    plt.xlabel('Índice de tiempo en TEST')
    plt.ylabel('Valor de la Serie')
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig('prediccion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráfico comparativo guardado como 'prediccion.png'.")

def main():
    print("="*70)
    print("PIPELINE COMPLETO: ADF -> TRN -> TST (Rolling Forecast)")
    print("="*70)
    y = load_data()
    if len(y) == 0:
        return
    print(f"\n[1/3] Datos cargados: {len(y)} observaciones")
    print("\n[2/3] Entrenamiento Two-Phase OLS (conjunto TRAIN 80%)...")
    train_result = train_model(y, p_max=10, q_max=10)
    d = train_result['d']
    print(f"  → Parámetros óptimos: p={train_result['p']}, q={train_result['q']}, d={d}")
    print("\n[3/3] Rolling Forecast sobre TEST (sin data leakage)...")
    results = rolling_forecast_metrics(train_result, d)
    print(f"  → Métricas por horizonte: {len(results)}")
    print("\n" + "="*70)
    print("MÉTRICAS POR HORIZONTE")
    print("="*70)
    for r in results:
        jb_text = f"JB_train={r['JB_train_h']:.4f}" if not np.isnan(r['JB_train_h']) else "JB_train=NA"
        print(f"h={r['h']} | mNSE_h={r['mNSE_h']:.4f} | MAPE_h={r['MAPE_h']:.2f}% | {jb_text}")
    jb_critical = 5.991
    for h in [1, 3, 5]:
        row = next((r for r in results if r['h'] == h), None)
        if row is None or np.isnan(row['JB_train_h']):
            continue
        if row['JB_train_h'] > jb_critical:
            print(f"h={h} -> JB_train > {jb_critical}: Se rechaza normalidad")
        else:
            print(f"h={h} -> JB_train ≤ {jb_critical}: No se rechaza normalidad")
    print("="*70)
    export_test_results(results, 'test.csv')
    plot_results(results)
    print("\n✓ Pipeline completado exitosamente (rolling forecast).")
    print("  - adf.csv (generado)")
    print("  - train.csv (exportado)")
    print("  - test.csv (exportado)"
          )
    print("  - prediccion.png (generado)")

if __name__ == '__main__':
    main()
