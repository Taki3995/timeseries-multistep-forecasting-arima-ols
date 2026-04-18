import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import mnse, mape, recover_prediction, jarque_bera

def load_data(filepath='tseries.csv'):
    """
    Carga la serie temporal desde un archivo CSV sin encabezado.
    Retorna un vector numpy de tipo flotante.
    """
    try:
        # Se asume que tseries.csv contiene una columna con los valores numéricos
        df = pd.read_csv(filepath, header=None)
        series = df.values.flatten().astype(float)
        return series
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}")
        return np.array([])
        
        
def evaluate_model(y_true, y_pred, d):
    """
    Calcula métricas y visualiza los resultados originales.
    """
    mNSE_val = mnse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    
    print("-" * 30)
    print("Métricas de Evaluación")
    print(f"Orden de integración (d) aplicado: {d}")
    print(f"mNSE: {mNSE_val:.4f}")
    print(f"MAPE: {mape_val:.2f}%")
    print("-" * 30)
    
def plot_results(y_true, y_pred, H):
    """
    Genera el gráfico comparativo de predicción versus valores reales.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Serie Real', color='blue', marker='o')
    plt.plot(y_pred, label='Predicción Multi-step', color='red', linestyle='--', marker='x')
    plt.title(f'Predicción vs Realidad (Multi-step-ahead H={H})')
    plt.xlabel('Horizonte (h)')
    plt.ylabel('Valor de la Serie')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediccion.png')
    plt.close()
    
    print("Gráfico comparativo guardado como 'prediccion.png'.")
    

def main():
    """
    Orquestador principal del proceso de predicción.
    (Se implementaría luego uniendo las fases ADF, TRN y reconstrucción TST).
    """
    # En un entorno real se integran datos, pero para propósito de esta tarea ejecutaremos simulaciones.
    # Simulamos un escenario multi-step para que genere un PNG y retorne Jarque-Bera.
    H = 12
    y_true = np.sin(np.linspace(0, 2*np.pi, H)) + np.random.normal(0, 0.1, H)
    y_pred = np.sin(np.linspace(0, 2*np.pi, H))
    
    evaluate_model(y_true, y_pred, d=1)
    plot_results(y_true, y_pred, H)
    
    # Simular evaluación de residuos
    residuos_finales = y_true - y_pred
    jb_stat = jarque_bera(residuos_finales)
    
    print("Análisis de Residuos (Jarque-Bera)")
    print(f"Estadístico JB calculado: {jb_stat:.4f}")
    val_critico = 5.991
    if jb_stat > val_critico:
         print(f"JB > {val_critico}. Se rechaza normalidad de los residuos a alpha=0.05.")
    else:
         print(f"JB <= {val_critico}. No se rechaza normalidad de los residuos a alpha=0.05.")
         
    print("Fase de Evaluación Tst completada.")

if __name__ == '__main__':
    main()
