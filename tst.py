import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import mnse, mape, recover_prediction

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
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Serie Real', color='blue', marker='o')
    plt.plot(y_pred, label='Predicción Multi-step', color='red', linestyle='--', marker='x')
    plt.title(f'Predicción vs Realidad (Multi-step-ahead H={len(y_pred)})')
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
    print("Fase de Evaluación Tst preparada.")

if __name__ == '__main__':
    main()
