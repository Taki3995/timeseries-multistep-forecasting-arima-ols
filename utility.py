import numpy as np

def pinv_svd(A, tol=1e-15):
    """
    Calcula la pseudo-inversa de Moore-Penrose usando SVD.
    A^+ = V_h^T \cdot diag(1/S) \cdot U^T
    """
    m, n = A.shape
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    
    # Filtrar valores singulares pequeños
    S_inv = np.zeros_like(S)
    mask = S > tol
    S_inv[mask] = 1.0 / S[mask]
    
    # Construir inversa
    A_pinv = Vh.T @ np.diag(S_inv) @ U.T
    return A_pinv

def jarque_bera(x):
    """
    Calcula manualmente el estadístico de Jarque-Bera.
    """
    n = len(x)
    mu = np.mean(x)
    
    # Momentos centrales empíricos
    mu2 = np.mean((x - mu)**2)
    mu3 = np.mean((x - mu)**3)
    mu4 = np.mean((x - mu)**4)
    
    if mu2 == 0:
        return 0.0 # Caso donde todos los valores son iguales
        
    s = mu3 / (mu2**(3/2))
    k = mu4 / (mu2**2)
    
    jb = (n / 6.0) * (s**2 + ((k - 3)**2) / 4.0)
    return jb
