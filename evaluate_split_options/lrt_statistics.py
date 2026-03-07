import torch
import math

def add_jitter(matrix, jitter=1e-6):
    """Adds a small diagonal regularization to ensure positive definiteness."""
    d = matrix.shape[0]
    return matrix + torch.eye(d, device=matrix.device, dtype=matrix.dtype) * jitter

def compute_gls_operators(U):
    """
    Precomputes the inverse and the projection operator for a given U matrix.
    S_i = X_i^T P_i X_i (where P_i handles the GLS mean centering automatically).
    """
    n = U.shape[0]
    
    # 1. Force Perfect Symmetry First
    U_sym = (U + U.T) / 2.0
    
    # 2. Adaptive Jitter for Robust Cholesky Decomposition
    jitter_init = 1e-6
    max_retries = 5
    L = None
    
    for i in range(max_retries):
        jitter = jitter_init * (10 ** i)  # Scales: 1e-6, 1e-5, 1e-4...
        U_reg = add_jitter(U_sym, jitter=jitter)
        
        try:
            L = torch.linalg.cholesky(U_reg)
            break  # Success! Exit the loop.
        except torch._C._LinAlgError:
            continue # Matrix still not positive-definite, try higher jitter
            
    if L is None:
        raise RuntimeError(f"Cholesky factorization failed for matrix of size {n}x{n} even with max jitter {jitter}.")
        
    U_inv = torch.cholesky_inverse(L)
    
    ones = torch.ones((n, 1), dtype=U.dtype, device=U.device)
    
    # mu_hat = (1^T U^-1 1)^-1 * 1^T U^-1 X
    term1 = torch.linalg.pinv(ones.T @ U_inv @ ones) # Scalar/1x1 matrix
    term2 = ones.T @ U_inv
    
    # Projection matrix P such that X^T P X = S (the scatter matrix)
    # P = U^-1 - U^-1 1 (1^T U^-1 1)^-1 1^T U^-1
    P = U_inv - (term2.T @ term1 @ term2)
    
    # Force symmetry on the final projection operator to be safe
    P = (P + P.T) / 2.0
    
    return U_inv, P, term1, term2

def compute_mle_and_lrt(X_A, X_B, P_A, P_B, n_A, n_B, return_matrices=False):
    n = n_A + n_B
    
    # Scatter matrices
    S_A = X_A.T @ P_A @ X_A
    S_B = X_B.T @ P_B @ X_B
    
    # MLEs
    V_A = S_A / n_A
    V_B = S_B / n_B
    V_global = (S_A + S_B) / n
    
    # Force symmetry before Cholesky/slogdet (Crucial fix!)
    V_global = (V_global + V_global.T) / 2.0
    V_A = (V_A + V_A.T) / 2.0
    V_B = (V_B + V_B.T) / 2.0
    
    logdet_V = torch.linalg.slogdet(add_jitter(V_global))[1]
    logdet_V_A = torch.linalg.slogdet(add_jitter(V_A))[1]
    logdet_V_B = torch.linalg.slogdet(add_jitter(V_B))[1]
    
    Lambda = n * logdet_V - n_A * logdet_V_A - n_B * logdet_V_B
    
    if return_matrices:
        return Lambda, V_A, V_B
    return Lambda

def simulate_null_data(n, p, mu_hat, L_U, L_V):
    """
    Simulates X^(c) ~ MN(1*mu_hat^T, U, V_hat)
    """
    # Z ~ N(0,1)
    Z = torch.randn(n, p, dtype=L_U.dtype, device=L_U.device)
    ones = torch.ones((n, 1), dtype=L_U.dtype, device=L_U.device)
    
    # X = M + L_U * Z * L_V^T
    X_sim = (ones @ mu_hat.reshape(1, -1)) + (L_U @ Z @ L_V.T)
    return X_sim
