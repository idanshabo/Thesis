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
    U_reg = add_jitter(U)
    
    # Use Cholesky for stable inversion
    L = torch.linalg.cholesky(U_reg)
    U_inv = torch.cholesky_inverse(L)
    
    ones = torch.ones((n, 1), dtype=U.dtype, device=U.device)
    
    # mu_hat = (1^T U^-1 1)^-1 * 1^T U^-1 X
    term1 = torch.inverse(ones.T @ U_inv @ ones) # Scalar/1x1 matrix
    term2 = ones.T @ U_inv
    
    # Projection matrix P such that X^T P X = S (the scatter matrix)
    # P = U^-1 - U^-1 1 (1^T U^-1 1)^-1 1^T U^-1
    P = U_inv - (term2.T @ term1 @ term2)
    
    return U_inv, P, term1, term2

def compute_mle_and_lrt(X_A, X_B, P_A, P_B, n_A, n_B):
    """
    Computes the LRT statistic Lambda for a single split.
    """
    n = n_A + n_B
    
    # Scatter matrices
    S_A = X_A.T @ P_A @ X_A
    S_B = X_B.T @ P_B @ X_B
    
    # MLEs
    V_A = S_A / n_A
    V_B = S_B / n_B
    V_global = (S_A + S_B) / n
    
    # Log determinants (using your existing robust get_log_det logic conceptually)
    # Adding small jitter to V matrices to prevent log(0) during bootstrap
    logdet_V = torch.linalg.slogdet(add_jitter(V_global))[1]
    logdet_V_A = torch.linalg.slogdet(add_jitter(V_A))[1]
    logdet_V_B = torch.linalg.slogdet(add_jitter(V_B))[1]
    
    # Lambda = n*log|V| - n_A*log|V_A| - n_B*log|V_B|
    Lambda = n * logdet_V - n_A * logdet_V_A - n_B * logdet_V_B
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
