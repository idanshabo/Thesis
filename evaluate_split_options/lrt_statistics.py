import torch
import math

def add_jitter(matrix, jitter=1e-6):
    """Adds a small diagonal regularization to ensure positive definiteness."""
    d = matrix.shape[0]
    return matrix + torch.eye(d, device=matrix.device, dtype=matrix.dtype) * jitter

def robust_cholesky(matrix, max_retries=8, initial_jitter=1e-6):
    """
    Attempts Cholesky decomposition with progressively increasing jitter.
    Falls back to Eigen-decomposition if it still fails (e.g., when p >= n).
    """
    for i in range(max_retries):
        jitter = initial_jitter * (10 ** i)
        try:
            return torch.linalg.cholesky(add_jitter(matrix, jitter=jitter))
        except torch._C._LinAlgError:
            continue
            
    # Absolute fallback: Eigen-decomposition to force positive-definiteness
    print("      [WARNING] Cholesky failed. Forcing PSD via Eigen-decomposition.")
    evals, evecs = torch.linalg.eigh(matrix)
    evals = torch.clamp(evals, min=1e-6) # Force all eigenvalues to be strictly > 0
    return evecs @ torch.diag(torch.sqrt(evals))

def compute_gls_operators(U):
    """
    Precomputes the inverse and the projection operator for a given U matrix.
    S_i = X_i^T P_i X_i (where P_i handles the GLS mean centering automatically).
    """
    n = U.shape[0]
    
    # 1. Force Perfect Symmetry First
    U_sym = (U + U.T) / 2.0
    
    # 2. Robust Cholesky Decomposition
    L = robust_cholesky(U_sym)
        
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

def compute_mean_lrt(X, U_inv, group_a_indices, group_b_indices):
    """
    LRT for H0: mu_A = mu_B (shared mean) vs H1: mu_A != mu_B (separate means),
    with V shared under both hypotheses.

    Lambda = n * (log|V_0| - log|V_1|)

    where V_0 = X^T P_0 X / n  (P_0 projects out shared GLS mean)
          V_1 = X^T P_1 X / n  (P_1 projects out group-specific GLS means)

    Args:
        X: (n, p) data matrix (already in pPCA space)
        U_inv: (n, n) inverse of phylogenetic covariance
        group_a_indices: list of row indices for group A
        group_b_indices: list of row indices for group B

    Returns:
        Lambda: float, the LRT statistic (chi-squared distributed under H0)
    """
    n, p = X.shape
    dtype = X.dtype
    device = X.device

    # H0: shared mean -> design matrix is just a column of ones
    ones = torch.ones((n, 1), dtype=dtype, device=device)
    denom_0 = ones.T @ U_inv @ ones  # scalar (1x1)
    P_0 = U_inv - (U_inv @ ones @ torch.linalg.pinv(denom_0) @ ones.T @ U_inv)
    P_0 = (P_0 + P_0.T) / 2.0

    # H1: separate means -> design matrix Z is n x 2 indicator
    Z = torch.zeros((n, 2), dtype=dtype, device=device)
    Z[group_a_indices, 0] = 1.0
    Z[group_b_indices, 1] = 1.0

    ZtUiZ = Z.T @ U_inv @ Z  # 2x2
    ZtUiZ_inv = torch.linalg.pinv(ZtUiZ)
    P_1 = U_inv - (U_inv @ Z @ ZtUiZ_inv @ Z.T @ U_inv)
    P_1 = (P_1 + P_1.T) / 2.0

    # Scatter matrices
    S_0 = X.T @ P_0 @ X
    S_1 = X.T @ P_1 @ X

    # MLEs for V
    V_0 = (S_0 + S_0.T) / (2.0 * n)
    V_1 = (S_1 + S_1.T) / (2.0 * n)

    logdet_V0 = torch.linalg.slogdet(add_jitter(V_0))[1]
    logdet_V1 = torch.linalg.slogdet(add_jitter(V_1))[1]

    Lambda = n * (logdet_V0 - logdet_V1)
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


def simulate_null_data_two_means(n, p, mu_A, mu_B, group_a_indices, group_b_indices, L_U, L_V):
    """
    Simulates X ~ MN(M, U, V) where M has mu_A for group A rows and mu_B for group B rows.
    Used for power simulations of the mean LRT.
    """
    Z = torch.randn(n, p, dtype=L_U.dtype, device=L_U.device)
    M = torch.zeros((n, p), dtype=L_U.dtype, device=L_U.device)
    M[group_a_indices] = mu_A.unsqueeze(0)
    M[group_b_indices] = mu_B.unsqueeze(0)
    X_sim = M + (L_U @ Z @ L_V.T)
    return X_sim
