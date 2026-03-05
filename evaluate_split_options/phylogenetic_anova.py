import torch

def compute_phylogenetic_transformation(C):
    """
    Computes the phylogenetic transformation matrix P.
    Uses robust 64-bit Eigen decomposition and dynamically drops zero-eigenvalues.
    """
    # Force 64-bit precision to prevent overflow
    C_double = C.double()
    L, V = torch.linalg.eigh(C_double)
    
    # Dynamic tolerance: drop any eigenvalue smaller than 0.001% of the max
    tol = 1e-5 * L.max() 
    mask = L > tol
    
    L_pos = L[mask]
    V_pos = V[:, mask]
    
    # P = Lambda^{-1/2} * V^T
    L_inv_sqrt = torch.diag(1.0 / torch.sqrt(L_pos))
    P = L_inv_sqrt @ V_pos.T
    
    return P

def get_sscp_trace(X, Y):
    """
    Computes the trace of the residual Sums of Squares and Cross-Products (SSCP) matrix 
    from the Ordinary Least Squares (OLS) of Y on X.
    Assumes X and Y are already phylogenetically transformed.
    """
    # Use 'gelsd' for maximum stability on ill-conditioned matrices
    sol = torch.linalg.lstsq(X, Y, driver='gelsd').solution
    residuals = Y - X @ sol
    return torch.sum(residuals ** 2)
    
def phylogenetic_anova_rrpp(Y, C, group_a_indices, group_b_indices, n_permutations=999):
    """
    Performs Phylogenetic ANOVA using Randomized Residual Permutation Procedure (RRPP).
    Tests H0: \\mu_A = \\mu_B.
    
    Args:
        Y: torch.Tensor of shape (N, p) containing embeddings.
        C: torch.Tensor of shape (N, N) containing the phylogenetic covariance matrix.
        group_a_indices: list/tensor of row indices for Group A.
        group_b_indices: list/tensor of row indices for Group B.
        n_permutations: int, number of bootstrap iterations.
        
    Returns:
        F_obs: float, the observed F-statistic.
        p_val: float, the empirical p-value.
    """
    # Enforce 64-bit precision for the entire statistical engine
    Y = Y.double()
    C = C.double()
    
    N, p = Y.shape
    device = Y.device
    
    X_0 = torch.ones((N, 1), device=device, dtype=torch.float64)
    X_F = torch.zeros((N, 2), device=device, dtype=torch.float64)
    X_F[:, 0] = 1.0
    X_F[group_a_indices, 1] = 1.0
    
    P = compute_phylogenetic_transformation(C)
    PX_0 = P @ X_0
    PX_F = P @ X_F
    PY = P @ Y
    
    sol_0 = torch.linalg.lstsq(PX_0, PY, driver='gelsd').solution
    Y_hat_0 = X_0 @ sol_0  
    E_0 = Y - Y_hat_0      
    
    ss_resid_F = get_sscp_trace(PX_F, PY)
    ss_resid_0 = get_sscp_trace(PX_0, PY)
    
    # Clamp to prevent negative sums of squares from floating point noise
    ss_model = torch.clamp(ss_resid_0 - ss_resid_F, min=0.0)
    ss_resid_F_safe = torch.clamp(ss_resid_F, min=1e-10)
    
    df_model = 1.0 
    df_resid = max(1.0, N - 2.0)
    
    F_obs = (ss_model / df_model) / (ss_resid_F_safe / df_resid)
    if torch.isnan(F_obs): F_obs = torch.tensor(0.0, dtype=torch.float64)
    
    F_null = torch.zeros(n_permutations, device=device, dtype=torch.float64)
    PY_hat_0 = P @ Y_hat_0
    
    for i in range(n_permutations):
        perm_idx = torch.randperm(N, device=device)
        E_star = E_0[perm_idx]
        PY_star = PY_hat_0 + (P @ E_star)
        
        ss_resid_F_star = get_sscp_trace(PX_F, PY_star)
        ss_resid_0_star = get_sscp_trace(PX_0, PY_star)
        
        ss_model_star = torch.clamp(ss_resid_0_star - ss_resid_F_star, min=0.0)
        ss_resid_F_star_safe = torch.clamp(ss_resid_F_star, min=1e-10)
        
        F_star = (ss_model_star / df_model) / (ss_resid_F_star_safe / df_resid)
        F_null[i] = F_star
        
    F_null = torch.nan_to_num(F_null, nan=0.0)
    count_exceed = torch.sum(F_null >= F_obs).item()
    p_val = (count_exceed + 1) / (n_permutations + 1)
    
    return F_obs.item(), p_val
