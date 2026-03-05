import torch

def compute_phylogenetic_transformation(C, tol=1e-6):
    """
    Computes the phylogenetic transformation matrix P.
    Uses Eigen decomposition and drops zero-eigenvalues to handle singular matrices
    (which naturally occur after shifting the covariance matrix to the local root).
    """
    # Eigen decomposition: C = V * L * V^T
    L, V = torch.linalg.eigh(C)
    
    # Only keep eigenvalues significantly greater than 0
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
    # Solve for beta: X * beta = Y
    solution = torch.linalg.lstsq(X, Y)
    
    # Calculate residuals
    residuals = Y - X @ solution.solution
    
    # The trace of the SSCP matrix (R^T R) is the sum of all squared residuals
    return torch.sum(residuals ** 2)

def phylogenetic_anova_rrpp(Y, C, group_a_indices, group_b_indices, n_permutations=999):
    """
    Performs Phylogenetic ANOVA using Randomized Residual Permutation Procedure (RRPP).
    Tests H0: \mu_A = \mu_B.
    
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
    N, p = Y.shape
    device = Y.device
    
    # 1. Setup Design Matrices
    # Null model X_0 only has an intercept
    X_0 = torch.ones((N, 1), device=device, dtype=Y.dtype)
    
    # Full model X_F has an intercept and a group indicator
    X_F = torch.zeros((N, 2), device=device, dtype=Y.dtype)
    X_F[:, 0] = 1.0
    X_F[group_a_indices, 1] = 1.0  # Group A is 1, Group B is 0
    
    # 2. Get Transformation Matrix P
    P = compute_phylogenetic_transformation(C)
    
    # Transform design matrices and data
    PX_0 = P @ X_0
    PX_F = P @ X_F
    PY = P @ Y
    
    # 3. Fit Null Model (to get residuals for RRPP)
    # Solving OLS on transformed data yields the GLS coefficients
    sol_0 = torch.linalg.lstsq(PX_0, PY).solution
    Y_hat_0 = X_0 @ sol_0  # Predicted values in the ORIGINAL space
    E_0 = Y - Y_hat_0      # Residuals in the ORIGINAL space
    
    # 4. Calculate Observed F-statistic
    ss_resid_F = get_sscp_trace(PX_F, PY)
    ss_resid_0 = get_sscp_trace(PX_0, PY)
    
    # Variance explained by the split
    ss_model = ss_resid_0 - ss_resid_F
    
    # Degrees of freedom: Full model has 2 params, Null has 1
    df_model = 1.0 
    df_resid = N - 2.0
    
    # F-ratio analogue
    F_obs = (ss_model / df_model) / (ss_resid_F / df_resid)
    
    # 5. RRPP Permutations
    F_null = torch.zeros(n_permutations, device=device)
    
    # Precompute P @ Y_hat_0 since it remains constant across permutations
    PY_hat_0 = P @ Y_hat_0
    
    for i in range(n_permutations):
        # Shuffle the rows of the raw residuals E_0
        perm_idx = torch.randperm(N, device=device)
        E_star = E_0[perm_idx]
        
        # Form pseudo-values Y* and apply the phylogenetic transformation
        # P @ Y* = P @ (Y_hat_0 + E_star) = P @ Y_hat_0 + P @ E_star
        PY_star = PY_hat_0 + (P @ E_star)
        
        # Calculate SS for the permuted data
        ss_resid_F_star = get_sscp_trace(PX_F, PY_star)
        ss_resid_0_star = get_sscp_trace(PX_0, PY_star)
        ss_model_star = ss_resid_0_star - ss_resid_F_star
        
        F_star = (ss_model_star / df_model) / (ss_resid_F_star / df_resid)
        F_null[i] = F_star
        
    # 6. Calculate Empirical p-value
    count_exceed = torch.sum(F_null >= F_obs).item()
    p_val = (count_exceed + 1) / (n_permutations + 1)
    
    return F_obs.item(), p_val
