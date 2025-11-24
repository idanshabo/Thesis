import torch
import pandas as pd
import math


def load_matrix_tensor(path):
    """
    Robustly loads a matrix from .pt or .csv into a torch.Tensor.
    """
    if path.endswith('.pt'):
        return torch.load(path, map_location='cpu').float()
    elif path.endswith('.csv'):
        df = pd.read_csv(path, index_col=0)
        return torch.tensor(df.values).float()
    else:
        raise ValueError(f"Unknown file format for matrix: {path}")

def get_log_det(matrix_tensor, regularization=1e-5):
    """
    Calculates log-determinant with regularization for stability.
    """
    if matrix_tensor.shape[0] != matrix_tensor.shape[1]:
         raise ValueError(f"Matrix must be square. Got {matrix_tensor.shape}")
    
    d = matrix_tensor.shape[0]
    # Add small noise to diagonal to prevent log(0) or negative det
    reg_matrix = matrix_tensor + torch.eye(d, device=matrix_tensor.device) * regularization
    
    sign, logdet = torch.linalg.slogdet(reg_matrix)
    
    if sign <= 0:
        print("Warning: Determinant sign is negative or zero. Using absolute value for LogDet.")
        
    return logdet

def calculate_matrix_normal_ll(n, p, u_tensor, v_tensor):
    """
    Calculates the Maximized Log-Likelihood for the Matrix Normal Distribution.
    
    Formula: LL_max = -0.5 * ( np(1 + ln(2pi)) + p*ln|U| + n*ln|V| )
    
    Args:
        n (int): Number of taxa (rows)
        p (int): Embedding dimension (columns)
        u_tensor (Tensor): Phylogenetic covariance matrix (n x n)
        v_tensor (Tensor): Estimated embedding covariance matrix (p x p)
    """
    # 1. Calculate Determinants
    logdet_u = get_log_det(u_tensor)
    logdet_v = get_log_det(v_tensor)

    # 2. Compute terms
    # Constant term: n*p * (1 + ln(2pi))
    term_const = n * p * (1 + math.log(2 * math.pi))
    
    # Row covariance term: p * ln|U|
    term_u = p * logdet_u
    
    # Column covariance term: n * ln|V|
    term_v = n * logdet_v
    
    # 3. Sum and scale
    ll = -0.5 * (term_const + term_u + term_v)
    return ll.item()

def calculate_bic_matrix_normal(log_likelihood, n, p, num_models=1):
    """
    Calculates BIC for Matrix Normal estimation.
    k = number of free parameters in V (Embedding Covariance).
    We assume U is fixed (phylogeny), so it adds 0 parameters.
    V is symmetric p x p, so p(p+1)/2 parameters.
    """
    params_per_v = (p * (p + 1)) / 2
    # Add p for the Mean matrix M (if M is estimated freely per column)
    # Often M is approximated as 0 or mean, let's assume minimal params for M (p)
    k = (params_per_v + p) * num_models
    
    # Sample size is N (taxa)
    bic = k * math.log(n) - 2 * log_likelihood
    return bic
