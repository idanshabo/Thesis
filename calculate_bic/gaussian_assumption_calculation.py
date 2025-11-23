import torch
import math


def calculate_log_likelihood(embeddings, regularization=1e-5):
    """
    Calculates the Log-Likelihood of the data under a Gaussian assumption.
    
    Args:
        embeddings (torch.Tensor): Shape (N, d)
        regularization (float): Small value added to diagonal for numerical stability.
    
    Returns:
        float: The log-likelihood value.
    """
    N, d = embeddings.shape
    
    # 1. Center the data (remove mean)
    mu = torch.mean(embeddings, dim=0)
    X_centered = embeddings - mu
    
    # 2. Calculate Empirical Covariance Matrix (Sigma)
    # Sigma = (1 / N) * (X - mu)^T * (X - mu)
    sigma = torch.matmul(X_centered.T, X_centered) / N
    
    # 3. Add Regularization (Critical for high dimensions to prevent singular matrix)
    # This ensures the determinant is not zero.
    sigma = sigma + torch.eye(d, device=embeddings.device) * regularization
    
    # 4. Calculate Log-Determinant of Sigma
    # We use cholesky or slogdet for stability
    sign, logdet = torch.linalg.slogdet(sigma)
    
    if sign <= 0:
        print("Warning: Covariance matrix is not positive definite. Increasing regularization.")
        return -float('inf')

    # 5. Calculate Log-Likelihood (simplified Gaussian formula)
    # LL = -0.5 * N * (d * ln(2pi) + ln(|Sigma|) + Tr(Sigma^-1 * S))
    # Note: Tr(Sigma^-1 * S) simplifies to 'd' when S is the MLE covariance.
    term1 = d * math.log(2 * math.pi)
    term2 = logdet
    term3 = d  # Trace part simplifies to d for MLE estimate
    
    ll = -0.5 * N * (term1 + term2 + term3)
    
    return ll.item()

def calculate_bic(log_likelihood, N, d, num_cov_matrices=1):
    """
    Calculates Bayesian Information Criterion (BIC).
    BIC = k * ln(N) - 2 * LL
    
    k (number of params) for covariance matrix approx: d(d+1)/2
    """
    # Number of unique parameters in a covariance matrix (symmetric d x d)
    params_per_matrix = (d * (d + 1)) / 2 + d # +d for the mean vector
    
    k = params_per_matrix * num_cov_matrices
    
    bic = k * math.log(N) - 2 * log_likelihood
    return bic
