import numpy as np
import pandas as pd
import os
from typing import Tuple, List

# Assume helper functions (is_well_scaled, stabilize_matrix, safe_inverse) are defined above

def matrix_normal_mle_fixed_u(X: List[np.ndarray],
                               U_path: str) -> Tuple[str, str, str]:
    """
    Maximum Likelihood Estimation for Matrix Normal Distribution with a fixed U,
    using the direct closed-form solution.

    Parameters:
    -----------
    X : List[np.ndarray]
        List of r independent n×p matrices from the matrix normal distribution.
    U_path : str
        Path to the saved n×n row covariance matrix U (as a CSV file).

    Returns:
    --------
    mean_output_path : str
        Path to the saved estimated mean matrix M (n×p).
    embeddings_cov_output_path : str
        Path to the saved estimated column covariance matrix V (p×p).
    U_path : str
        The original path to the U matrix.
    """
    try:
        U = pd.read_csv(U_path, index_col=0).values
    except Exception as e:
        raise IOError(f"Failed to load U matrix from {U_path}. Error: {e}")

    X_array = np.array(X)
    if X_array.ndim == 2:
        X_array = np.expand_dims(X_array, axis=0)

    r, n, p = X_array.shape

    if U.shape != (n, n):
        raise ValueError(f"The shape of the U matrix must be ({n}, {n}), but got {U.shape}.")

    if r * n <= p:
        print(f"Warning: The condition rn > p is not met (r={r}, n={n}, p={p}). "
              "The estimated V matrix may be singular.")

    # --- MLE Calculation using Closed-Form Solution ---

    # Step 1: Calculate the sample mean of the data.
    M = np.mean(X_array, axis=0)

    # Step 2: Center the data. This is the crucial step to ensure the data has mean 0,
    # as required by the formula. X_centered now corresponds to the 'X' in the formula's context.
    X_centered = X_array - M

    # Step 3: Apply the generalized formula for V_hat.
    # V_hat = (1/(r*n)) * Σ [ X_centered_k.T * U⁻¹ * X_centered_k ]
    
    U_inv = safe_inverse(U)

    # Sum the quadratic form over all r samples
    sum_term = np.zeros((p, p))
    for k in range(r):
        # This is X_centered' * U_inv * X_centered for the k-th sample
        term = X_centered[k].T @ U_inv @ X_centered[k]
        sum_term += term
    
    # Normalize by (n*r) to get the final estimate
    V_hat = (1 / (n * r)) * sum_term
    
    V_hat_stabilized = stabilize_matrix(V_hat)

    # --- Save Results to CSV Files ---
    output_dir = os.path.dirname(U_path)
    pfam_family = os.path.basename(output_dir)

    M_df = pd.DataFrame(M)
    V_hat_df = pd.DataFrame(V_hat_stabilized)
    
    mean_output_path = os.path.join(output_dir, f'{pfam_family}_Mean.csv')
    embeddings_cov_output_path = os.path.join(output_dir, f'{pfam_family}_embeddings_cov_mat.csv')

    M_df.to_csv(mean_output_path)
    V_hat_df.to_csv(embeddings_cov_output_path)

    return mean_output_path, embeddings_cov_output_path, U_path
