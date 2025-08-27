import numpy as np
from scipy.linalg import cholesky
import torch
from typing import Tuple, List, Optional


def is_well_scaled(matrix: np.ndarray,
                  min_threshold: float = 1e-10,
                  max_threshold: float = 1e10) -> bool:
    """
    Check if matrix elements are within reasonable scale.
    """
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        return False
    abs_vals = np.abs(matrix[matrix != 0])  # Exclude zeros
    if len(abs_vals) == 0:
        return False
    return np.all(abs_vals > min_threshold) and np.all(abs_vals < max_threshold)

def stabilize_matrix(matrix: np.ndarray,
                    min_eigenval: float = 1e-10,
                    max_eigenval: float = 1e10) -> np.ndarray:
    """
    Stabilize a matrix by adjusting its eigenvalues to be within reasonable bounds.
    """
    eigenvals, eigenvecs = np.linalg.eigh(matrix)

    # Clip eigenvalues to reasonable range
    eigenvals = np.clip(eigenvals, min_eigenval, max_eigenval)

    # Reconstruct matrix
    return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

def safe_inverse(matrix: np.ndarray,
                ridge: float = 1e-6,
                min_eigenval: float = 1e-10) -> np.ndarray:
    """
    Compute inverse with protection against small eigenvalues.
    """
    try:
        # Try standard inverse first
        inv = np.linalg.inv(matrix)
        if is_well_scaled(inv):
            return inv
    except np.linalg.LinAlgError:
        pass

    # Add ridge and ensure minimum eigenvalue
    n = matrix.shape[0]
    stabilized = stabilize_matrix(matrix + ridge * np.eye(n), min_eigenval=min_eigenval)
    return np.linalg.inv(stabilized)

def matrix_normal_mle(X: List[np.ndarray],
                     epsilon1: float = 1e-6,
                     epsilon2: float = 1e-6,
                     V0: Optional[np.ndarray] = None,
                     max_iter: int = 1000,
                     min_samples_factor: float = 2.0,
                     min_eigenval: float = 1e-10,
                     max_eigenval: float = 1e10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerically stable Maximum Likelihood Estimation for Matrix Normal Distribution.

    Parameters:
    -----------
    X : List[np.ndarray]
        List of r independent n×p matrices from the matrix normal distribution
    epsilon1, epsilon2 : float
        Convergence criteria (default: 1e-6)
    V0 : Optional[np.ndarray]
        Initial guess for V matrix. If None, identity matrix is used
    max_iter : int
        Maximum number of iterations (default: 1000)
    min_samples_factor : float
        Factor to multiply minimum required samples (default: 2.0)
    min_eigenval : float
        Minimum allowed eigenvalue (default: 1e-10)
    max_eigenval : float
        Maximum allowed eigenvalue (default: 1e10)
    """
    # Convert list of matrices to 3D array
    X_array = np.array(X)
    if X_array.ndim  == 2:
        r = 1
        n, p = X_array.shape
    else:
        r, n, p = X_array.shape

    # More conservative sample size requirement
    if r < max(n/p, p/n) + 1:
        print(f"Warning: Sample size r ({r}) is too small for MLE existence")
        return(None, None, None)

    # Compute M (sample mean)
    M = np.mean(X_array, axis=0)

    # Initialize V
    if V0 is None:
        V_star = np.eye(p)
    else:
        # Ensure V0 is well-scaled
        if not is_well_scaled(V0):
            print("Warning: Initial V0 has extreme values. Using identity matrix instead.")
            V_star = np.eye(p)
        else:
            V_star = stabilize_matrix(V0, min_eigenval, max_eigenval)

    # Initialize variables
    step = 0
    U_plus = None
    V_plus = None

    while True:
        # Store previous values
        U_star = U_plus
        V_star_old = V_star

        # Center the data
        X_centered = X_array - M

        # Update U
        sum_term = np.zeros((n, n))
        V_inv = safe_inverse(V_star, min_eigenval=min_eigenval)

        if r == 1:
            sum_term = X_centered.T @ V_inv @ X_centered
        else:
            for k in range(r):
                term = X_centered[k] @ V_inv @ X_centered[k].T
                sum_term += term
        U_plus = (1/(p*r)) * sum_term

        # Stabilize U
        U_plus = stabilize_matrix(U_plus, min_eigenval, max_eigenval)

        # Update V
        sum_term = np.zeros((p, p))
        U_inv = safe_inverse(U_plus, min_eigenval=min_eigenval)

        if r == 1:
            sum_term = X_centered.T @ U_inv @ X_centered
        else:
            for k in range(r):
                term = X_centered[k].T @ U_inv @ X_centered[k]
                sum_term += term
        V_plus = (1/(n*r)) * sum_term

        # Stabilize V
        V_plus = stabilize_matrix(V_plus, min_eigenval, max_eigenval)

        # Update V_star for next iteration
        V_star = V_plus

        # Check convergence
        if U_star is not None:
            U_diff = np.linalg.norm(U_plus - U_star, ord='fro')
            V_diff = np.linalg.norm(V_plus - V_star_old, ord='fro')

            if not is_well_scaled(U_plus) or not is_well_scaled(V_plus):
                print("Numerical instability detected. Matrix values out of reasonable range.")
                return(None, None, None)

            if U_diff < epsilon1 and V_diff < epsilon2:
                break

        step += 1
        if step >= max_iter:
            print(f"Warning: Maximum iterations ({max_iter}) reached without convergence")
            break

    return M, U_plus, V_plus

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
