import pandas as pd
import numpy as np

def assure_cov_mat_positive_definite(path_to_matrix, alpha=1e-6):
    """
    Adds a small value (alpha) to the diagonal of the matrix to make it positive definite.
    """
    matrix = pd.read_csv(path_to_matrix)    
    eigenvalues = np.linalg.eigvals(matrix)
    smallest_eigenvalue = np.min(np.real(eigenvalues)) 

    if smallest_eigenvalue < 0:
        if abs(smallest_eigenvalue) < alpha:
            print(f"Smallest eigenvalue is negative, but very close to 0 ({smallest_eigenvalue}). Regularizing matrix...")
            matrix = matrix + alpha * np.eye(matrix.shape[0])
        else:
            print(f"Smallest eigenvalue is negative ({smallest_eigenvalue}), which is too far from 0. Matrix is not well-behaved.")
    else:
        print(f"Smallest eigenvalue is positive: {smallest_eigenvalue}. Matrix is behaving well.")
    
    # Convert back to DataFrame (optional, for consistency with input format)
    matrix_df = pd.DataFrame(matrix, index=pd.read_csv(path_to_matrix, index_col=0).index, 
                             columns=pd.read_csv(path_to_matrix, index_col=0).columns)
    
    # Optionally, save the updated matrix back to CSV
    matrix_df.to_csv(path_to_matrix)
    
    return path_to_matrix
