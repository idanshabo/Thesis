from ete3 import Tree
import numpy as np
import pandas as pd
import os
import re

def tree_to_covariance_matrix(tree_path, output_path = None):
    if not output_path:
        base_path = os.path.splitext(tree_path)[0].replace('.tree', '')
        cov_mat_file_path = base_path + '_cov_mat.csv'
    output_path = cov_mat_file_path

    if os.path.exists(output_path):
        print(f"covariance matrix already exists in path {output_path}")
        return(output_path)

    
    print("starting to calculate covariance matrix\n")
    # Load and root the tree
    tree = Tree(tree_path, format=1)
    tree.set_outgroup(tree.get_midpoint_outgroup())  # Or choose a known outgroup

    # Get the actual root node
    root = tree.get_tree_root()

    # Get species (leaf names)
    species = [leaf.name for leaf in tree.get_leaves()]
    
    # Clean species names (column and row names)
    cleaned_species = [re.sub(r'[\\/*?:"<>|]', '_', sp) for sp in species]
    cleaned_species = [re.sub(r'\s+', '_', sp) for sp in cleaned_species]  # Optional: replace spaces with underscores
    
    n = len(species)

    # Initialize the covariance matrix
    cov_matrix = np.zeros((n, n))

    # Compute shared distance from root to MRCA
    for i, sp1 in enumerate(species):
        for j, sp2 in enumerate(species):
            if i == j:
                cov_matrix[i, j] = root.get_distance(sp1)
            else:
                mrca = tree.get_common_ancestor(sp1, sp2)
                shared_distance = root.get_distance(mrca)
                cov_matrix[i, j] = shared_distance

    # Convert to DataFrame with cleaned species names
    cov_df = pd.DataFrame(cov_matrix, index=cleaned_species, columns=cleaned_species)
    cov_df = assure_cov_mat_positive_definite(cov_df)
    cov_df.to_csv(output_path, index=True)
    print("Calculated covariance matrix successfully")
    return output_path


def assure_cov_mat_positive_definite(matrix, alpha=1e-6):
    """
    Adds a small value (alpha) to the diagonal of the matrix to make it positive definite.
    """
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
    matrix_df = pd.DataFrame(matrix)
    return matrix_df
