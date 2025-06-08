import numpy as np
import pandas as pd
import Torch


def align_embeddings_with_covariance(cov_matrix: pd.DataFrame, data_dict_path: str):
    """
    Aligns the order of embeddings and file names to match the order of the covariance matrix.
    
    Parameters:
    - cov_matrix: pd.DataFrame, covariance matrix with protein names as index and columns.
    - data_dict: dict, contains:
        - 'embeddings': np.ndarray, shape (n_proteins, embedding_dim)
        - 'file_names': list of str, length n_proteins

    Returns:
    - aligned_embeddings: np.ndarray, embeddings reordered to match the covariance matrix
    - aligned_file_names: list of str, file names reordered to match the covariance matrix
    """
    data_dict = torch.load(data_dict_path, map_location='cpu')
    cov_names = list(cov_matrix.index)
    file_names = data_dict['file_names']
    embeddings = data_dict['embeddings']

    if set(cov_names) != set(file_names):
        missing_in_cov = set(file_names) - set(cov_names)
        missing_in_files = set(cov_names) - set(file_names)
        raise ValueError(f"Mismatch between covariance matrix and file names:\n"
                         f"Missing in covariance matrix: {missing_in_cov}\n"
                         f"Missing in file names: {missing_in_files}")

    # Create a mapping from file name to embedding row
    name_to_index = {name: idx for idx, name in enumerate(file_names)}
    
    # Reorder embeddings and file_names
    aligned_embeddings = np.array([embeddings[name_to_index[name]] for name in cov_names])
    aligned_file_names = cov_names  # Now aligned

    return aligned_embeddings, aligned_file_names
