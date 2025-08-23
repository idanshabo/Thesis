import torch
import pandas as pd


def check_matching_names(cov_matrix_path, data_dict_path):
    # Check if the column names of the covariance matrix match the file names in the embeddings
    data_dict = torch.load(data_dict_path, map_location='cpu')
    cov_matrix = pd.read_csv(cov_matrix_path)
    embeddings_file_names = set(data_dict['file_names'])

    cov_columns = set(cov_matrix.columns[1:])
    
    if cov_columns != embeddings_file_names:
        missing_in_cov = embeddings_file_names - cov_columns
        missing_in_files = cov_columns - embeddings_file_names
        print(f"Mismatch found! The following species are missing:")
        print(f"Missing in covariance matrix: {missing_in_cov}")
        print(f"Missing in file names (embeddings): {missing_in_files}")
        return False
    else:
        print("All species names match between covariance matrix and embeddings.")
        return True
