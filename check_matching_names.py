import torch

def check_matching_names(cov_matrix, data_dict_path):
    # Check if the column names of the covariance matrix match the file names in the embeddings
    data_dict = torch.load(data_dict_path, map_location='cpu')
    cov_names = list(cov_matrix.index)
    file_names = data_dict['file_names']

    cov_columns = cov_matrix.columns
    embeddings_file_names = file_names
    
    if set(cov_columns) != set(embeddings_file_names):
        missing_in_cov = set(embeddings_file_names) - set(cov_columns)
        missing_in_files = set(cov_columns) - set(embeddings_file_names)
        print(f"Mismatch found! The following species are missing:")
        print(f"Missing in covariance matrix: {missing_in_cov}")
        print(f"Missing in file names (embeddings): {missing_in_files}")
        return False
    else:
        print("All species names match between covariance matrix and embeddings.")
        return True
