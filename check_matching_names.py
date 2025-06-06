def check_matching_names(cov_matrix_df, embeddings, file_names):
    # Check if the column names of the covariance matrix match the file names in the embeddings
    cov_columns = cov_matrix_df.columns
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
