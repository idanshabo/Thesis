from handling_esm_embeddings.create_esm_embeddings import create_esm_embeddings_from_fasta
from handling_esm_embeddings.convert_embeddings_to_one_mean_embedding import convert_embeddings_to_one_mean_embedding
import os


def normalize_matrix(matrix):
    mean = np.mean(matrix)
    std_dev = np.std(matrix)

    # Step 2: Normalize the matrix
    normalized_matrix = (matrix - mean) / std_dev
    return normalized_matrix



    
    cov_matrix = pd.read_csv(cov_matrix_path, index_col=0)
    cov_names = list(cov_matrix.index)
    file_names = data_dict['file_names']
    embeddings = data_dict['embeddings']
    
def create_normalized_mean_embeddings_matrix(fasta_file_path, output_path=None):
    if not output_path:
        output_path = os.path.join(os.path.dirname(fasta_file_path), 'embeddings_output')
    create_esm_embeddings_from_fasta(fasta_file_path, output_path)
    mean_embeddings_output_path = convert_embeddings_to_one_mean_embedding(output_path)
    data_dict = torch.load(data_dict_path, map_location='cpu')
    
    normalized_mean_embeddings = normalize_matrix()
    
    return 
