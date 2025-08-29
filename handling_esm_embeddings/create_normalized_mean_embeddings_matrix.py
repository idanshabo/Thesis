from handling_esm_embeddings.create_esm_embeddings import create_esm_embeddings_from_fasta
from handling_esm_embeddings.convert_embeddings_to_one_mean_embedding import convert_embeddings_to_one_mean_embedding
import os
import numpy as np


def normalize_matrix(matrix):
    mean = np.mean(matrix)
    std_dev = np.std(matrix)
    normalized_matrix = (matrix - mean) / std_dev
    return normalized_matrix

    
def create_normalized_mean_embeddings_matrix(fasta_file_path, output_path=None):
    if not output_path:
        output_path = os.path.join(os.path.dirname(fasta_file_path), 'embeddings_output')
        normalized_mean_embeddings_output_path = output_path + '/mean_embeddings_output/normalized_mean_protein_embeddings.pt'
    
    create_esm_embeddings_from_fasta(fasta_file_path, output_path)
    mean_embeddings, file_names = convert_embeddings_to_one_mean_embedding(output_path)
    normalized_mean_embeddings = normalize_matrix(mean_embeddings)
    
    torch.save({
        'embeddings': normalized_mean_embeddings,
        'file_names': file_names
    }, normalized_mean_embeddings_output_path)
    return normalized_mean_embeddings_output_path
