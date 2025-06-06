from handling_esm_embeddings.create_esm_embeddings import create_esm_embeddings_from_fasta
from handling_esm_embeddings.convert_embeddings_to_one_mean_embedding import convert_embeddings_to_one_mean_embedding
import os


def run_pipeline(MSA_file_path, output_path=None):
    if not output_path:
        base_path = os.path.splitext(MSA_file_path)[0].replace('.fasta', '')
        output_path = base_path + '/embeddings_output'
    create_esm_embeddings_from_fasta(MSA_file_path, output_path)
    mean_embeddings_output_path = convert_embeddings_to_one_mean_embedding(output_path)
    return mean_embeddings_output_path
