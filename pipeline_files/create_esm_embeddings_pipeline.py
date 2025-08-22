from handling_esm_embeddings.create_esm_embeddings import create_esm_embeddings_from_fasta
from handling_esm_embeddings.convert_embeddings_to_one_mean_embedding import convert_embeddings_to_one_mean_embedding
from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
import os


def run_pipeline(MSA_file_path, output_path=None, print_file_content=False):
    if print_file_content:
        read_stockholm_file_and_print_content(MSA_file_path)
    fasta_file_path = convert_stockholm_to_fasta(MSA_file_path)
    if not output_path:
        output_path = os.path.join(os.path.dirname(fasta_file_path), 'embeddings_output')
    create_esm_embeddings_from_fasta(fasta_file_path, output_path)
    mean_embeddings_output_path = convert_embeddings_to_one_mean_embedding(output_path)
    return mean_embeddings_output_path
