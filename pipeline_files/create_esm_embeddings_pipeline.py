from handling_esm_embeddings.create_esm_embeddings import create_esm_embeddings_from_fasta
from handling_esm_embeddings.convert_embeddings_to_one_mean_embedding import convert_embeddings_to_one_mean_embedding
from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
import os


def run_pipeline(MSA_file_path, output_path=None, print_file_content=False):
    if not output_path:
        base_path = os.path.splitext(MSA_file_path)[0].replace('.fasta', '')
        output_path = base_path + '/embeddings_output'
    if print_file_content:
        read_stockholm_file_and_print_content(MSA_file_path)
    fasta_file_path = convert_stockholm_to_fasta(MSA_file_path)
    create_esm_embeddings_from_fasta(fasta_file_path, output_path)
    mean_embeddings_output_path = convert_embeddings_to_one_mean_embedding(output_path)
    return mean_embeddings_output_path
