from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree
from phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from handling_esm_embeddings.create_esm_embeddings import create_esm_embeddings_from_fasta
from handling_esm_embeddings.convert_embeddings_to_one_mean_embedding import convert_embeddings_to_one_mean_embedding
from estimate_matrix_normal.estimate_matrix_normal import matrix_normal_mle_fixed_u, matrix_normal_mle
import os


def run_pipeline(MSA_file_path, print_file_content=False, output_path=None):
    if print_file_content:
        read_stockholm_file_and_print_content(MSA_file_path)
    fasta_file_path = convert_stockholm_to_fasta(MSA_file_path)
    phylogenetic_tree_path = run_fasttree(fasta_file_path)
    cov_mat = tree_to_covariance_matrix(phylogenetic_tree_path)
    if not output_path:
        base_path = os.path.splitext(MSA_file_path)[0].replace('.fasta', '')
        output_path = base_path + '/embeddings_output'
    create_esm_embeddings_from_fasta(MSA_file_path, output_path)
    mean_embeddings_output_path = convert_embeddings_to_one_mean_embedding(output_path)
    matrix_normal_estimation = matrix_normal_mle_fixed_u(X=List[np.ndarray], U: np.ndarray)
    return mean_embeddings_output_path
    return cov_mat
