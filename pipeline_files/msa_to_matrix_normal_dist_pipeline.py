from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree
from phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from handling_esm_embeddings.create_normalized_mean_embeddings_matrix import create_normalized_mean_embeddings_matrix
from estimate_matrix_normal.estimate_matrix_normal import matrix_normal_mle_fixed_u, matrix_normal_mle
from align_embeddings_with_covariance import align_embeddings_with_covariance
from check_matching_names import check_matching_names
import os
import torch


def run_pipeline(MSA_file_path, print_file_content=False, output_path=None):
    if print_file_content:
        read_stockholm_file_and_print_content(MSA_file_path)
    fasta_file_path = convert_stockholm_to_fasta(MSA_file_path)
    phylogenetic_tree_path = run_fasttree(fasta_file_path)
    cov_mat_path = tree_to_covariance_matrix(phylogenetic_tree_path)
    if not output_path:
        output_path = os.path.join(os.path.dirname(fasta_file_path), 'embeddings_output')

    normalized_mean_embeddings_path = create_normalized_mean_embeddings_matrix(fasta_file_path, output_path)
    embeddings_matrix = align_embeddings_with_covariance(cov_mat_path, normalized_mean_embeddings_path)
    Mean_mat, V_mat, cov_mat = matrix_normal_mle_fixed_u(X=[embeddings_matrix], U_path=cov_mat_path)
    return Mean_mat, V_mat, cov_mat
