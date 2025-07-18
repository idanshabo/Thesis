from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree
from phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from handling_esm_embeddings.create_esm_embeddings import create_esm_embeddings_from_fasta
from handling_esm_embeddings.convert_embeddings_to_one_mean_embedding import convert_embeddings_to_one_mean_embedding
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
    cov_mat = tree_to_covariance_matrix(phylogenetic_tree_path)
    if not output_path:
        base_path = os.path.splitext(fasta_file_path)[0].replace('.fasta', '')
        output_path = base_path + '/embeddings_output'
    create_esm_embeddings_from_fasta(fasta_file_path, output_path)
    mean_embeddings_dict_path = convert_embeddings_to_one_mean_embedding(output_path)
    matching_names = check_matching_names(cov_mat, mean_embeddings_dict_path)
    if not matching_names:
        return(False)
    embeddings_matrix, protein_list = align_embeddings_with_covariance(cov_mat, mean_embeddings_dict_path)
    print(embeddings_matrix.shape)
    print(cov_mat.shape)
    Mean_mat, V_mat = matrix_normal_mle_fixed_u(X=[embeddings_matrix], U=cov_mat)
    return Mean_mat, V_mat, cov_mat
