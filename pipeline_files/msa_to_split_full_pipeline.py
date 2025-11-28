from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree
from phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from handling_esm_embeddings.create_normalized_mean_embeddings_matrix import create_normalized_mean_embeddings_matrix
from check_matching_names import check_matching_names
import os
import torch
from evaluate_split_options import evaluate_top_splits


def run_pipeline(MSA_file_path, print_file_content=False, output_path=None):
    if print_file_content:
        read_stockholm_file_and_print_content(MSA_file_path)
    fasta_file_path = convert_stockholm_to_fasta(MSA_file_path)
    phylogenetic_tree_path = run_fasttree(fasta_file_path)
    cov_mat_path = tree_to_covariance_matrix(phylogenetic_tree_path)
    if not output_path:
        output_path = os.path.join(os.path.dirname(fasta_file_path), 'embeddings_output')
    
    normalized_mean_embeddings_path = create_normalized_mean_embeddings_matrix(fasta_file_path, output_path)
    significant_splits_output_path = os.path.join(os.path.dirname(fasta_file_path), 'significant_splits')
    results = evaluate_top_splits(phylogenetic_tree_path, cov_mat_path, normalized_mean_embeddings_path, output_path=significant_splits_output_path, k=5, target_pca_variance=0.99, standardize=True)
    
    files = os.listdir(significant_splits_output_path)
    if files:
        for file_name in files:
            file_path = os.path.join(significant_splits_output_path, file_name)
            print("Processing:", file_path)
            # your code here
    return results
