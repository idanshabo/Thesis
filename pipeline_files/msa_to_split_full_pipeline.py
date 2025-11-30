from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree
from phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from handling_esm_embeddings.create_normalized_mean_embeddings_matrix import create_normalized_mean_embeddings_matrix
from check_matching_names import check_matching_names
import os
import torch
import json
from Bio import SeqIO
from evaluate_split_options.evaluate_split_options import evaluate_top_splits
from significant_split_evaluation.visualisations import visualize_split_msa_sorted
from significant_split_evaluation.visualisations import plot_split_covariance
from significant_split_evaluation.handle_splits_evaluation import get_split_info
from significant_split_evaluation.structures.visualize_structures_pipeline import visualize_structures_pipeline


def run_pipeline(MSA_file_path, 
                 print_file_content=False, 
                 output_path=None, 
                 number_of_nodes_to_evaluate=5,
                 target_pca_variance=0.99,
                 standardize=True):
    if print_file_content:
        read_stockholm_file_and_print_content(MSA_file_path)
    fasta_file_path = convert_stockholm_to_fasta(MSA_file_path)
    phylogenetic_tree_path = run_fasttree(fasta_file_path)
    cov_mat_path = tree_to_covariance_matrix(phylogenetic_tree_path)
    if not output_path:
        output_path = os.path.join(os.path.dirname(fasta_file_path), 'embeddings_output')
    
    normalized_mean_embeddings_path = create_normalized_mean_embeddings_matrix(fasta_file_path, output_path)
    results = evaluate_top_splits(phylogenetic_tree_path, 
                                  cov_mat_path, 
                                  normalized_mean_embeddings_path, 
                                  output_path=os.path.dirname(fasta_file_path), 
                                  k=number_of_nodes_to_evaluate, 
                                  target_pca_variance=target_pca_variance, 
                                  standardize=standardize)
    significant_splits_output_path = os.path.join(os.path.dirname(fasta_file_path), 'splits_evaluations', 'significant_splits')
    sig_splits = os.listdir(significant_splits_output_path)
    if sig_splits:        
      for folder_name in sig_splits:
          folder_path = os.path.join(significant_splits_output_path, folder_name)
          split_info = get_split_info(folder_path)
          visualize_split_msa_sorted(fasta_file_path, split_info, folder_path)
          plot_split_covariance(fasta_file_path, split_info, folder_path)
          visualize_structures_pipeline(fasta_file_path, split_info, folder_path)
    return results
