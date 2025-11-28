from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree
from phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from handling_esm_embeddings.create_normalized_mean_embeddings_matrix import create_normalized_mean_embeddings_matrix
from check_matching_names import check_matching_names
import os
import torch
import json
from evaluate_split_options.evaluate_split_options import evaluate_top_splits
from significant_split_evaluation.visualisations import visualize_split_msa_sorted
from significant_split_evaluation.structures.predict_structures import process_fasta_to_structures

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
    items = os.listdir(significant_splits_output_path)
    
    for folder_name in items:
        folder_path = os.path.join(significant_splits_output_path, folder_name)
        
        # 2. Check if the item is actually a folder
        if os.path.isdir(folder_path):
            
            # 3. Find the JSON file inside this folder
            sub_files = os.listdir(folder_path)
            json_file = [f for f in sub_files if f.endswith('.json')][0]
            json_file_path = os.path.join(folder_path, json_file)
            with open(json_file_path, 'r') as f:
              split_info = json.load(f)
            viz_dir = os.path.join(folder_path, "visualization")
            os.makedirs(viz_dir, exist_ok=True)
            specific_output_plot = os.path.join(viz_dir, "ordered_split_MSA.png")
            visualize_split_msa_sorted(fasta_file_path, split_info, specific_output_plot)
            # your code here
    structures_folder = os.path.join(os.path.dirname(fasta_file_path), 'structures')
    process_fasta_to_structures(fasta_file_path, structures_folder)
    return results
