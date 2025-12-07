from utils.read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from utils.convert_stockholm_to_fasta import convert_stockholm_to_fasta
from utils.phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from utils.check_matching_names import check_matching_names
from utils.save_results_json import save_results_json
from create_phylogenetic_tree.run_fasttree import run_fasttree
from handling_esm_embeddings.create_normalized_mean_embeddings_matrix import create_normalized_mean_embeddings_matrix
import os
import shutil
import json
import torch
import numpy as np
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
                 pca_min_variance=0.99,
                 pca_min_components=100,
                 standardize=True):

    # --- 1. Setup Directories ---
    base_dir = os.path.dirname(os.path.abspath(MSA_file_path))
    family_name = os.path.splitext(os.path.basename(MSA_file_path))[0]

    calc_dir = os.path.join(base_dir, f"{family_name}_calculations")
    out_dir = os.path.join(base_dir, f"{family_name}_outputs")

    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if print_file_content:
        read_stockholm_file_and_print_content(MSA_file_path)

    # --- 2. Run Calculations & Move Intermediate Files ---
    
    # Fasta
    temp_fasta_path = convert_stockholm_to_fasta(MSA_file_path)
    fasta_file_path = os.path.join(calc_dir, os.path.basename(temp_fasta_path))
    shutil.move(temp_fasta_path, fasta_file_path)

    # Tree
    temp_tree_path = run_fasttree(fasta_file_path)
    phylogenetic_tree_path = os.path.join(calc_dir, os.path.basename(temp_tree_path))
    shutil.move(temp_tree_path, phylogenetic_tree_path)

    # Covariance
    temp_cov_path = tree_to_covariance_matrix(phylogenetic_tree_path)
    cov_mat_path = os.path.join(calc_dir, os.path.basename(temp_cov_path))
    shutil.move(temp_cov_path, cov_mat_path)
    
    # Embeddings
    normalized_mean_embeddings_path = create_normalized_mean_embeddings_matrix(fasta_file_path, calc_dir)

    # --- 3. Evaluate Splits (Output to _outputs folder) ---
    results = evaluate_top_splits(phylogenetic_tree_path, 
                                  cov_mat_path, 
                                  normalized_mean_embeddings_path, 
                                  output_path=out_dir, 
                                  k=number_of_nodes_to_evaluate, 
                                  pca_min_variance=pca_min_variance, 
                                  pca_min_components=pca_min_components,
                                  standardize=standardize)

    # --- 4. Save Results Object (Using NumpyEncoder) ---
    results_file_path = os.path.join(out_dir, "results.json")
    try:
        with open(results_file_path, 'w') as f:
            # cls=NumpyEncoder handles the conversion of float32, int64, bool_, etc.
            json.dump(results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error saving JSON even with Encoder: {e}")
        # Last resort fallback
        with open(results_file_path, 'w') as f:
            f.write(str(results))

    # --- 5. Visualizations ---
    significant_splits_output_path = os.path.join(out_dir, 'splits_evaluations', 'significant_splits')
    
    if os.path.exists(significant_splits_output_path):
        sig_splits = os.listdir(significant_splits_output_path)
        if sig_splits:        
            for folder_name in sig_splits:
                folder_path = os.path.join(significant_splits_output_path, folder_name)
                # Ensure we are looking at directories, not hidden files
                if os.path.isdir(folder_path):
                    split_info = get_split_info(folder_path)
                    
                    visualize_split_msa_sorted(fasta_file_path, split_info, folder_path)
                    plot_split_covariance(cov_mat_path, split_info, folder_path)
                    visualize_structures_pipeline(fasta_file_path, split_info, folder_path)
