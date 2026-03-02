import os
import shutil
import time
import torch
import numpy as np
from sklearn.decomposition import PCA

# --- Utils Imports ---
from utils.read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from utils.convert_stockholm_to_fasta import convert_stockholm_to_fasta
from utils.phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from utils.save_results_json import save_results_json
from utils.order_covariance_matrix import order_covariance_matrix_by_tree
from create_phylogenetic_tree.run_fasttree import run_fasttree
from handling_esm_embeddings.create_normalized_mean_embeddings_matrix import create_normalized_mean_embeddings_matrix
from evaluate_split_options.evaluate_split_options import evaluate_top_splits
from significant_split_evaluation.visualisations import visualize_split_msa_sorted, plot_split_covariance, run_variance_analysis
from significant_split_evaluation.handle_splits_evaluation import get_split_info
from significant_split_evaluation.structures.visualize_structures_pipeline import visualize_structures_pipeline

# --- Metadata Utilities ---
from utils.metadata_tracker import MetadataTracker
from utils.msa_stats import get_msa_stats

def run_pipeline(MSA_file_path, 
                 print_file_content=False, 
                 output_path=None, 
                 number_of_nodes_to_evaluate=5,
                 pca_min_variance=0.99,
                 pca_min_components=100,
                 standardize=True):

    # --- 0. Initialize Metadata Tracker ---
    base_dir = os.path.dirname(os.path.abspath(MSA_file_path))
    family_name = os.path.basename(MSA_file_path).split('.')[0] 
    out_dir = os.path.join(base_dir, f"{family_name}_outputs")
    calc_dir = os.path.join(base_dir, f"{family_name}_calculations")
    
    # Initialize tracker
    tracker = MetadataTracker(os.path.join(out_dir, "pipeline_metadata.json"))
    tracker.add_stat("pipeline_stats", "family_name", family_name)
    pipeline_start = time.time()
    
    try:
        # --- 1. Setup & MSA Stats ---
        tracker.start_timer("Setup_and_MSA_Stats")
        os.makedirs(calc_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        if print_file_content:
            read_stockholm_file_and_print_content(MSA_file_path)

        msa_stats = get_msa_stats(MSA_file_path)
        for k, v in msa_stats.items():
            tracker.add_stat("msa_stats", k, v)
        tracker.stop_timer()

        # --- 2. Calculations ---
        tracker.start_timer("Calculations")
        
        # Fasta
        temp_fasta = convert_stockholm_to_fasta(MSA_file_path)
        fasta_path = os.path.join(calc_dir, os.path.basename(temp_fasta))
        shutil.move(temp_fasta, fasta_path)

        # Tree
        temp_tree = run_fasttree(fasta_path)
        tree_path = os.path.join(calc_dir, os.path.basename(temp_tree))
        shutil.move(temp_tree, tree_path)

        # Covariance
        temp_cov = tree_to_covariance_matrix(tree_path)
        cov_path = os.path.join(calc_dir, os.path.basename(temp_cov))
        shutil.move(temp_cov, cov_path)
        cov_ordered_path = order_covariance_matrix_by_tree(cov_path, tree_path)
        
        # Embeddings
        emb_out_dir = os.path.join(calc_dir, 'embeddings_output')
        norm_emb_path = create_normalized_mean_embeddings_matrix(fasta_path, emb_out_dir)
        
        # Metadata: Embedding Dimension
        try:
            emb_data = torch.load(norm_emb_path)
            if isinstance(emb_data, torch.Tensor): emb_data = emb_data.numpy()
            pca = PCA(n_components=pca_min_variance)
            pca.fit(emb_data)
            tracker.add_stat("pipeline_stats", f"embedding_dim_{int(pca_min_variance*100)}pct_var", int(pca.n_components_))
        except Exception as e:
            print(f"Warning: Could not calc embedding dim: {e}")

        tracker.stop_timer()

        # --- 3. Evaluate Splits ---
        tracker.start_timer("Split_Evaluation")
        results = evaluate_top_splits(tree_path, cov_path, norm_emb_path, 
                                      output_path=out_dir, k=number_of_nodes_to_evaluate, 
                                      pca_min_variance=pca_min_variance, 
                                      pca_min_components=pca_min_components, 
                                      standardize=standardize)
        tracker.stop_timer()

        # --- 4. Save Results ---
        save_results_json(results, os.path.join(out_dir, "results.json"))

        # --- 5. Visualizations & Metadata ---
        tracker.start_timer("Visualization_and_Analysis")
        sig_splits_path = os.path.join(out_dir, 'significant_splits')
        
        sig_count = 0
        if os.path.exists(sig_splits_path):
            splits = [d for d in os.listdir(sig_splits_path) if os.path.isdir(os.path.join(sig_splits_path, d))]
            sig_count = len(splits)
            
            for folder_name in splits:
                folder_path = os.path.join(sig_splits_path, folder_name)
                split_info = get_split_info(folder_path)
                
                visualize_split_msa_sorted(fasta_path, split_info, folder_path)
                plot_split_covariance(cov_ordered_path, split_info, folder_path)
                
                # --- TM Score Calculation happens HERE ---
                # pass max_structures=50 to limit calculation time
                # capture tm_stats return value
                tm_stats = visualize_structures_pipeline(
                    fasta_path, 
                    split_info, 
                    folder_path, 
                    cov_ordered_path
                )
                
                if tm_stats:
                    tracker.add_split_stat(folder_name, tm_stats)
                
                run_variance_analysis(folder_path)

        tracker.add_stat("pipeline_stats", "num_significant_splits", sig_count)
        tracker.stop_timer()

    except Exception as e:
        print(f"!!! Pipeline Crashed !!! Error: {e}")
        tracker.add_stat("errors", "crash_reason", str(e))
        raise e

    finally:
        total_time = time.time() - pipeline_start
        tracker.add_stat("timings", "total_pipeline_time", round(total_time, 2))
        tracker.save()
        print(f"Metadata saved to: {tracker.output_path}")
