import os
import re
import shutil
import time
import argparse
import requests

# --- Utils Imports ---
from utils.read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from utils.convert_stockholm_to_fasta import convert_stockholm_to_fasta
from utils.phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from utils.save_results_json import save_results_json
from utils.order_covariance_matrix import order_covariance_matrix_by_tree
from create_phylogenetic_tree.run_fasttree import run_fasttree
from handling_esm_embeddings.create_normalized_mean_embeddings_matrix import create_normalized_mean_embeddings_matrix
from evaluate_split_options.evaluate_split_options import evaluate_top_splits
from significant_split_evaluation.visualisations import visualize_split_msa_sorted, plot_split_covariance, run_variance_analysis, plot_side_by_side_embedding_covariance
from significant_split_evaluation.handle_splits_evaluation import get_split_info
from significant_split_evaluation.structures.visualize_structures_pipeline import visualize_structures_pipeline

# --- Metadata Utilities ---
from utils.metadata_tracker import MetadataTracker
from utils.msa_stats import get_msa_stats, calc_norm_branch_length, calc_msa_similarity


def extract_protein_flags(clean_desc):
    """Checks the description against lists of keywords for different protein groups."""
    desc_lower = clean_desc.lower()
    
    # Define our categories and their associated keywords
    categories = {
        "is_membrane": ["membrane", "transmembrane"],
        "is_kinase": ["kinase"],
        "is_disordered": ["disordered", "unstructured", "idp"],
        "is_allosteric": ["allosteric", "conformational change"],
        "is_nucleic_binding": ["dna-binding", "rna-binding", "transcription factor"],
        "is_viral": ["viral", "virus", "phage"],
        "is_receptor": ["receptor", "gpcr"],
        "is_chaperone": ["chaperone", "heat shock"]
    }
    
    # Create a dictionary to hold the results (default all to False)
    flags = {key: False for key in categories}
    
    # Check for keywords
    for flag, keywords in categories.items():
        if any(keyword in desc_lower for keyword in keywords):
            flags[flag] = True
            
    return flags

def fetch_pfam_metadata(family_id):
    """Fetches family description and GO terms from InterPro API."""
    url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{family_id.upper()}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json().get('metadata', {})
            
            # Safely extract the description text, handling different API formats
            desc_data = data.get('description')
            desc_text = ""
            
            if isinstance(desc_data, dict):
                desc_text = desc_data.get('text', '')
            elif isinstance(desc_data, list) and len(desc_data) > 0:
                desc_text = str(desc_data[0])
            elif isinstance(desc_data, str):
                desc_text = desc_data
                
            # Clean up HTML tags (like <p>) from the text
            clean_desc = re.sub('<[^<]+>', '', desc_text).strip()
            
            # Get all the boolean flags dynamically
            result_dict = {"description": clean_desc if clean_desc else "No description available"}
            result_dict.update(extract_protein_flags(clean_desc))
            
            return result_dict
            
    except requests.exceptions.Timeout:
        print(f"Warning: Connection to InterPro timed out for {family_id}.")
    except Exception as e:
        print(f"Warning: Could not fetch metadata for {family_id}: {e}")
    
    return {"description": "Not Found / Error", "is_membrane": False, "is_kinase": False}

def run_preprocess(MSA_file_path, tracker, family_name):
    """Step 1: Fetch data, calculate MSA stats, update JSON tracker."""
    print(f"--- Running Preprocessing for {family_name} ---")
    tracker.start_timer("Setup_and_MSA_Stats")
    
    # 1. Get MSA Stats
    msa_stats = get_msa_stats(MSA_file_path)
    for k, v in msa_stats.items():
        tracker.add_stat("msa_stats", k, v)
        
    # 2. Fetch external metadata (InterPro/Pfam)
    api_info = fetch_pfam_metadata(family_name)
    for k, v in api_info.items():
        tracker.add_stat("family_info", k, v)
        
    # Initialize placeholders for future steps
    tracker.add_stat("pipeline_stats", "num_significant_splits", None)
    
    tracker.stop_timer()
    print("Preprocessing complete.")

def run_find_splits(MSA_file_path, args, tracker, calc_dir, out_mode_dir):
    """Step 2: Run computations and evaluate splits."""
    print(f"--- Running Split Finding for {args.family} using {args.embedding} ---")
    tracker.start_timer("Calculations")

    # Define exact target paths first
    fasta_path = os.path.join(calc_dir, f"{args.family}.fasta")
    tree_path = os.path.join(calc_dir, f"{args.family}.tree")
    cov_path = os.path.join(calc_dir, f"{args.family}_cov_mat.csv") 
    cov_ordered_path = os.path.join(calc_dir, f"{args.family}_cov_mat_tree_ordered.csv")

    # 1. Fasta
    # Assuming convert_stockholm_to_fasta accepts output_path. 
    # If not, use: if temp_fasta != fasta_path: shutil.move(temp_fasta, fasta_path)
    temp_fasta = convert_stockholm_to_fasta(MSA_file_path)
    if temp_fasta != fasta_path: 
        shutil.move(temp_fasta, fasta_path)
    tracker.calc_and_add_sequence_similarity(fasta_path)
    
    # 2. Tree (Using the output_path argument your function already has!)
    run_fasttree(fasta_path, output_path=tree_path)
    tracker.calc_and_add_tree_stats(tree_path)
    
    # 3. Covariance 
    cov_path = tree_to_covariance_matrix(tree_path, output_path=cov_path)
    cov_ordered_path = order_covariance_matrix_by_tree(cov_path, tree_path)
    
    emb_out_dir = os.path.join(calc_dir, f'embeddings_{args.embedding}')
    norm_emb_path = create_normalized_mean_embeddings_matrix(fasta_path, mode=args.embedding, output_path=emb_out_dir)
    tracker.stop_timer()

    tracker.start_timer("Split_Evaluation")
    results, raw_splits_count, unique_splits_count, final_p_dim, sf_stats = evaluate_top_splits(
        tree_path, cov_ordered_path, norm_emb_path, 
        output_path=out_mode_dir, 
        calc_dir=calc_dir,
        fasta_path=fasta_path,
        k=args.nodes, 
        pca_min_variance=args.pca_var, 
        pca_min_components=args.pca_comp, 
        standardize=args.standardize,
        tree_alpha=args.alpha,
        existing_msa_stats=tracker.metadata.get("msa_stats", {})
    )
    
    tracker.add_stat("pipeline_stats", "num_raw_candidate_splits", raw_splits_count)
    tracker.add_stat("pipeline_stats", "num_unique_candidate_splits", unique_splits_count)

    for sf_name, stats in sf_stats.items():
        tracker.add_stat("msa_stats", f"{sf_name}_avg_sequence_similarity_pct", stats["avg_sequence_similarity_pct"])
        tracker.add_stat("msa_stats", f"{sf_name}_normalized_total_branch_length", stats["normalized_total_branch_length"])
    
    for sf_name, dim in final_p_dim.items():
      tracker.add_stat("pipeline_stats", f"{sf_name}_final_embedding_dim", dim)
    save_results_json(results, os.path.join(out_mode_dir, "results.json"))
    tracker.stop_timer()
    print("Split finding complete.")

def run_visualize(args, tracker, fasta_path_global, cov_ordered_path_global, out_mode_dir, calc_dir):
    """Step 3: Generate plots and analyze structures."""
    if not args.generate_plots:
        print("Skipping visualization (--generate_plots is False)")
        return
        
    print(f"--- Running Visualizations for {args.family} ---")
    tracker.start_timer("Visualization_and_Analysis")
    
    # --- VISUALIZATION 1: Global Macro View (Mean Shifts) ---
    subfamilies_summary_path = os.path.join(out_mode_dir, "subfamilies_summary.json")
    try:
        from significant_split_evaluation.visualisations import plot_global_subfamilies
        plot_global_subfamilies(cov_ordered_path_global, subfamilies_summary_path, out_mode_dir)
    except Exception as e:
        print(f"Warning: Could not plot global subfamilies: {e}")
    
    # --- VISUALIZATION 2: Local Micro View (Covariance Splits) ---
    total_sig_count = 0
    
    for sf_folder in os.listdir(out_mode_dir):
        if not sf_folder.startswith("subfamily_"):
            continue
            
        sf_dir = os.path.join(out_mode_dir, sf_folder)
        sig_splits_path = os.path.join(sf_dir, 'significant_splits')
        
        # FIX: Point to the cropped local assets in calc_dir, NOT out_mode_dir
        sf_idx = sf_folder.split("_")[1]
        calc_sf_dir = os.path.join(calc_dir, f"subfamily_{sf_idx}")
        local_fasta_path = os.path.join(calc_sf_dir, f"subfamily_{sf_idx}.fasta")
        local_cov_path = os.path.join(calc_sf_dir, f"subfamily_{sf_idx}_cov_mat.csv")
        
        if os.path.exists(sig_splits_path):
            splits = [d for d in os.listdir(sig_splits_path) if os.path.isdir(os.path.join(sig_splits_path, d))]
            total_sig_count += len(splits)
            
            for folder_name in splits:
                folder_path = os.path.join(sig_splits_path, folder_name)
                split_info = get_split_info(folder_path)
                
                print(f"Visualizing {sf_folder} -> {folder_name}...")
                
                visualize_split_msa_sorted(local_fasta_path, split_info, folder_path)
                plot_split_covariance(local_cov_path, split_info, folder_path)
                plot_side_by_side_embedding_covariance(folder_path, split_info)
                
                tm_stats = visualize_structures_pipeline(local_fasta_path, split_info, folder_path, local_cov_path, ss_predictor=args.ss_predictor)
                if tm_stats:
                    tracker.add_split_stat(f"{sf_folder}_{folder_name}", tm_stats)
                
                run_variance_analysis(folder_path)

    tracker.add_stat("pipeline_stats", "num_significant_covariance_splits", total_sig_count)
    tracker.stop_timer()
    print("Visualization complete.")

def main():
    parser = argparse.ArgumentParser(description="MSA to Split Modular Pipeline")
    
    # Core arguments matching your mentor's request
    parser.add_argument('--input', type=str, required=True, help="Path to the MSA file")
    parser.add_argument('--family', type=str, required=True, help="Family name (e.g., pf00228)")
    parser.add_argument('--operation', type=str, required=True, 
                        choices=['preprocess', 'find_best_split', 'visualize', 'full'], 
                        help="Which part of the pipeline to run")
    
    # Parameters
    parser.add_argument('--embedding', type=str, default="sequence", help="Embedding mode (e.g., sequence, structure)")
    parser.add_argument('--nodes', type=int, default=None, help="Number of nodes to evaluate")
    parser.add_argument('--pca_comp', type=int, default=None, help="PCA minimum components, if not entered and --pca_var not entered - no ppca is calculated")
    parser.add_argument('--pca_var', type=float, default=None, help="PCA minimum variance, if not entered and --pca_comp not entered - no ppca is calculated")
    parser.add_argument('--standardize', type=str, default="TRUE", choices=["TRUE", "FALSE"], help="Standardize data")
    parser.add_argument('--generate_plots', type=str, default="TRUE", choices=["TRUE", "FALSE"], help="Generate plots during visualization")
    parser.add_argument('--alpha', type=float, default=0.10, help="Minimum branch size & redundancy overlap threshold (e.g., 0.10 for 10%)")
    parser.add_argument('--ss_predictor', type=str, default="netsurfp", choices=["netsurfp", "esmfold"], 
                        help="Predictor to use for 2D secondary structure logos (default: netsurfp)")
    args = parser.parse_args()

    args.standardize = args.standardize == "TRUE"
    args.generate_plots = args.generate_plots == "TRUE"

    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(args.input))
    out_dir = os.path.join(base_dir, f"{args.family}_outputs")
    calc_dir = os.path.join(base_dir, f"{args.family}_calculations")
    out_mode_dir = os.path.join(out_dir, f"{args.embedding}_embeddings")
    
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_mode_dir, exist_ok=True)

    # Initialize Tracker ONCE per run. 
    # If the JSON exists from a previous step (like preprocess), it should load it and append to it.
    tracker_path = os.path.join(out_dir, "pipeline_metadata.json")
    tracker = MetadataTracker(tracker_path)
    tracker.add_stat("pipeline_stats", "family_name", args.family)
    tracker.add_stat("pipeline_stats", "embedding_mode", args.embedding)

    try:
        if args.operation in ['preprocess', 'full']:
            run_preprocess(args.input, tracker, args.family)
            
        if args.operation in ['find_best_split', 'full']:
            run_find_splits(args.input, args, tracker, calc_dir, out_mode_dir)
            
        if args.operation in ['visualize', 'full']:
            # Determine paths needed for visualization based on previous steps
            fasta_path = os.path.join(calc_dir, f"{args.family}.fasta")
            cov_ordered_path = os.path.join(calc_dir, f"{args.family}_cov_mat_tree_ordered.csv")
            run_visualize(args, tracker, fasta_path, cov_ordered_path, out_mode_dir, calc_dir)

    except Exception as e:
        print(f"!!! Pipeline Crashed !!! Error: {e}")
        tracker.add_stat("errors", "crash_reason", str(e))
        raise e
    finally:
        tracker.save()
        print(f"Metadata saved/updated at: {tracker.output_path}")

if __name__ == "__main__":
    main()
