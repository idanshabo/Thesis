import os
import random
from Bio import SeqIO

# Import the local modules
from significant_split_evaluation.structures.structure_predictor import run_prediction_batch
from significant_split_evaluation.structures.structure_analysis import calculate_tm_matrix
from significant_split_evaluation.structures.visualization import plot_tm_heatmap
from significant_split_evaluation.structures.structures_from_experiments import get_pdb_from_uniprot, select_best_pdb, prepare_experimental_folder


def normalize_id(identifier):
    """
    Standardizes IDs by replacing typical problem characters.
    Seq/1 -> Seq_1
    """
    return identifier.replace("/", "_")


def visualize_structures_pipeline(fasta_path, split_data, sig_split_folder):
    """
    Orchestrates generation of two plots:
    1. Predicted Structures (Always)
    2. Experimental Structures (Conditional)
    """
    base_output = os.path.join(os.path.dirname(fasta_path), 'structures')
    
    # Define separate folders to keep data clean
    dir_predicted = os.path.join(base_output, 'predicted_esm')
    dir_experimental = os.path.join(base_output, 'experimental_pdb')
    plot_folder = os.path.join(sig_split_folder, "visualization")
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # ---------------------------------------------------------
    # PART 1: PREDICTED STRUCTURES (ESMFold) - ALWAYS RUN
    # ---------------------------------------------------------
    print("\n=== PART 1: Predicted Structures (ESMFold) ===")
    
    # Load FASTA
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print("Error: FASTA not found.")
        return

    # Sampling Logic (Same as before)
    id_map = {normalize_id(r.id): r.id for r in records}
    
    def get_valid_ids(input_list):
        valid = []
        for item in input_list:
            if item in id_map.values(): valid.append(item)
            elif normalize_id(item) in id_map: valid.append(id_map[normalize_id(item)])
        return valid

    valid_a = get_valid_ids(split_data['group_a'])
    valid_b = get_valid_ids(split_data['group_b'])
    
    # Process Strategy
    if len(records) <= 200:
        processing_list = list(id_map.values())
        analysis_a, analysis_b = valid_a, valid_b
    else:
        # Sample 50 from each
        sample_a = random.sample(valid_a, min(len(valid_a), 50))
        sample_b = random.sample(valid_b, min(len(valid_b), 50))
        processing_list = sample_a + sample_b
        analysis_a, analysis_b = sample_a, sample_b

    # 1. Predict
    run_prediction_batch(records, dir_predicted, allow_list=processing_list)
    
    # 2. Matrix & Plot
    print("Generating Predicted Heatmap...")
    df_pred, stats_pred, split_pred = calculate_tm_matrix(analysis_a, analysis_b, dir_predicted)
    
    if df_pred is not None:
        plot_tm_heatmap(df_pred, stats_pred, split_pred, plot_folder, filename="tm_score_PREDICTED.png")

    # ---------------------------------------------------------
    # PART 2: REAL STRUCTURES (Experimental) - CONDITIONAL
    # ---------------------------------------------------------
    print("\n=== PART 2: Experimental Structures (PDB) ===")
    
    # 1. Check coverage and download
    success, exp_a_ids, exp_b_ids = prepare_experimental_folder(
        split_data['group_a'], 
        split_data['group_b'], 
        dir_experimental
    )
    
    if success:
        # 2. Matrix & Plot (Only if we have enough data)
        print("Generating Experimental Heatmap...")
        df_exp, stats_exp, split_exp = calculate_tm_matrix(exp_a_ids, exp_b_ids, dir_experimental)
        
        if df_exp is not None:
            plot_tm_heatmap(df_exp, stats_exp, split_exp, plot_folder, filename="tm_score_EXPERIMENTAL.png")
    else:
        print("Skipping Experimental Plot (insufficient PDB coverage).")
