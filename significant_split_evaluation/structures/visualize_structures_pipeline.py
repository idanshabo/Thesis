import os
import random
from Bio import SeqIO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def align_covariance_to_samples(cov_path, sample_list):
    """
    Loads the covariance matrix and re-indexes it to match the exact 
    list of proteins used in the structure prediction.
    """
    try:
        df_full = pd.read_csv(cov_path, index_col=0)
    except Exception as e:
        print(f"Error loading covariance matrix: {e}")
        return None

    aligned_ids = []
    df_index_set = set(df_full.index)
    df_index_map = {normalize_id(x): x for x in df_full.index}

    for sample_id in sample_list:
        if sample_id in df_index_set:
            aligned_ids.append(sample_id)
        elif normalize_id(sample_id) in df_index_map:
            aligned_ids.append(df_index_map[normalize_id(sample_id)])
        else:
            # If sample not in covariance matrix, skip
            pass

    if not aligned_ids:
        print("Warning: No overlap found between sampled structures and covariance matrix.")
        return None

    # Extract and Reorder: [Group A samples] -> [Group B samples]
    df_filtered = df_full.loc[aligned_ids, aligned_ids]
    return df_filtered


def plot_combined_panel(ordered_cov_path, df_tm, sample_list, split_point, output_folder, filename="combined_analysis.png"):
    """
    Plots filtered Covariance (Left) and TM-Score (Right) side-by-side.
    """
    # 1. Prepare Covariance Data (Aligned)
    df_cov_aligned = align_covariance_to_samples(ordered_cov_path, sample_list)

    if df_cov_aligned is None:
        print("Skipping combined plot: Covariance alignment failed.")
        return

    # 2. Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # ==========================================
    # LEFT PANEL: Covariance (Filtered)
    # ==========================================
    sns.heatmap(df_cov_aligned, cmap='viridis', cbar=True, 
                xticklabels=False, yticklabels=False, square=True, ax=ax1)
    
    # Draw Crosshair
    ax1.axvline(x=split_point, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=split_point, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Labels
    ax1.text(split_point/2, -0.5, "Group A", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.text(-0.5, split_point/2, "Group A", ha='right', va='center', fontsize=11, fontweight='bold')
    
    center_b = split_point + ((len(df_cov_aligned) - split_point) / 2)
    ax1.text(center_b, -0.5, "Group B", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.text(-0.5, center_b, "Group B", ha='right', va='center', fontsize=11, fontweight='bold')

    ax1.set_title(f"Covariance Signal\n(Filtered to {len(df_cov_aligned)} sampled IDs)", fontsize=14, pad=10)

    # ==========================================
    # RIGHT PANEL: TM Score
    # ==========================================
    if df_tm is not None:
        sns.heatmap(df_tm, cmap='RdYlBu_r', vmin=0, vmax=1.0, cbar=True,
                    xticklabels=False, yticklabels=False, square=True, ax=ax2)
        
        # Draw Crosshair
        ax2.axvline(x=split_point, color='black', linestyle='--', linewidth=1.5)
        ax2.axhline(y=split_point, color='black', linestyle='--', linewidth=1.5)
        
        # Labels
        ax2.text(split_point/2, -0.5, "Group A", ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax2.text(-0.5, split_point/2, "Group A", ha='right', va='center', fontsize=11, fontweight='bold')
        
        ax2.text(center_b, -0.5, "Group B", ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax2.text(-0.5, center_b, "Group B", ha='right', va='center', fontsize=11, fontweight='bold')

        ax2.set_title(f"Predicted Structural Similarity (TM-Score)\n(ESMFold Prediction)", fontsize=14, pad=10)
    else:
        ax2.text(0.5, 0.5, "TM Data Missing", ha='center', va='center')

    plt.tight_layout()
    final_path = os.path.join(output_folder, filename)
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"Combined plot saved to: {final_path}")


def visualize_structures_pipeline(fasta_path, split_data, sig_split_folder, ordered_cov_path):
    """
    Orchestrates generation of two structure analyses:
    1. Predicted Structures (Side-by-Side with Covariance)
    2. Experimental Structures (Standard Heatmap)
    """
    base_output = os.path.join(os.path.dirname(fasta_path), 'structures')
    
    dir_predicted = os.path.join(base_output, 'predicted_esm')
    dir_experimental = os.path.join(base_output, 'experimental_pdb')
    plot_folder = os.path.join(sig_split_folder, "visualization")
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # ---------------------------------------------------------
    # PART 1: PREDICTED STRUCTURES (ESMFold) - ALWAYS RUN
    # ---------------------------------------------------------
    print("\n=== PART 1: Predicted Structures (ESMFold) ===")
    
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print("Error: FASTA not found.")
        return

    # Sampling Logic
    id_map = {normalize_id(r.id): r.id for r in records}
    
    def get_valid_ids(input_list):
        valid = []
        for item in input_list:
            if item in id_map.values(): valid.append(item)
            elif normalize_id(item) in id_map: valid.append(id_map[normalize_id(item)])
        return valid

    valid_a = get_valid_ids(split_data['group_a'])
    valid_b = get_valid_ids(split_data['group_b'])
    
    # Sample 50 from each
    sample_a = random.sample(valid_a, min(len(valid_a), 50))
    sample_b = random.sample(valid_b, min(len(valid_b), 50))
    
    # Combined list for calculation (Order matters: A then B)
    processing_list = sample_a + sample_b
    
    # 1. Predict
    run_prediction_batch(records, dir_predicted, allow_list=processing_list)
    
    # 2. Matrix
    print("Generating Predicted Heatmap...")
    df_pred, stats_pred, split_pred = calculate_tm_matrix(sample_a, sample_b, dir_predicted)
    
    # 3. PLOT SIDE-BY-SIDE (New Function)
    if df_pred is not None:
        split_point = len(sample_a)
        plot_combined_panel(
            ordered_cov_path=ordered_cov_path,
            df_tm=df_pred,
            sample_list=processing_list, 
            split_point=split_point,
            output_folder=plot_folder,
            filename="combined_covariance_tm_prediction.png"
        )

    # ---------------------------------------------------------
    # PART 2: REAL STRUCTURES (Experimental) - CONDITIONAL
    # ---------------------------------------------------------
    # RESTORED BLOCK
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
            # We use the ORIGINAL plotter here because experimental data 
            # is sparse and IDs often don't match the covariance matrix 1:1.
            plot_tm_heatmap(df_exp, stats_exp, split_exp, plot_folder, filename="tm_score_EXPERIMENTAL.png")
    else:
        print("Skipping Experimental Plot (insufficient PDB coverage).")
