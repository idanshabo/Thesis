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


def get_aligned_matrices(cov_path, tm_df, sample_list, sort_by="groups"):
    """
    Returns aligned Covariance and TM dataframes.
    Includes DOUBLE-SIDED normalization to handle 'Seq/1' vs 'Seq_1' mismatches.
    """
    # 1. Load Full Covariance
    try:
        df_cov_full = pd.read_csv(cov_path, index_col=0)
    except Exception as e:
        print(f"Error loading covariance: {e}")
        return None, None

    # 2. Map Covariance IDs: Normalized -> Original
    cov_map = {normalize_id(x): x for x in df_cov_full.index}
    
    # 3. Map TM Matrix IDs: Normalized -> Original (THE FIX)
    # This allows us to look up 'Seq/1' using key 'Seq_1'
    tm_map = {normalize_id(x): x for x in tm_df.columns}

    # 4. Determine the Target Order of IDs
    target_order_norm = [] # We store NORMALIZED IDs here for consistency
    
    if sort_by == "groups":
        # Order: Group A list then Group B list
        for sample_id in sample_list:
            norm_sample = normalize_id(sample_id)
            # Only keep if it exists in BOTH matrices
            if norm_sample in cov_map and norm_sample in tm_map:
                target_order_norm.append(norm_sample)
                
    elif sort_by == "covariance":
        # Order: Based on the Covariance CSV row order
        # We first identify which of our samples exist in the Cov matrix
        sample_set_norm = {normalize_id(s) for s in sample_list}
        
        # Iterate through the Covariance Index (preserving its spectral order)
        for cov_idx in df_cov_full.index:
            norm_cov = normalize_id(cov_idx)
            # If this Cov row corresponds to one of our samples AND exists in TM matrix
            if norm_cov in sample_set_norm and norm_cov in tm_map:
                target_order_norm.append(norm_cov)

    if not target_order_norm:
        print(f"CRITICAL WARNING: No overlapping IDs found after normalization.")
        print(f"  Sample Norm: {normalize_id(sample_list[0])}")
        print(f"  Cov keys ex: {list(cov_map.keys())[:2]}")
        print(f"  TM keys ex:  {list(tm_map.keys())[:2]}")
        return None, None

    # 5. Build the Final Aligned Lists
    # We use the maps to get the original keys for each dataframe
    cov_keys = [cov_map[norm] for norm in target_order_norm]
    tm_keys  = [tm_map[norm]  for norm in target_order_norm]

    # 6. Extract and Return
    # loc[rows, cols]
    df_cov_aligned = df_cov_full.loc[cov_keys, cov_keys]
    df_tm_aligned = tm_df.loc[tm_keys, tm_keys]
    
    return df_cov_aligned, df_tm_aligned
    

def plot_side_by_side(df_cov, df_tm, split_point, output_path, title_suffix=""):
    """
    Generic plotter for two matrices side-by-side.
    """
    if df_cov.empty or df_tm.empty:
        print(f"Skipping plot {title_suffix}: Dataframe is empty.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # --- LEFT: COVARIANCE ---
    sns.heatmap(df_cov, cmap='viridis', cbar=True, 
                xticklabels=False, yticklabels=False, square=True, ax=ax1)
    ax1.set_title(f"Covariance Signal\n{title_suffix}", fontsize=14, pad=10)

    # --- RIGHT: TM SCORE ---
    if df_tm is not None:
        sns.heatmap(df_tm, cmap='RdYlBu_r', vmin=0, vmax=1.0, cbar=True,
                    xticklabels=False, yticklabels=False, square=True, ax=ax2)
        ax2.set_title(f"TM-Score Structure\n{title_suffix}", fontsize=14, pad=10)
    
    # --- OVERLAYS ---
    if split_point is not None:
        # Validate split point isn't larger than data
        # Note: If we filtered some IDs out due to mismatch, split_point might be off.
        # Ideally, we recalculate split_point based on how many 'Group A' survived.
        # For now, we use a safe clamp.
        actual_split = min(split_point, len(df_tm))
        
        # Calc Stats
        avg_a = df_tm.iloc[:actual_split, :actual_split].values.mean()
        avg_b = df_tm.iloc[actual_split:, actual_split:].values.mean()
        avg_inter = df_tm.iloc[:actual_split, actual_split:].values.mean()
        
        # Draw Lines
        for ax, color in [(ax1, 'white'), (ax2, 'black')]:
            ax.axvline(x=actual_split, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axhline(y=actual_split, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            
            # Labels
            center_b = actual_split + ((len(df_tm) - actual_split) / 2)
            ax.text(actual_split/2, -0.5, "Group A", ha='center', va='bottom', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(-0.5, actual_split/2, "Group A", ha='right', va='center', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(center_b, -0.5, "Group B", ha='center', va='bottom', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(-0.5, center_b, "Group B", ha='right', va='center', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')

        # Stats Text (Right Panel)
        if not pd.isna(avg_a):
            ax2.text(actual_split/2, actual_split/2, f"Avg: {avg_a:.2f}", 
                     ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        if not pd.isna(avg_b):
            center_b = actual_split + ((len(df_tm) - actual_split) / 2)
            ax2.text(center_b, center_b, f"Avg: {avg_b:.2f}", 
                     ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        if not pd.isna(avg_inter):
            ax2.text(center_b, actual_split/2, f"Inter: {avg_inter:.2f}", 
                     ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def visualize_structures_pipeline(fasta_path, split_data, sig_split_folder, ordered_cov_path):
    """
    Main Pipeline
    """
    base_output = os.path.join(os.path.dirname(fasta_path), 'structures')
    dir_predicted = os.path.join(base_output, 'predicted_esm')
    dir_experimental = os.path.join(base_output, 'experimental_pdb')
    plot_folder = os.path.join(sig_split_folder, "visualization")
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # --- PART 1: PREPARATION ---
    print("\n=== Running Structural Pipeline ===")
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print("Error: FASTA not found.")
        return

    id_map = {normalize_id(r.id): r.id for r in records}
    
    def get_valid_ids(input_list):
        valid = []
        for item in input_list:
            if item in id_map.values(): valid.append(item)
            elif normalize_id(item) in id_map: valid.append(id_map[normalize_id(item)])
        return valid

    valid_a = get_valid_ids(split_data['group_a'])
    valid_b = get_valid_ids(split_data['group_b'])
    
    # Sample
    sample_a = random.sample(valid_a, min(len(valid_a), 50))
    sample_b = random.sample(valid_b, min(len(valid_b), 50))
    processing_list = sample_a + sample_b
    
    # Predict
    run_prediction_batch(records, dir_predicted, allow_list=processing_list)
    
    # TM Matrix
    print("Calculating TM Matrix...")
    df_pred, stats_pred, split_pred = calculate_tm_matrix(sample_a, sample_b, dir_predicted)
    
    if df_pred is None or df_pred.empty:
        print("TM Matrix calculation failed. Exiting.")
        return

    # --- PART 2: PLOT 1 - ORDERED BY GROUPS ---
    print("Generating Plot 1: Ordered by Groups...")
    
    cov_grp, tm_grp = get_aligned_matrices(ordered_cov_path, df_pred, processing_list, sort_by="groups")
    
    # We need to recalculate the visual split point just in case some IDs were dropped
    # The split point is how many 'Group A' proteins survived the intersection
    if cov_grp is not None:
        # Count how many of our original 'sample_a' are in the final aligned matrix
        norm_keys = [normalize_id(k) for k in cov_grp.index]
        set_a_norm = {normalize_id(k) for k in sample_a}
        
        # Count intersection to place the line correctly
        real_split_point = sum(1 for k in norm_keys if k in set_a_norm)
        
        plot_side_by_side(
            df_cov=cov_grp, 
            df_tm=tm_grp,
            split_point=real_split_point,
            output_path=os.path.join(plot_folder, "combined_ordered_by_groups.png"),
            title_suffix="(Ordered by Split Group)"
        )

    # --- PART 3: PLOT 2 - ORDERED BY COVARIANCE ---
    print("Generating Plot 2: Ordered by Covariance...")
    
    cov_ord, tm_ord = get_aligned_matrices(ordered_cov_path, df_pred, processing_list, sort_by="covariance")
    
    if cov_ord is not None:
        plot_side_by_side(
            df_cov=cov_ord, 
            df_tm=tm_ord,
            split_point=None, 
            output_path=os.path.join(plot_folder, "combined_ordered_by_covariance.png"),
            title_suffix="(Ordered by Covariance Index)"
        )

    # --- PART 4: EXPERIMENTAL ---
    print("\n=== Experimental Structures Check ===")
    success, exp_a_ids, exp_b_ids = prepare_experimental_folder(split_data['group_a'], split_data['group_b'], dir_experimental)
    if success:
        df_exp, stats_exp, split_exp = calculate_tm_matrix(exp_a_ids, exp_b_ids, dir_experimental)
        if df_exp is not None and not df_exp.empty:
            plot_tm_heatmap(df_exp, stats_exp, split_exp, plot_folder, filename="tm_score_EXPERIMENTAL.png")
