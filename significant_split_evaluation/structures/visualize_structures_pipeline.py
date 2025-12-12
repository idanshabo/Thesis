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
    Includes robust checks for empty intersections.
    """
    # 1. Load Full Covariance
    try:
        df_cov_full = pd.read_csv(cov_path, index_col=0)
    except Exception as e:
        print(f"Error loading covariance: {e}")
        return None, None

    # 2. Create Mapping
    # Map normalized_id -> original_id_in_cov_file
    df_index_map = {normalize_id(str(x)): x for x in df_cov_full.index}
    df_index_set = set(df_cov_full.index)

    # 3. Determine the Order of IDs (Covariance Side)
    final_order = []
    
    if sort_by == "groups":
        # ORDER: Group A then Group B
        for sample_id in sample_list:
            if sample_id in df_index_set:
                final_order.append(sample_id)
            elif normalize_id(sample_id) in df_index_map:
                final_order.append(df_index_map[normalize_id(sample_id)])
                
    elif sort_by == "covariance":
        # ORDER: Covariance Matrix Order
        sample_set_norm = {normalize_id(str(s)) for s in sample_list}
        for cov_id in df_cov_full.index:
            if normalize_id(str(cov_id)) in sample_set_norm:
                final_order.append(cov_id)

    if not final_order:
        print("Warning: No overlap found between sample list and covariance matrix.")
        return None, None

    # 4. Map to TM DataFrame Columns
    # The TM DF usually has normalized IDs as columns.
    tm_keys = []
    valid_final_order = []
    
    # Debug: Check one ID if we fail later
    tm_cols = set(tm_df.columns)
    
    for cov_id in final_order:
        norm_id = normalize_id(str(cov_id))
        
        # Try direct match or normalized match in TM columns
        if cov_id in tm_cols:
            tm_keys.append(cov_id)
            valid_final_order.append(cov_id)
        elif norm_id in tm_cols:
            tm_keys.append(norm_id)
            valid_final_order.append(cov_id)

    # 5. Safety Check for Empty Result
    if not valid_final_order:
        print(f"CRITICAL WARNING: Found {len(final_order)} IDs in Covariance, but matched 0 in TM Matrix.")
        print(f"  Example Cov ID: {final_order[0]}")
        print(f"  Example TM Col: {list(tm_cols)[0] if len(tm_cols)>0 else 'Empty'}")
        return None, None

    # 6. Filter and Return
    df_cov_aligned = df_cov_full.loc[valid_final_order, valid_final_order]
    df_tm_aligned = tm_df.loc[tm_keys, tm_keys]
    
    return df_cov_aligned, df_tm_aligned


def plot_side_by_side(df_cov, df_tm, split_point, output_path, title_suffix=""):
    """
    Generic plotter for two matrices side-by-side.
    """
    # Safety Check against the "zero-size array" error
    if df_cov.empty or df_tm.empty:
        print(f"Skipping plot {title_suffix}: Dataframe is empty.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # --- LEFT: COVARIANCE ---
    try:
        sns.heatmap(df_cov, cmap='viridis', cbar=True, 
                    xticklabels=False, yticklabels=False, square=True, ax=ax1)
        ax1.set_title(f"Covariance Signal\n{title_suffix}", fontsize=14, pad=10)
    except ValueError as e:
        print(f"Error plotting Covariance: {e}")
        return

    # --- RIGHT: TM SCORE ---
    if df_tm is not None:
        sns.heatmap(df_tm, cmap='RdYlBu_r', vmin=0, vmax=1.0, cbar=True,
                    xticklabels=False, yticklabels=False, square=True, ax=ax2)
        ax2.set_title(f"TM-Score Structure\n{title_suffix}", fontsize=14, pad=10)
    
    # --- OVERLAYS ---
    if split_point is not None and df_tm is not None:
        # Validate split point isn't larger than data
        actual_split = min(split_point, len(df_tm))
        
        # Calc Stats
        if len(df_tm) > 0:
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
            # Check for NaN (if a block is empty)
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
    Main Pipeline:
    1. Runs prediction.
    2. Generates Plot 1 (Group Order) and Plot 2 (Covariance Order).
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
        print("TM Matrix calculation failed or returned empty. Exiting.")
        return

    # --- PART 2: PLOT 1 - ORDERED BY GROUPS ---
    print("Generating Plot 1: Ordered by Groups...")
    
    cov_grp, tm_grp = get_aligned_matrices(ordered_cov_path, df_pred, processing_list, sort_by="groups")
    
    if cov_grp is not None and not cov_grp.empty:
        plot_side_by_side(
            df_cov=cov_grp, 
            df_tm=tm_grp,
            split_point=len(sample_a),
            output_path=os.path.join(plot_folder, "combined_ordered_by_groups.png"),
            title_suffix="(Ordered by Split Group)"
        )
    else:
        print("Skipping Plot 1 (Empty Data)")

    # --- PART 3: PLOT 2 - ORDERED BY COVARIANCE ---
    print("Generating Plot 2: Ordered by Covariance...")
    
    cov_ord, tm_ord = get_aligned_matrices(ordered_cov_path, df_pred, processing_list, sort_by="covariance")
    
    if cov_ord is not None and not cov_ord.empty:
        plot_side_by_side(
            df_cov=cov_ord, 
            df_tm=tm_ord,
            split_point=None, 
            output_path=os.path.join(plot_folder, "combined_ordered_by_covariance.png"),
            title_suffix="(Ordered by Covariance Index)"
        )
    else:
        print("Skipping Plot 2 (Empty Data)")

    # --- PART 4: EXPERIMENTAL ---
    print("\n=== Experimental Structures Check ===")
    success, exp_a_ids, exp_b_ids = prepare_experimental_folder(split_data['group_a'], split_data['group_b'], dir_experimental)
    if success:
        df_exp, stats_exp, split_exp = calculate_tm_matrix(exp_a_ids, exp_b_ids, dir_experimental)
        if df_exp is not None and not df_exp.empty:
            plot_tm_heatmap(df_exp, stats_exp, split_exp, plot_folder, filename="tm_score_EXPERIMENTAL.png")
