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
from significant_split_evaluation.structures.visualize_representative_structure import get_group_representative, align_and_visualize_pair


def normalize_id(identifier):
    """
    Standardizes IDs by replacing typical problem characters.
    Seq/1 -> Seq_1
    """
    return identifier.replace("/", "_")


def get_intervals_from_index(index, group_a_ids, group_b_ids):
    """
    Scans the dataframe index and identifies contiguous blocks of groups.
    Returns a list of tuples: (Label, Start_Index, End_Index)
    """
    # Pre-process sets for fast lookup
    set_a = {normalize_id(x) for x in group_a_ids}
    set_b = {normalize_id(x) for x in group_b_ids}
    
    intervals = []
    if len(index) == 0:
        return intervals

    # Determine first label
    first_id = normalize_id(index[0])
    current_label = "A" if first_id in set_a else ("B" if first_id in set_b else "?")
    start_idx = 0

    for i, uid in enumerate(index):
        nid = normalize_id(uid)
        label = "A" if nid in set_a else ("B" if nid in set_b else "?")
        
        # If label changes (and it's not the very first item)
        if label != current_label:
            intervals.append((current_label, start_idx, i))
            current_label = label
            start_idx = i
            
    # Append the final block
    intervals.append((current_label, start_idx, len(index)))
    return intervals


def get_aligned_matrices(cov_path, tm_df, sample_list, sort_by="groups"):
    """
    Returns aligned Covariance and TM dataframes with double-sided normalization.
    """
    # 1. Load Full Covariance
    try:
        df_cov_full = pd.read_csv(cov_path, index_col=0)
    except Exception as e:
        print(f"Error loading covariance: {e}")
        return None, None

    # 2. Map Covariance IDs
    cov_map = {normalize_id(x): x for x in df_cov_full.index}
    
    # 3. Map TM Matrix IDs
    tm_map = {normalize_id(x): x for x in tm_df.columns}

    # 4. Determine Target Order
    target_order_norm = [] 
    
    if sort_by == "groups":
        # Order: Group A list then Group B list
        for sample_id in sample_list:
            norm_sample = normalize_id(sample_id)
            if norm_sample in cov_map and norm_sample in tm_map:
                target_order_norm.append(norm_sample)
                
    elif sort_by == "covariance":
        # Order: Based on the Covariance CSV row order (Spectral Order)
        sample_set_norm = {normalize_id(s) for s in sample_list}
        for cov_idx in df_cov_full.index:
            norm_cov = normalize_id(cov_idx)
            if norm_cov in sample_set_norm and norm_cov in tm_map:
                target_order_norm.append(norm_cov)

    if not target_order_norm:
        print(f"CRITICAL WARNING: No overlapping IDs found after normalization.")
        return None, None

    # 5. Extract
    cov_keys = [cov_map[norm] for norm in target_order_norm]
    tm_keys  = [tm_map[norm]  for norm in target_order_norm]

    df_cov_aligned = df_cov_full.loc[cov_keys, cov_keys]
    df_tm_aligned = tm_df.loc[tm_keys, tm_keys]
    
    return df_cov_aligned, df_tm_aligned
    

def plot_side_by_side_dynamic(df_cov, df_tm, group_a_ids, group_b_ids, output_path, title_suffix=""):
    """
    Plots two matrices. 
    Dynamically scans the index to draw dashed lines whenever the Group changes.
    """
    if df_cov.empty or df_tm.empty:
        print(f"Skipping plot {title_suffix}: Dataframe is empty.")
        return

    # 1. Calculate Intervals (Blocks)
    # Both DFs have the same index order now, so we can use df_cov.index
    intervals = get_intervals_from_index(df_cov.index, group_a_ids, group_b_ids)

    # 2. Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # --- LEFT: COVARIANCE ---
    sns.heatmap(df_cov, cmap='viridis', cbar=True, 
                xticklabels=False, yticklabels=False, square=True, ax=ax1)
    ax1.set_title(f"Covariance Signal\n{title_suffix}", fontsize=14, pad=10)

    # --- RIGHT: TM SCORE ---
    if df_tm is not None:
        sns.heatmap(df_tm, cmap='RdYlBu_r', vmin=0, vmax=1.0, cbar=True,
                    xticklabels=False, yticklabels=False, square=True, ax=ax2)
        ax2.set_title(f"Structural Similarity (TM)\n{title_suffix}", fontsize=14, pad=10)

    # --- 3. DRAW LINES AND LABELS ---
    # We iterate over the intervals we calculated
    for label, start, end in intervals:
        
        # Draw Line at the END of the block (unless it's the very last block)
        if end < len(df_cov):
            for ax, color in [(ax1, 'white'), (ax2, 'black')]:
                ax.axvline(x=end, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
                ax.axhline(y=end, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

        # Add Labels (Only if block is big enough to be legible > 2% of map)
        size = end - start
        if size > (len(df_cov) * 0.02):
            center = start + (size / 2)
            group_name = "Group A" if label == "A" else ("Group B" if label == "B" else "?")
            
            for ax, color in [(ax1, 'white'), (ax2, 'black')]:
                # Side Label (Left Axis)
                ax.text(-0.5, center, group_name, ha='right', va='center', 
                        fontsize=10, fontweight='bold', color='black') # Keep text black for readability outside plot
                
                # Bottom Label (X Axis)
                ax.text(center, len(df_cov) + 0.5, group_name, ha='center', va='top', 
                        fontsize=10, fontweight='bold', color='black', rotation=45)

            # --- Calculate Stats for this block (Diagonal) on TM Plot ---
            if df_tm is not None:
                # Extract the square block [start:end, start:end]
                block_val = df_tm.iloc[start:end, start:end].values.mean()
                if not pd.isna(block_val):
                    ax2.text(center, center, f"{block_val:.2f}", 
                             ha='center', va='center', fontsize=11, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

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
        
    if df_pred is not None and not df_pred.empty:
        print("\n=== Generating Representative Alignment ===")
        
        # 1. Select the "Centroid" representative for each group
        # Note: These IDs come from the dataframe index/columns
        rep_a_id = get_group_representative(df_pred, sample_a)
        rep_b_id = get_group_representative(df_pred, sample_b)
        
        if rep_a_id and rep_b_id:
            # 2. Construct Paths using normalize_id
            # This ensures '0A967AWS5_9FLAO/1-48' becomes '0A967AWS5_9FLAO_1-48'
            norm_a_id = normalize_id(rep_a_id)
            norm_b_id = normalize_id(rep_b_id)
            
            pdb_a = os.path.join(dir_predicted, f"{norm_a_id}.pdb")
            pdb_b = os.path.join(dir_predicted, f"{norm_b_id}.pdb")
            
            # 3. Define Output Path
            align_output = os.path.join(sig_split_folder, "representative_structural_alignment")
            
            # 4. Run Visualization
            if os.path.exists(pdb_a) and os.path.exists(pdb_b):
                align_and_visualize_pair(
                    pdb_a, 
                    pdb_b, 
                    align_output,
                    label_a=f"Group A (Rep: {rep_a_id})", 
                    label_b=f"Group B (Rep: {rep_b_id})"
                )
            else:
                print(f"Could not find PDB files:\nExpected: {pdb_a}\nExpected: {pdb_b}")

    # --- PART 2: PLOT 1 - ORDERED BY GROUPS ---
    print("Generating Plot 1: Ordered by Groups...")
    
    cov_grp, tm_grp = get_aligned_matrices(ordered_cov_path, df_pred, processing_list, sort_by="groups")
    
    if cov_grp is not None:
        plot_side_by_side_dynamic(
            df_cov=cov_grp, 
            df_tm=tm_grp,
            group_a_ids=sample_a,
            group_b_ids=sample_b,
            output_path=os.path.join(sig_split_folder, "combined_ordered_by_groups.png"),
            title_suffix="(Ordered by Split Group)"
        )

    # --- PART 3: PLOT 2 - ORDERED BY COVARIANCE ---
    print("Generating Plot 2: Ordered by Covariance...")
    
    cov_ord, tm_ord = get_aligned_matrices(ordered_cov_path, df_pred, processing_list, sort_by="covariance")
    
    if cov_ord is not None:
        plot_side_by_side_dynamic(
            df_cov=cov_ord, 
            df_tm=tm_ord,
            group_a_ids=sample_a,
            group_b_ids=sample_b,
            output_path=os.path.join(sig_split_folder, "combined_ordered_by_covariance.png"),
            title_suffix="(Ordered by Covariance Index)"
        )

    # --- PART 4: EXPERIMENTAL ---
    print("\n=== Experimental Structures Check ===")
    success, exp_a_ids, exp_b_ids = prepare_experimental_folder(split_data['group_a'], split_data['group_b'], dir_experimental)
    if success:
        df_exp, stats_exp, split_exp = calculate_tm_matrix(exp_a_ids, exp_b_ids, dir_experimental)
        if df_exp is not None and not df_exp.empty:
            plot_tm_heatmap(df_exp, stats_exp, split_exp, sig_split_folder, filename="tm_score_EXPERIMENTAL.png")
