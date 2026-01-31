import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logomaker
from Bio import SeqIO

# Import the local modules
from significant_split_evaluation.structures.structure_predictor import run_prediction_batch
from significant_split_evaluation.structures.structure_analysis import calculate_tm_matrix
from significant_split_evaluation.structures.visualization import plot_tm_heatmap
from significant_split_evaluation.structures.visualize_representative_structure import get_group_representative, align_and_visualize_pair
from significant_split_evaluation.structures.structure_from_experiments_2 import prepare_global_structure_map, check_and_download_structures


def normalize_id(identifier):
    """
    Standardizes IDs by replacing typical problem characters.
    Seq/1 -> Seq_1
    """
    return identifier.replace("/", "_")


def align_matrices(df1, df2):
    """
    Ensures both DataFrames have the same columns (amino acids)
    and the same index (positions). Missing columns are filled with 0.
    """
    # Get union of all characters found in both groups
    all_chars = sorted(list(set(df1.columns) | set(df2.columns)))

    # Reindex both to this full set of characters
    df1_aligned = df1.reindex(columns=all_chars, fill_value=0)
    df2_aligned = df2.reindex(columns=all_chars, fill_value=0)

    return df1_aligned, df2_aligned


def generate_comparative_logos(records, group_a_ids, group_b_ids, output_path, highlight_threshold=0.6):
    """
    Generates a stacked sequence logo plot comparing Group A and Group B,
    highlighting divergent positions.
    """
    print("--- Generating Scaled Logos with Highlights ---")
    
    # 1. Extract Sequences
    # Create a quick lookup map for records
    seq_map = {r.id: str(r.seq).upper() for r in records}
    # Add normalized keys to map just in case
    for r in records:
        seq_map[normalize_id(r.id)] = str(r.seq).upper()

    seqs_a = []
    seqs_b = []

    # Helper to find seq
    def get_seq(uid):
        if uid in seq_map: return seq_map[uid]
        if normalize_id(uid) in seq_map: return seq_map[normalize_id(uid)]
        return None

    for uid in group_a_ids:
        s = get_seq(uid)
        if s: seqs_a.append(s)
    
    for uid in group_b_ids:
        s = get_seq(uid)
        if s: seqs_b.append(s)

    if not seqs_a or not seqs_b:
        print("Error: Empty groups for Logo plotting.")
        return

    # 2. Create Count Matrices
    try:
        counts_a = logomaker.alignment_to_matrix(seqs_a)
        counts_b = logomaker.alignment_to_matrix(seqs_b)

        # 3. Align Matrices
        counts_a, counts_b = align_matrices(counts_a, counts_b)

        # 4. Convert to Information (Bits)
        info_a = logomaker.transform_matrix(counts_a, from_type='counts', to_type='information')
        info_b = logomaker.transform_matrix(counts_b, from_type='counts', to_type='information')

        # 5. Determine Global Y-Axis Max
        max_a = info_a.sum(axis=1).max()
        max_b = info_b.sum(axis=1).max()
        global_max = max(max_a, max_b) * 1.1 

        # 6. Calculate Differences for Highlighting
        prob_a = logomaker.transform_matrix(counts_a, from_type='counts', to_type='probability')
        prob_b = logomaker.transform_matrix(counts_b, from_type='counts', to_type='probability')
        
        # Simple difference score
        diff_score = np.sum(np.abs(prob_a - prob_b), axis=1)
        divergent_positions = diff_score[diff_score > highlight_threshold].index.tolist()

        # 7. Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

        logomaker.Logo(info_a, ax=ax1, color_scheme='chemistry')
        logomaker.Logo(info_b, ax=ax2, color_scheme='chemistry')

        ax1.set_ylim(0, global_max)
        ax2.set_ylim(0, global_max)

        ax1.set_title(f"Group A ({len(seqs_a)} sequences)", fontsize=14, fontweight='bold')
        ax2.set_title(f"Group B ({len(seqs_b)} sequences)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Bits")
        ax2.set_ylabel("Bits")

        # Add Highlights
        for pos in divergent_positions:
            ax1.axvspan(pos - 0.5, pos + 0.5, color='red', alpha=0.2, zorder=0)
            ax2.axvspan(pos - 0.5, pos + 0.5, color='red', alpha=0.2, zorder=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved Logo Plot: {output_path}")

    except Exception as e:
        print(f"Error generating Logo plot: {e}")


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
                        fontsize=10, fontweight='bold', color='black')
                
                # Bottom Label (X Axis)
                ax.text(center, len(df_cov) + 0.5, group_name, ha='center', va='top', 
                        fontsize=10, fontweight='bold', color='black', rotation=45)

            # --- Calculate Stats for this block (Diagonal) on TM Plot ---
            if df_tm is not None:
                block_val = df_tm.iloc[start:end, start:end].values.mean()
                if not pd.isna(block_val):
                    ax2.text(center, center, f"{block_val:.2f}", 
                             ha='center', va='center', fontsize=11, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_experimental_grouped_tm(df_tm, group_a_pdbs, group_b_pdbs, output_path):
    """
    Plots the TM Score matrix for Experimental structures.
    Crucially, it REORDERS the dataframe to ensure Group A is top-left 
    and Group B is bottom-right, drawing a separator line between them.
    """
    if df_tm is None or df_tm.empty:
        print("Skipping Experimental Plot: Matrix is empty.")
        return

    # 1. Clean and Reorder Indices
    df_keys = set(df_tm.index)
    
    valid_a = [x for x in group_a_pdbs if x in df_keys]
    valid_b = [x for x in group_b_pdbs if x in df_keys and x not in set(valid_a)]

    ordered_ids = valid_a + valid_b
    
    if not ordered_ids:
        print("Error: No matching PDB IDs found in the TM Matrix.")
        return

    # Reorder the Matrix: Group A first, then Group B
    df_sorted = df_tm.loc[ordered_ids, ordered_ids]

    # 2. Setup Plot
    plt.figure(figsize=(11, 10))
    sns.heatmap(df_sorted, cmap='RdYlBu_r', vmin=0, vmax=1.0, cbar=True,
                xticklabels=False, yticklabels=False, square=True)
    
    plt.title("Experimental Structural Similarity (TM)\n(Ordered by Split Groups)", fontsize=14, pad=10)

    # 3. Draw Dynamic Separator Lines
    boundary_idx = len(valid_a)
    
    if 0 < boundary_idx < len(ordered_ids):
        # Vertical and Horizontal Separators
        plt.axvline(x=boundary_idx, color='black', linestyle='--', linewidth=2, alpha=0.8)
        plt.axhline(y=boundary_idx, color='black', linestyle='--', linewidth=2, alpha=0.8)

    # 4. Add Group Labels
    if boundary_idx > 0:
        center_a = boundary_idx / 2
        plt.text(-0.5, center_a, "Group A\n(Exp)", ha='right', va='center', fontweight='bold', fontsize=11)
        plt.text(center_a, len(ordered_ids) + 0.5, "Group A", ha='center', va='top', fontweight='bold', fontsize=11, rotation=45)

    if len(valid_b) > 0:
        center_b = boundary_idx + (len(valid_b) / 2)
        plt.text(-0.5, center_b, "Group B\n(Exp)", ha='right', va='center', fontweight='bold', fontsize=11)
        plt.text(center_b, len(ordered_ids) + 0.5, "Group B", ha='center', va='top', fontweight='bold', fontsize=11, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Experimental Group Plot: {output_path}")
    

def get_actual_structure_path(directory, pdb_id):
    """
    Helper to check if file exists as .pdb or .cif
    """
    path_pdb = os.path.join(directory, f"{pdb_id}.pdb")
    if os.path.exists(path_pdb): return path_pdb
    
    path_cif = os.path.join(directory, f"{pdb_id}.cif")
    if os.path.exists(path_cif): return path_cif
    
    return None

def visualize_structures_pipeline(fasta_path, split_data, sig_split_folder, ordered_cov_path):
    """
    Main Pipeline.
    """
    base_output = os.path.join(os.path.dirname(fasta_path), 'structures')
    dir_predicted = os.path.join(base_output, 'predicted_esm')
    dir_experimental = os.path.join(base_output, 'experimental_pdb')

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

    # --- PART 0: LOGO PLOTS ---
    logo_path = os.path.join(sig_split_folder, "comparative_sequence_logos.png")
    generate_comparative_logos(records, valid_a, valid_b, logo_path, highlight_threshold=0.8)
    
    # --- PART 1: PREDICTED ---
    sample_a = random.sample(valid_a, min(len(valid_a), 50))
    sample_b = random.sample(valid_b, min(len(valid_b), 50))
    processing_list = sample_a + sample_b
    
    run_prediction_batch(records, dir_predicted, allow_list=processing_list)
    
    print("Calculating TM Matrix (Predicted)...")
    df_pred, stats_pred, split_pred = calculate_tm_matrix(sample_a, sample_b, dir_predicted)
    
    if df_pred is not None and not df_pred.empty:
        # 1a. Representative Alignment
        print("\n=== Generating Representative Alignment (Predicted) ===")
        rep_a_id = get_group_representative(df_pred, sample_a)
        rep_b_id = get_group_representative(df_pred, sample_b)
        
        if rep_a_id and rep_b_id:
            pdb_a = os.path.join(dir_predicted, f"{normalize_id(rep_a_id)}.pdb")
            pdb_b = os.path.join(dir_predicted, f"{normalize_id(rep_b_id)}.pdb")
            align_output = os.path.join(sig_split_folder, "representative_structural_alignment_predicted")
            
            if os.path.exists(pdb_a) and os.path.exists(pdb_b):
                align_and_visualize_pair(pdb_a, pdb_b, align_output,
                    label_a=f"Group A (Rep: {rep_a_id})", label_b=f"Group B (Rep: {rep_b_id})")

        # 1b. Plotting
        print("Generating Predicted TM Matrix Plot:")
        cov_ord, tm_ord = get_aligned_matrices(ordered_cov_path, df_pred, processing_list, sort_by="covariance")
        if cov_ord is not None:
            plot_side_by_side_dynamic(cov_ord, tm_ord, sample_a, sample_b, 
                os.path.join(sig_split_folder, "combined_ordered_by_covariance.png"), "(Ordered by Covariance Index)")

    # --- PART 2: EXPERIMENTAL ---
    print("\n=== Experimental Structures Check ===")
    
    pfam_id = os.path.basename(fasta_path).split('.')[0]
    global_map = prepare_global_structure_map(pfam_id, fasta_path)
    
    if global_map:
        success, exp_a_ids, exp_b_ids = check_and_download_structures(
            global_map, split_data['group_a'], split_data['group_b'], dir_experimental
        )
        
        if success:
            # 1. Convert to sets to ensure internal uniqueness
            set_a = set(exp_a_ids)
            set_b = set(exp_b_ids)
            
            # 2. Identify and remove overlapping IDs 
            # (PDBs that appear in both groups are ambiguous for comparison)
            overlap = set_a.intersection(set_b)
            if overlap:
                print(f"    [Cleanup] Removing {len(overlap)} ambiguous PDBs found in both groups: {overlap}")
                set_a = set_a - overlap
                set_b = set_b - overlap

            # 3. Convert back to sorted lists
            clean_a = sorted(list(set_a))
            clean_b = sorted(list(set_b))

            if len(clean_a) < 1 or len(clean_b) < 1:
                print("    [Structure Skip] Not enough unique, non-overlapping experimental structures.")
            else:
                print(f"Calculating TM Matrix (Experimental) for {len(clean_a) + len(clean_b)} structures...")
                
                # Pass the cleaned, disjoint lists
                df_exp, stats_exp, split_exp = calculate_tm_matrix(clean_a, clean_b, dir_experimental)
                
                if df_exp is not None and not df_exp.empty:
                    # A. HEATMAP
                    plot_experimental_grouped_tm(
                        df_exp, 
                        clean_a, 
                        clean_b, 
                        os.path.join(sig_split_folder, "experimental_tm_ordered_by_groups.png")
                    )

                    # B. REPRESENTATIVE ALIGNMENT (NEW)
                    print("\n=== Generating Representative Alignment (Experimental) ===")
                    # Select best representative for the specific group based on the matrix
                    rep_a_exp = get_group_representative(df_exp, clean_a)
                    rep_b_exp = get_group_representative(df_exp, clean_b)

                    if rep_a_exp and rep_b_exp:
                        # Find actual file paths (check .pdb and .cif)
                        pdb_a_path = get_actual_structure_path(dir_experimental, rep_a_exp)
                        pdb_b_path = get_actual_structure_path(dir_experimental, rep_b_exp)

                        if pdb_a_path and pdb_b_path:
                            align_output_exp = os.path.join(sig_split_folder, "representative_structural_alignment_experimental")
                            
                            align_and_visualize_pair(
                                pdb_a_path, 
                                pdb_b_path, 
                                align_output_exp,
                                label_a=f"Group A (Exp: {rep_a_exp})", 
                                label_b=f"Group B (Exp: {rep_b_exp})"
                            )
                        else:
                            print("Error: Could not locate Experimental PDB files on disk.")
