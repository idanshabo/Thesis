import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tmtools import tm_align

# SciPy for clustering/ordering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering

# Biopython
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def get_pdb_data(pdb_path):
    """
    Parses a PDB file and returns coordinates and sequence (1-letter).
    Uses Bio.SeqUtils.seq1 for modern Biopython compatibility.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("id", pdb_path)
    except Exception:
        return None, None
    
    coords = []
    sequence = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    # 'custom_map' handles non-standard residues like MSE (Selenomethionine)
                    # seq1 handles standard 3-letter codes automatically
                    res_letter = seq1(residue.resname, custom_map={'MSE': 'M'})
                    
                    # Ensure it's a valid 1-letter code
                    if len(res_letter) == 1 and res_letter != 'X':
                        coords.append(residue['CA'].get_coord())
                        sequence.append(res_letter)
            return np.array(coords), "".join(sequence) # Return first chain
    return None, None


def map_leaves_to_pdbs(leaf_names, pdb_folder):
    """
    Maps tree leaf names (e.g., 'Seq/1-100') to PDB file paths.
    """
    valid_pdbs = {}
    missing = []

    for leaf in leaf_names:
        safe_name = leaf.replace("/", "_")
        expected_path = os.path.join(pdb_folder, f"{safe_name}.pdb")
        
        if os.path.exists(expected_path):
            valid_pdbs[leaf] = expected_path
        else:
            missing.append(leaf)
            
    return valid_pdbs, missing


# ==========================================
# 3. CLUSTERING & VISUALIZATION LOGIC
# ==========================================

def get_optimized_order(tm_submatrix, labels):
    """
    Takes a square subset of the TM matrix (e.g., just Group A),
    converts it to distance (1 - TM), and calculates the optimal 
    leaf ordering for clustering.
    """
    n = len(labels)
    if n < 3: 
        return labels # Too small to cluster

    # 1. Convert Similarity (TM) to Distance
    # TM-score is 0..1, so Distance = 1 - TM. Clip to avoid negative float errors.
    dist_matrix = np.clip(1.0 - tm_submatrix, 0, 1)
    
    # 2. Convert to condensed distance matrix (required for linkage)
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # 3. Hierarchical Clustering (Average Linkage)
    Z = linkage(condensed_dist, method='average')
    
    # 4. Optimal Leaf Ordering
    # Rotates branches to maximize similarity between adjacent leaves
    Z_ordered = optimal_leaf_ordering(Z, condensed_dist)
    
    # 5. Get sorted indices
    sorted_indices = leaves_list(Z_ordered)
    
    # Return labels in the new order
    return [labels[i] for i in sorted_indices]


def calculate_average_excluding_diagonal(matrix):
    """
    Calculates the mean of a square matrix excluding the diagonal.
    Useful for Intra-group comparisons where diagonal is always 1.0.
    """
    # Create a copy to avoid modifying original
    m = matrix.copy()
    
    # Fill diagonal with NaN so it's ignored by nanmean
    np.fill_diagonal(m, np.nan)
    
    return np.nanmean(m)


def compute_and_plot_sorted_matrix(cache_a, cache_b, title, output_folder):
    keys_a = list(cache_a.keys())
    keys_b = list(cache_b.keys())
    
    initial_order = keys_a + keys_b
    full_cache = {**cache_a, **cache_b}
    n = len(initial_order)
    
    print(f"Calculating Matrix ({n}x{n})...")
    
    # 1. Compute Full Matrix
    raw_matrix = np.zeros((n, n))
    
    for i, name_r in enumerate(initial_order):
        coords_r, seq_r = full_cache[name_r]
        for j, name_c in enumerate(initial_order):
            if i == j:
                raw_matrix[i, j] = 1.0
                continue
            if j < i: 
                raw_matrix[i, j] = raw_matrix[j, i]
                continue
            
            coords_c, seq_c = full_cache[name_c]
            try:
                # CALL YOUR TM_ALIGN WRAPPER HERE
                res = tm_align(coords_r, coords_c, seq_r, seq_c)
                score = (res.tm_norm_chain1 + res.tm_norm_chain2) / 2.0
                raw_matrix[i, j] = score
            except:
                raw_matrix[i, j] = 0.0

    # 2. Extract Sub-matrices
    len_a = len(keys_a)
    matrix_a = raw_matrix[0:len_a, 0:len_a]
    matrix_b = raw_matrix[len_a:, len_a:]
    matrix_inter = raw_matrix[0:len_a, len_a:] # A rows, B columns

    # 3. Calculate Statistics
    avg_a = calculate_average_excluding_diagonal(matrix_a)
    avg_b = calculate_average_excluding_diagonal(matrix_b)
    avg_inter = np.mean(matrix_inter) # No diagonal to exclude here

    print(f"Stats - Intra A: {avg_a:.3f} | Intra B: {avg_b:.3f} | Inter: {avg_inter:.3f}")

    # 4. Sorting Logic
    def get_sort_order(sub_matrix, labels):
        if len(labels) < 3: return labels
        dist_mat = np.clip(1.0 - sub_matrix, 0, 1)
        condensed = squareform(dist_mat, checks=False)
        Z = linkage(condensed, method='average')
        Z_ordered = optimal_leaf_ordering(Z, condensed)
        return [labels[i] for i in leaves_list(Z_ordered)]

    sorted_keys_a = get_sort_order(matrix_a, keys_a)
    sorted_keys_b = get_sort_order(matrix_b, keys_b)
    final_order = sorted_keys_a + sorted_keys_b
    
    # 5. Re-Index Data
    df_raw = pd.DataFrame(raw_matrix, index=initial_order, columns=initial_order)
    df_sorted = df_raw.reindex(index=final_order, columns=final_order)
    
    # 6. High-Res Visualization
    # Increase figsize for better resolution
    plt.figure(figsize=(14, 12)) 
    
    use_labels = True if n < 50 else False
    
    # Plot heatmap
    ax = sns.heatmap(df_sorted, cmap="viridis", vmin=0, vmax=1, 
                xticklabels=use_labels, yticklabels=use_labels,
                cbar_kws={'label': 'TM-Score'})

    # Visual Separators
    split_pos = len(sorted_keys_a)
    plt.axhline(split_pos, color='white', linewidth=2, linestyle='--')
    plt.axvline(split_pos, color='white', linewidth=2, linestyle='--')
    
    # 7. Add Statistical Text Annotations
    # Font settings
    font_args = {'color': 'white', 'ha': 'center', 'va': 'center', 'fontweight': 'bold', 'fontsize': 14}
    
    # Intra-A (Top Left)
    plt.text(split_pos/2, split_pos/2, 
             f"Group A\nAvg: {avg_a:.2f}", **font_args)
    
    # Intra-B (Bottom Right)
    plt.text(split_pos + (n-split_pos)/2, split_pos + (n-split_pos)/2, 
             f"Group B\nAvg: {avg_b:.2f}", **font_args)

    # Inter-Group (Bottom Left - represents the A vs B comparison)
    # Note: The matrix is symmetric, so Top-Right and Bottom-Left are the same.
    plt.text(split_pos/2, split_pos + (n-split_pos)/2, 
             f"Inter-Group\nAvg: {avg_inter:.2f}", **font_args)

    plt.title(f"{title}\n(High Resolution | Ordered)", fontsize=16)
    plt.tight_layout()
    
    # Save with high DPI
    save_path = os.path.join(output_folder, "tm_score_matrix_high_res.png")
    plt.savefig(save_path, dpi=300) # Standard print quality
    print(f"Saved high-res heatmap to: {save_path}\n")
    plt.close()


def compare_groups_structural(split_data, pdb_folder, sample_size=None):
    """
    Main orchestration function.
    """
    group_a_leaves = split_data['group_a']
    group_b_leaves = split_data['group_b']

    # 1. Map files
    pdbs_a, _ = map_leaves_to_pdbs(group_a_leaves, pdb_folder)
    pdbs_b, _ = map_leaves_to_pdbs(group_b_leaves, pdb_folder)

    print(f"\n--- Split Analysis (Node Support: {split_data.get('support', 'N/A')}) ---")
    print(f"Group A Found: {len(pdbs_a)}")
    print(f"Group B Found: {len(pdbs_b)}")

    # 2. Subsampling
    if sample_size:
        if len(pdbs_a) > sample_size:
            print(f"  -> Subsampling Group A to {sample_size}")
            keys_a = random.sample(list(pdbs_a.keys()), sample_size)
            pdbs_a = {k: pdbs_a[k] for k in keys_a}
        
        if len(pdbs_b) > sample_size:
            print(f"  -> Subsampling Group B to {sample_size}")
            keys_b = random.sample(list(pdbs_b.keys()), sample_size)
            pdbs_b = {k: pdbs_b[k] for k in keys_b}

    if len(pdbs_a) == 0 or len(pdbs_b) == 0:
        print("Error: Missing PDBs in one or both groups.")
        return

    # 3. Preload Data
    print("Pre-loading PDB data...")
    cache_a = {name: get_pdb_data(path) for name, path in pdbs_a.items()}
    cache_b = {name: get_pdb_data(path) for name, path in pdbs_b.items()}
    
    # Remove failed loads (None)
    cache_a = {k: v for k, v in cache_a.items() if v[0] is not None}
    cache_b = {k: v for k, v in cache_b.items() if v[0] is not None}

    # 4. Run Comparisons
    if len(cache_a) < 2 or len(cache_b) < 2:
        print("Not enough sequences to perform matrix clustering.")
        return

    compute_and_plot_sorted_matrix(
        cache_a, 
        cache_b, 
        title=f"Structural Similarity (Node Support: {split_data.get('support', 'N/A')})",
        output_folder=pdb_folder
    )
