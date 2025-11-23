import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one


def get_pdb_data(pdb_path):
    """
    Parses a PDB file and returns coordinates and sequence (1-letter).
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
                    res_letter = three_to_one(residue.resname, custom_map={'MSE': 'M'})
                    if len(res_letter) == 1:
                        coords.append(residue['CA'].get_coord())
                        sequence.append(res_letter)
            return np.array(coords), "".join(sequence) # Return first chain
    return None, None

def map_leaves_to_pdbs(leaf_names, pdb_folder):
    """
    Maps tree leaf names (e.g., 'Seq/1-100') to PDB file paths (e.g., 'Seq_1-100.pdb').
    """
    valid_pdbs = {}
    missing = []

    for leaf in leaf_names:
        # Sanitize name: Replace / with _ to match filesystem
        safe_name = leaf.replace("/", "_")
        
        # Construct expected path
        expected_path = os.path.join(pdb_folder, f"{safe_name}.pdb")
        
        if os.path.exists(expected_path):
            valid_pdbs[leaf] = expected_path
        else:
            missing.append(leaf)
            
    return valid_pdbs, missing

# ==========================================
# 3. COMPARISON & VISUALIZATION
# ==========================================

def compute_and_plot_matrix(cache_row, cache_col, title, filename_suffix, output_folder):
    """
    Generic function to compute TM-score matrix between two sets of structures,
    print statistics, and save a heatmap.
    """
    results = []
    
    # Check if this is an intra-group comparison (same object passed)
    # We use this to exclude the diagonal (self-comparison) from statistics
    is_intra = (cache_row is cache_col)

    print(f"Calculating Matrix: {title} ({len(cache_row)} x {len(cache_col)})...")

    for name_r, (coords_r, seq_r) in cache_row.items():
        for name_c, (coords_c, seq_c) in cache_col.items():
            
            try:
                res = tm_align(coords_r, coords_c, seq_r, seq_c)
                # Average the normalized scores for symmetry
                tm_score = (res.tm_norm_chain1 + res.tm_norm_chain2) / 2.0
                
                results.append({
                    "Row_ID": name_r,
                    "Col_ID": name_c,
                    "TM_Score": tm_score
                })
            except Exception as e:
                print(f"Error aligning {name_r} vs {name_c}: {e}")

    if not results:
        print("No successful alignments.")
        return

    df = pd.DataFrame(results)
    
    # --- STATISTICS ---
    # If intra-group, exclude self-comparisons (diagonal) from the average
    # because comparing a protein to itself always gives 1.0, which inflates the mean.
    if is_intra:
        valid_scores = df[df["Row_ID"] != df["Col_ID"]]["TM_Score"]
        if valid_scores.empty:
            avg_tm = 1.0 # Only one item in group
        else:
            avg_tm = valid_scores.mean()
        print(f"Average TM-Score (excluding diagonal): {avg_tm:.4f}")
    else:
        avg_tm = df["TM_Score"].mean()
        print(f"Average TM-Score: {avg_tm:.4f}")

    # --- VISUALIZATION ---
    try:
        # Pivot for heatmap
        heatmap_data = df.pivot(index="Row_ID", columns="Col_ID", values="TM_Score")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap="viridis", vmin=0, vmax=1, 
                    cbar_kws={'label': 'TM-Score'})
        plt.title(f"{title}\nAvg TM-Score: {avg_tm:.3f}")
        plt.xlabel(None) # Clean up labels if names are long
        plt.ylabel(None)
        plt.tight_layout()
        
        save_path = os.path.join(output_folder, f"heatmap_{filename_suffix}.png")
        plt.savefig(save_path)
        print(f"Saved heatmap to: {save_path}\n")
        plt.close() # Close to free memory
        
    except Exception as e:
        print(f"Could not generate heatmap: {e}")


def compare_groups_structural(split_data, pdb_folder, sample_size=None):
    """
    Orchestrates 3 comparisons: Intra-A, Intra-B, and Inter-A-vs-B.
    """
    group_a_leaves = split_data['group_a']
    group_b_leaves = split_data['group_b']

    # 1. Map files
    pdbs_a, _ = map_leaves_to_pdbs(group_a_leaves, pdb_folder)
    pdbs_b, _ = map_leaves_to_pdbs(group_b_leaves, pdb_folder)

    print(f"\n--- Split Analysis (Node Support: {split_data['support']}) ---")
    print(f"Group A Total: {len(pdbs_a)}")
    print(f"Group B Total: {len(pdbs_b)}")

    # 2. Subsampling (Consistent for all comparisons)
    # We sample ONCE so that the A-vs-A comparison uses the exact same
    # proteins as the A-side of the A-vs-B comparison.
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
        print("Error: Missing PDBs.")
        return

    # 3. Preload Data
    print("Pre-loading PDB data...")
    cache_a = {name: get_pdb_data(path) for name, path in pdbs_a.items()}
    cache_b = {name: get_pdb_data(path) for name, path in pdbs_b.items()}
    
    # Remove failed loads
    cache_a = {k: v for k, v in cache_a.items() if v[0] is not None}
    cache_b = {k: v for k, v in cache_b.items() if v[0] is not None}

    # 4. Run Comparisons
    
    # --- Comparison 1: A vs A (Baseline for Group A) ---
    compute_and_plot_matrix(
        cache_a, cache_a, 
        title="Intra-Group A Similarity", 
        filename_suffix="intra_A", 
        output_folder=pdb_folder
    )

    # --- Comparison 2: B vs B (Baseline for Group B) ---
    compute_and_plot_matrix(
        cache_b, cache_b, 
        title="Intra-Group B Similarity", 
        filename_suffix="intra_B", 
        output_folder=pdb_folder
    )

    # --- Comparison 3: A vs B (The Hypothesis Test) ---
    compute_and_plot_matrix(
        cache_a, cache_b, 
        title="Inter-Group (A vs B) Similarity", 
        filename_suffix="inter_AvB", 
        output_folder=pdb_folder
    )
