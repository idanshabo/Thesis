import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tmtools import tm_align
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering

def get_pdb_data(pdb_path):
    """Parses PDB to return coords and sequence."""
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
                    res_letter = seq1(residue.resname, custom_map={'MSE': 'M'})
                    if len(res_letter) == 1 and res_letter != 'X':
                        coords.append(residue['CA'].get_coord())
                        sequence.append(res_letter)
            return np.array(coords), "".join(sequence)
    return None, None

def _calculate_average_excluding_diagonal(matrix):
    m = matrix.copy()
    np.fill_diagonal(m, np.nan)
    return np.nanmean(m)

def _get_sort_order(sub_matrix, labels):
    """Helper to perform hierarchical clustering ordering."""
    if len(labels) < 3: return labels
    # TM to Distance
    dist_mat = np.clip(1.0 - sub_matrix, 0, 1)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method='average')
    Z_ordered = optimal_leaf_ordering(Z, condensed)
    return [labels[i] for i in leaves_list(Z_ordered)]

def calculate_tm_matrix(group_a_ids, group_b_ids, pdb_folder):
    """
    Orchestrates the data loading, matrix calculation, and sorting.
    Returns: (Sorted DataFrame, Stats Dictionary, Split Position)
    """
    
    # 1. Map and Load Data
    all_ids = group_a_ids + group_b_ids
    cache = {}
    
    print("Loading PDB files for analysis...")
    for pid in all_ids:
        safe_id = pid.replace("/", "_")
        path = os.path.join(pdb_folder, f"{safe_id}.pdb")
        if os.path.exists(path):
            data = get_pdb_data(path)
            if data[0] is not None:
                cache[pid] = data
    
    # Update lists to only include successfully loaded IDs
    valid_a = [x for x in group_a_ids if x in cache]
    valid_b = [x for x in group_b_ids if x in cache]
    
    if len(valid_a) < 2 or len(valid_b) < 2:
        print("Not enough valid structures to calculate matrix.")
        return None, None, None

    # 2. Build Matrix
    combined_order = valid_a + valid_b
    n = len(combined_order)
    raw_matrix = np.zeros((n, n))
    
    print(f"Computing TM-Alignments ({n}x{n})...")
    for i, r_id in enumerate(combined_order):
        coords_r, seq_r = cache[r_id]
        for j, c_id in enumerate(combined_order):
            if i == j:
                raw_matrix[i, j] = 1.0
                continue
            if j < i:
                raw_matrix[i, j] = raw_matrix[j, i]
                continue
            
            coords_c, seq_c = cache[c_id]
            try:
                res = tm_align(coords_r, coords_c, seq_r, seq_c)
                score = (res.tm_norm_chain1 + res.tm_norm_chain2) / 2.0
                raw_matrix[i, j] = score
            except:
                raw_matrix[i, j] = 0.0

    # 3. Calculate Stats
    len_a = len(valid_a)
    sub_a = raw_matrix[0:len_a, 0:len_a]
    sub_b = raw_matrix[len_a:, len_a:]
    sub_inter = raw_matrix[0:len_a, len_a:]
    
    stats = {
        'avg_a': _calculate_average_excluding_diagonal(sub_a),
        'avg_b': _calculate_average_excluding_diagonal(sub_b),
        'avg_inter': np.mean(sub_inter)
    }

    # 4. Sort Groups
    sorted_a = _get_sort_order(sub_a, valid_a)
    sorted_b = _get_sort_order(sub_b, valid_b)
    final_order = sorted_a + sorted_b
    
    # 5. Create DataFrame
    df_raw = pd.DataFrame(raw_matrix, index=combined_order, columns=combined_order)
    df_sorted = df_raw.reindex(index=final_order, columns=final_order)
    
    return df_sorted, stats, len(sorted_a)
