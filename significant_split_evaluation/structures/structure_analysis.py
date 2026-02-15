import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
from Bio.SeqUtils import seq1
from tmtools import tm_align
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering

def get_structure_data(file_path):
    """
    Parses PDB or MMCIF to return coords and sequence.
    Detects format based on file extension.
    """
    # 1. Select Parser based on extension
    if file_path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
        # MMCIF parser requires an ID, we can use the filename
        struct_id = os.path.basename(file_path)
    else:
        parser = PDBParser(QUIET=True)
        struct_id = "id"

    try:
        structure = parser.get_structure(struct_id, file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None
    
    coords = []
    sequence = []
    
    # 2. Extract Coordinates (CA atoms only)
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check for standard amino acids using CA atom availability
                if 'CA' in residue:
                    try:
                        # Convert 3-letter code to 1-letter
                        res_letter = seq1(residue.resname, custom_map={'MSE': 'M'})
                        
                        # Filter out unknowns or non-standard that map to 'X' (optional, but safer for alignment)
                        if len(res_letter) == 1 and res_letter != 'X':
                            coords.append(residue['CA'].get_coord())
                            sequence.append(res_letter)
                    except Exception:
                        pass
        # Usually we only take the first model if multiple exist
        break 
            
    if not coords:
        return None, None

    return np.array(coords), "".join(sequence)

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
    
    print("Loading structure files (PDB/CIF) for analysis...")
    
    for pid in all_ids:
        safe_id = pid.replace("/", "_")
        safe_id_lower = safe_id.lower()
        
        possible_paths = [
            os.path.join(pdb_folder, f"{safe_id}.pdb"),
            os.path.join(pdb_folder, f"{safe_id}.cif"),
            os.path.join(pdb_folder, f"{safe_id_lower}.pdb"),
            os.path.join(pdb_folder, f"{safe_id_lower}.cif")
        ]
        
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if found_path:
            # Assuming get_structure_data is defined or imported in your file
            data = get_structure_data(found_path) 
            if data[0] is not None:
                cache[pid] = data
        else:
            pass
    
    # Update lists to only include successfully loaded IDs
    valid_a = [x for x in group_a_ids if x in cache]
    valid_b = [x for x in group_b_ids if x in cache]
    
    if len(valid_a) < 1 or len(valid_b) < 1:
        print(f"Not enough valid structures. Found A:{len(valid_a)}, B:{len(valid_b)}")
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
                # Assuming tm_align is imported
                res = tm_align(coords_r, coords_c, seq_r, seq_c)
                score = (res.tm_norm_chain1 + res.tm_norm_chain2) / 2.0
                raw_matrix[i, j] = score
            except Exception as e:
                print(f"Alignment error {r_id} vs {c_id}: {e}")
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

    # 4. Sort and Dataframe
    sorted_a = _get_sort_order(sub_a, valid_a)
    sorted_b = _get_sort_order(sub_b, valid_b)
    final_order = sorted_a + sorted_b
    
    df_raw = pd.DataFrame(raw_matrix, index=combined_order, columns=combined_order)
    df_sorted = df_raw.reindex(index=final_order, columns=final_order)
    
    return df_sorted, stats, len(sorted_a)
