import os
import pandas as pd
import pymol
from pymol import cmd

def get_group_representative(df_tm, group_ids):
    """
    Identifies the 'Medoid' of the group: the structure that is most 
    similar (highest average TM-score) to all other members of the group.
    """
    # Filter the TM matrix to only include the specific group members
    # We use intersection to ensure we only look at IDs that actually exist in the matrix
    valid_ids = [uid for uid in group_ids if uid in df_tm.index]
    
    if not valid_ids:
        return None

    # Subset the matrix
    sub_matrix = df_tm.loc[valid_ids, valid_ids]
    
    # Calculate mean TM score for each sample against its group members
    # ID with highest mean is the most "representative" center of the cluster
    mean_scores = sub_matrix.mean(axis=1)
    representative_id = mean_scores.idxmax()
    
    print(f"Selected Representative for Group (n={len(valid_ids)}): {representative_id} (Avg TM: {mean_scores.max():.2f})")
    return representative_id

def align_and_visualize_pair(pdb_path_a, pdb_path_b, output_base_path, label_a="Group_A", label_b="Group_B"):
    """
    Uses PyMOL to align two structures and save the visualization.
    """
    # 1. Initialize PyMOL (Idempotent - safe to call multiple times)
    try:
        pymol.finish_launching(['pymol', '-qc'])
    except:
        pass # PyMOL might already be running

    # Reset ensures we don't have leftover objects from previous runs
    cmd.reinitialize()

    # 2. Load the PDB files
    # We give them friendly object names like 'Group_A_Rep'
    cmd.load(pdb_path_a, label_a)
    cmd.load(pdb_path_b, label_b)

    # 3. Align mobile (B) onto ref (A)
    # align returns: [RMSD, atoms_aligned, n_cycles, rmsd_pre_cycle, n_atoms_pre_cycle, score, n_atoms_score]
    align_results = cmd.align(label_b, label_a)
    rmsd_val = align_results[0]
    print(f"Aligned {label_b} to {label_a} | RMSD: {rmsd_val:.3f} Å")

    # 4. Styling (Mol*-like aesthetic)
    cmd.hide('all')
    cmd.show('cartoon')
    cmd.set('ray_opaque_background', 0) # Transparent background
    
    # Coloring
    cmd.color('cyan', label_a)
    cmd.color('magenta', label_b)
    
    # Center the camera on the alignment
    cmd.zoom()

    # 5. Save Outputs
    # Save Image
    png_path = f"{output_base_path}_alignment.png"
    cmd.png(png_path, width=1200, height=1200, dpi=300, ray=1)
    
    # Save Session (allows you to open in PyMOL desktop and rotate)
    pse_path = f"{output_base_path}_session.pse"
    cmd.save(pse_path)
    
    print(f"Saved visualization to: {png_path}")
