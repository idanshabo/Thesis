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
    valid_ids = [uid for uid in group_ids if uid in df_tm.index]
    
    if not valid_ids:
        return None

    # Subset the matrix
    sub_matrix = df_tm.loc[valid_ids, valid_ids]
    
    # Calculate mean TM score for each sample
    mean_scores = sub_matrix.mean(axis=1)
    representative_id = mean_scores.idxmax()
    
    print(f"Selected Representative for Group (n={len(valid_ids)}): {representative_id} (Avg TM: {mean_scores.max():.2f})")
    return representative_id


def align_and_visualize_pair(pdb_path_a, pdb_path_b, output_base_path, 
                             label_a="Group A", label_b="Group B"):
    """
    Generates two visualizations:
    1. Superimposed (Overlapping alignment)
    2. Side-by-Side (Separated for clarity)
    """
    # 1. Initialize PyMOL
    try:
        pymol.finish_launching(['pymol', '-qc'])
    except:
        pass
    cmd.reinitialize()

    # 2. Load the PDB files
    cmd.load(pdb_path_a, 'obj_A')
    cmd.load(pdb_path_b, 'obj_B')

    # 3. Align B onto A
    align_results = cmd.align('obj_B', 'obj_A')
    print(f"Alignment RMSD: {align_results[0]:.3f} Å")

    # 4. Basic Styling
    cmd.hide('all')
    cmd.show('cartoon')
    
    color_a = 'cyan'
    color_b = 'magenta'
    cmd.color(color_a, 'obj_A')
    cmd.color(color_b, 'obj_B')
    
    # Global Label Settings
    cmd.set("label_font_id", 13)
    cmd.set("label_size", 24) # Slightly larger for readability
    cmd.set("label_color", "black")
    cmd.bg_color('white')
    cmd.set('ray_opaque_background', 1)

    # =================================================================
    # PLOT 1: SUPERIMPOSED
    # =================================================================
    ([min_x, min_y, min_z], [max_x, max_y, max_z]) = cmd.get_extent('all')
    center_x = (min_x + max_x) / 2
    height = max_y - min_y

    # Add Labels (Stacked vertically to prevent overlap)
    # Title at top
    cmd.pseudoatom("title_1", pos=[center_x, max_y + (height * 0.2), min_z], 
                   label="Representative Alignment (Superimposed)")
    
    # Legend at bottom (Stacked with large gap)
    # Lower Group A first, then Group B below it
    cmd.pseudoatom("leg_a_1", pos=[center_x, min_y - (height * 0.1), min_z], 
                   label=f"{label_a} ({color_a})")
    cmd.pseudoatom("leg_b_1", pos=[center_x, min_y - (height * 0.25), min_z], 
                   label=f"{label_b} ({color_b})")

    # Zoom and Save
    cmd.zoom('visible', buffer=5)
    cmd.png(f"{output_base_path}_superimposed.png", width=1600, height=1200, dpi=300, ray=1)
    print(f"Saved: {output_base_path}_superimposed.png")

    # =================================================================
    # PLOT 2: SIDE-BY-SIDE
    # =================================================================
    # Clear previous labels
    cmd.delete("title_1")
    cmd.delete("leg_a_1")
    cmd.delete("leg_b_1")

    # Move Group B to the right
    width = max_x - min_x
    shift = width + 20
    cmd.translate([shift, 0, 0], 'obj_B', camera=0)

    # Recalculate dimensions for the wider scene
    ([min_x, min_y, min_z], [max_x, max_y, max_z]) = cmd.get_extent('all')
    center_x = (min_x + max_x) / 2
    
    # Get centers of individual objects for specific labels
    ([ax1, ay1, az1], [ax2, ay2, az2]) = cmd.get_extent('obj_A')
    center_a_x = (ax1 + ax2) / 2
    
    ([bx1, by1, bz1], [bx2, by2, bz2]) = cmd.get_extent('obj_B')
    center_b_x = (bx1 + bx2) / 2
    
    height = max_y - min_y

    # Add New Labels
    cmd.pseudoatom("title_2", pos=[center_x, max_y + (height * 0.2), min_z], 
                   label="Representative Alignment (Side-by-Side)")

    # Label A under object A
    cmd.pseudoatom("leg_a_2", pos=[center_a_x, min_y - (height * 0.1), min_z], 
                   label=f"{label_a} ({color_a})")
    
    # Label B under object B (Same Y height as A to keep them aligned)
    cmd.pseudoatom("leg_b_2", pos=[center_b_x, min_y - (height * 0.1), min_z], 
                   label=f"{label_b} ({color_b})")

    # Zoom and Save
    cmd.zoom('visible', buffer=5)
    cmd.png(f"{output_base_path}_side_by_side.png", width=2400, height=1200, dpi=300, ray=1)
    
    cmd.save(f"{output_base_path}_combined.pse")
    print(f"Saved: {output_base_path}_side_by_side.png")
