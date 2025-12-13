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

def align_and_visualize_pair(pdb_path_a, pdb_path_b, output_base_path, 
                             label_a="Group A", label_b="Group B"):
    """
    Uses PyMOL to align two structures, add dynamic labels/legend, and save.
    Corrects the 'zoom' issue to ensure labels are not cut off.
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
    rmsd_val = align_results[0]
    print(f"Alignment RMSD: {rmsd_val:.3f} Å")

    # 4. Styling (Cartoon representation)
    cmd.hide('all')
    cmd.show('cartoon')
    
    color_a = 'cyan'
    color_b = 'magenta'
    cmd.color(color_a, 'obj_A')
    cmd.color(color_b, 'obj_B')

    # =================================================================
    # DYNAMIC LABEL PLACEMENT
    # =================================================================
    # Get the bounding box of the aligned molecules
    ([min_x, min_y, min_z], [max_x, max_y, max_z]) = cmd.get_extent('all')
    
    center_x = (min_x + max_x) / 2
    height = max_y - min_y
    
    # Place Title slightly closer (15% above) to minimize empty space
    title_y = max_y + (height * 0.15)
    cmd.pseudoatom("title_pos", pos=[center_x, title_y, min_z], label="Representative Structural Alignment")
    
    # Place Legend (Line 1)
    leg1_y = min_y - (height * 0.15)
    cmd.pseudoatom("leg_a", pos=[center_x, leg1_y, min_z], label=f"{label_a} ({color_a})")
    
    # Place Legend (Line 2)
    leg2_y = min_y - (height * 0.25)
    cmd.pseudoatom("leg_b", pos=[center_x, leg2_y, min_z], label=f"{label_b} ({color_b})")

    # Label Styling
    cmd.set("label_font_id", 13)      # Arial-like font
    cmd.set("label_size", 20)         # Slightly smaller font to fit better
    cmd.set("label_color", "black")
    
    # 'visible' includes the protein AND the pseudoatom labels
    # 'buffer=5' adds a 5 Angstrom margin around the edges so text isn't flush with the border
    cmd.zoom('visible', buffer=5)

    # 5. Save Outputs
    cmd.bg_color('white') 
    cmd.set('ray_opaque_background', 1)

    # Save PNG
    png_path = f"{output_base_path}.png"
    # Increasing width slightly helps long text strings fit
    cmd.png(png_path, width=1600, height=1200, dpi=300, ray=1)
    
    # Save Session
    cmd.save(f"{output_base_path}.pse")
    
    print(f"Saved visualization to: {png_path}")
