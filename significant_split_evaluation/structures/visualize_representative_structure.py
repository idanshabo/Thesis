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
    """
    # 1. Initialize PyMOL
    try:
        pymol.finish_launching(['pymol', '-qc'])
    except:
        pass
    cmd.reinitialize()

    # 2. Load the PDB files
    # We rename them internally to 'obj_A' and 'obj_B' to keep code clean
    cmd.load(pdb_path_a, 'obj_A')
    cmd.load(pdb_path_b, 'obj_B')

    # 3. Align B onto A
    align_results = cmd.align('obj_B', 'obj_A')
    rmsd_val = align_results[0]
    print(f"Alignment RMSD: {rmsd_val:.3f} Å")

    # 4. Styling (Cartoon representation)
    cmd.hide('all')
    cmd.show('cartoon')
    
    # Define Colors
    color_a = 'cyan'
    color_b = 'magenta'
    cmd.color(color_a, 'obj_A')
    cmd.color(color_b, 'obj_B')
    
    # Orient view nicely
    cmd.zoom()

    # =================================================================
    # DYNAMIC LABEL PLACEMENT (Title & Legend)
    # =================================================================
    # Get the bounding box of the aligned molecules to know where to put text
    # extent returns [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    ([min_x, min_y, min_z], [max_x, max_y, max_z]) = cmd.get_extent('all')
    
    center_x = (min_x + max_x) / 2
    height = max_y - min_y
    
    # --- Add Title (Above the structure) ---
    title_y = max_y + (height * 0.2) # 20% above the top
    cmd.pseudoatom("title_pos", pos=[center_x, title_y, min_z], label="Representative Structural Alignment")
    
    # --- Add Legend (Below the structure) ---
    # Legend Line 1
    leg1_y = min_y - (height * 0.1)
    cmd.pseudoatom("leg_a", pos=[center_x, leg1_y, min_z], label=f"{label_a} ({color_a})")
    
    # Legend Line 2
    leg2_y = min_y - (height * 0.18)
    cmd.pseudoatom("leg_b", pos=[center_x, leg2_y, min_z], label=f"{label_b} ({color_b})")

    # --- Label Styling ---
    # Global label settings
    cmd.set("label_font_id", 13)      # Arial-like font
    cmd.set("label_size", 24)         # Font size
    cmd.set("label_color", "black")   # Text color
    
    # Specific colors for the legend text to match the structures (Optional)
    # Note: PyMOL labels usually take one color. Keeping them black is safer for readability.
    
    # 5. Save Outputs
    # White background looks best for reports
    cmd.bg_color('white') 
    cmd.set('ray_opaque_background', 1)

    # Save PNG
    png_path = f"{output_base_path}.png"
    cmd.png(png_path, width=1200, height=1200, dpi=300, ray=1)
    
    # Save Session (PSE)
    cmd.save(f"{output_base_path}.pse")
    
    print(f"Saved visualization with legend to: {png_path}")
