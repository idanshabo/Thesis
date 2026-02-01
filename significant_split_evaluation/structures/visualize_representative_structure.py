import os
import pandas as pd
import pymol
from pymol import cmd


def get_group_representative(df_tm, group_ids):
    """
    Identifies the 'Medoid' of the group: the structure that is most 
    similar (highest average TM-score) to all other members of the group.
    """
    valid_ids = [uid for uid in group_ids if uid in df_tm.index]
    if not valid_ids: return None

    sub_matrix = df_tm.loc[valid_ids, valid_ids]
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
    try:
        pymol.finish_launching(['pymol', '-qc'])
    except:
        pass
    cmd.reinitialize()

    # Load structures
    cmd.load(pdb_path_a, 'obj_A')
    cmd.load(pdb_path_b, 'obj_B')

    # Align B to A
    align_results = cmd.align('obj_B', 'obj_A')
    print(f"Alignment RMSD: {align_results[0]:.3f} Å")

    # Styling
    cmd.hide('all')
    cmd.show('cartoon')
    color_a = 'cyan'
    color_b = 'magenta'
    cmd.color(color_a, 'obj_A')
    cmd.color(color_b, 'obj_B')
    
    # Global Settings
    cmd.set("label_font_id", 13)
    cmd.set("label_size", 18)    # Readable size
    cmd.set("label_color", "black")
    cmd.bg_color('white')
    cmd.set('ray_opaque_background', 1)

    # --- PLOT 1: SUPERIMPOSED ---
    # 1. Get positions of PROTEINS only
    ([min_x, min_y, min_z], [max_x, max_y, max_z]) = cmd.get_extent('obj_A or obj_B')
    center_x = (min_x + max_x) / 2
    height = max_y - min_y

    # 2. Place Labels relative to proteins
    # Title slightly higher
    cmd.pseudoatom("title_1", pos=[center_x, max_y + (height * 0.15), min_z], 
                   label="Representative Alignment (Superimposed)")
    
    # Legends below
    cmd.pseudoatom("leg_a_1", pos=[center_x, min_y - (height * 0.15), min_z], 
                   label=f"{label_a} ({color_a})")
    cmd.pseudoatom("leg_b_1", pos=[center_x, min_y - (height * 0.30), min_z], 
                   label=f"{label_b} ({color_b})")

    # 3. ZOOM LAST (Crucial Fix)
    # We zoom on 'visible' which now includes the proteins AND the new labels
    # Buffer 3.0 gives extra padding so bottom text isn't cut
    cmd.zoom('visible', buffer=3.0)
    
    cmd.png(f"{output_base_path}_superimposed.png", width=2000, height=1600, dpi=300, ray=1)
    print(f"Saved: {output_base_path}_superimposed.png")

    # --- PLOT 2: SIDE-BY-SIDE ---
    cmd.delete("title_1")
    cmd.delete("leg_a_1")
    cmd.delete("leg_b_1")

    # Move Group B to the right
    width = max_x - min_x
    shift = width + 5
    cmd.translate([shift, 0, 0], 'obj_B', camera=0)

    # 1. Recalculate extent for the new wide scene
    ([min_x, min_y, min_z], [max_x, max_y, max_z]) = cmd.get_extent('obj_A or obj_B')
    center_x = (min_x + max_x) / 2
    height = max_y - min_y
    
    # 2. Place Labels
    cmd.pseudoatom("title_2", pos=[center_x, max_y + (height * 0.15), min_z], 
                   label="Representative Alignment (Side-by-Side)")

    cmd.pseudoatom("leg_a_2", pos=[center_x, min_y - (height * 0.15), min_z], 
                   label=f"{label_a} ({color_a})")
    
    cmd.pseudoatom("leg_b_2", pos=[center_x, min_y - (height * 0.30), min_z], 
                   label=f"{label_b} ({color_b})")

    # 3. ZOOM LAST
    cmd.zoom('visible', buffer=3.0)

    # Wide canvas to fit text
    cmd.png(f"{output_base_path}_side_by_side.png", width=3000, height=1600, dpi=300, ray=1)
    
    cmd.save(f"{output_base_path}_combined.pse")
    print(f"Saved: {output_base_path}_side_by_side.png")
