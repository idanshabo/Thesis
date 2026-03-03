import os
import shutil
import uuid
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import logomaker
from collections import Counter
from Bio.PDB import PDBParser
import warnings
from Bio import BiopythonWarning

# Standard colors for Q8 secondary structure
Q8_COLORS = {
    'H': '#FF00FF', # Alpha helix
    'G': '#FFC0CB', # 3-10 helix
    'I': '#8B008B', # Pi helix
    'E': '#FFFF00', # Extended strand
    'B': '#BDB76B', # Beta bridge
    'T': '#00FFFF', # Turn
    'S': '#00FF00', # Bend
    'C': '#808080', # Coil/Loop
    '-': '#FFFFFF'  # Gap
}

def normalize_id(identifier):
    return identifier.replace("/", "_")

def get_dssp_q8_from_pdb(pdb_path):
    """Calculates 1D secondary structure string from a PDB using MDTraj."""
    try:
        traj = md.load(pdb_path)
        ss_array = md.compute_dssp(traj, simplified=False)[0]
        ss_list = []
        for char in ss_array:
            if char in [' ', '-', 'NA']: ss_list.append('C')
            else: ss_list.append(char)
        return "".join(ss_list)
    except Exception as e:
        print(f"    [MDTraj Warning] Failed for {pdb_path}: {e}")
        return None

def get_representative_structure(aligned_sequences, seq_ids, rep_id, dir_predicted):
    """
    Fetches the actual Q8 structure for the specific representative protein
    and maps it to the MSA alignment length by re-inserting gaps.
    """
    norm_rep = normalize_id(rep_id)
    
    # 1. Find the representative sequence in our lists
    rep_idx = -1
    for i, sid in enumerate(seq_ids):
        if normalize_id(sid) == norm_rep:
            rep_idx = i
            break
            
    if rep_idx == -1:
        print(f"    [Warning] Representative ID {rep_id} not found in alignment.")
        return "-" * len(aligned_sequences[0])
        
    seq = aligned_sequences[rep_idx]
    ungapped = seq.replace("-", "")
    
    # 2. Get its actual predicted structure
    pdb_path = os.path.join(dir_predicted, f"{norm_rep}.pdb")
    ss_ungapped = None
    if os.path.exists(pdb_path):
        ss_ungapped = get_dssp_q8_from_pdb(pdb_path)
        
    if not ss_ungapped or len(ss_ungapped) != len(ungapped):
        print(f"    [Warning] Could not get valid structure for representative {rep_id}.")
        return "-" * len(seq)
        
    # 3. Re-insert gaps based exactly on the representative's MSA sequence
    ss_gapped = []
    idx = 0
    for char in seq:
        if char == "-":
            ss_gapped.append("-")
        else:
            ss_gapped.append(ss_ungapped[idx])
            idx += 1
            
    return "".join(ss_gapped)

def generate_comparative_logos(records, group_a_ids, group_b_ids, rep_a, rep_b, dir_predicted, output_path, highlight_threshold=0.6):
    print("\n--- Generating Comparative Sequence Logos with Representative SS ---")
    seq_map = {}
    for r in records:
        seq_map[r.id] = str(r.seq).upper()
        seq_map[normalize_id(r.id)] = str(r.seq).upper()

    seqs_a, seqs_b = [], []
    valid_ids_a, valid_ids_b = [], []
    
    for uid in group_a_ids:
        key = uid if uid in seq_map else normalize_id(uid)
        if key in seq_map: 
            seqs_a.append(seq_map[key])
            valid_ids_a.append(key)
            
    for uid in group_b_ids:
        key = uid if uid in seq_map else normalize_id(uid)
        if key in seq_map: 
            seqs_b.append(seq_map[key])
            valid_ids_b.append(key)

    if not seqs_a or not seqs_b:
        print("Error: One or both groups are empty. Skipping Logo plot.")
        return

    try:
        counts_a = logomaker.alignment_to_matrix(seqs_a)
        counts_b = logomaker.alignment_to_matrix(seqs_b)
        counts_a, counts_b = align_matrices(counts_a, counts_b)

        info_a = logomaker.transform_matrix(counts_a, from_type='counts', to_type='information')
        info_b = logomaker.transform_matrix(counts_b, from_type='counts', to_type='information')
        global_max = max(info_a.sum(axis=1).max(), info_b.sum(axis=1).max()) * 1.1

        prob_a = logomaker.transform_matrix(counts_a, from_type='counts', to_type='probability')
        prob_b = logomaker.transform_matrix(counts_b, from_type='counts', to_type='probability')
        diff_score = np.sum(np.abs(prob_a - prob_b), axis=1)
        divergent_positions = diff_score[diff_score > highlight_threshold].index.tolist()

        # Fetch Representative Structures instead of Consensus
        ss_rep_a = get_representative_structure(seqs_a, valid_ids_a, rep_a, dir_predicted)
        ss_rep_b = get_representative_structure(seqs_b, valid_ids_b, rep_b, dir_predicted)

        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(4, 1, height_ratios=[4, 0.5, 4, 0.5], hspace=0.3)

        ax_logo_a = fig.add_subplot(gs[0])
        ax_ss_a = fig.add_subplot(gs[1], sharex=ax_logo_a)
        ax_logo_b = fig.add_subplot(gs[2], sharex=ax_logo_a)
        ax_ss_b = fig.add_subplot(gs[3], sharex=ax_logo_a)

        logomaker.Logo(info_a, ax=ax_logo_a, color_scheme='chemistry')
        logomaker.Logo(info_b, ax=ax_logo_b, color_scheme='chemistry')
        ax_logo_a.set_ylim(0, global_max)
        ax_logo_b.set_ylim(0, global_max)
        
        # Display the representative used in the title
        ax_logo_a.set_title(f"Group A ({len(seqs_a)} sequences) | Structure: {rep_a}", fontsize=14, fontweight='bold')
        ax_logo_b.set_title(f"Group B ({len(seqs_b)} sequences) | Structure: {rep_b}", fontsize=14, fontweight='bold')
        ax_logo_a.set_ylabel("Bits")
        ax_logo_b.set_ylabel("Bits")

        draw_biological_ss_track(ax_ss_a, ss_rep_a)
        draw_biological_ss_track(ax_ss_b, ss_rep_b)

        for pos in divergent_positions:
            for ax in [ax_logo_a, ax_ss_a, ax_logo_b, ax_ss_b]:
                ax.axvspan(pos - 0.5, pos + 0.5, color='red', alpha=0.2, zorder=0)

        plt.setp(ax_logo_a.get_xticklabels(), visible=False)
        plt.setp(ax_logo_b.get_xticklabels(), visible=False)
        plt.setp(ax_ss_a.get_xticklabels(), visible=False)
        
        handles = [patches.Patch(color=color, label=state) for state, color in Q8_COLORS.items() if state != '-']
        fig.legend(handles=handles, loc='lower center', ncol=8, bbox_to_anchor=(0.5, -0.05), frameon=False)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Comparative Sequence Logos (with Representative SS): {output_path}")

    except Exception as e:
        print(f"Error generating Logo plot: {e}")
