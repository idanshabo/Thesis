import os
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import logomaker
import mdtraj as md
from collections import Counter

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
    """Calculates 1D secondary structure string from an ESMFold PDB using MDTraj."""
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

def get_netsurfp3_q8_from_sequence(sequence, seq_id, cache_dir="ss_cache", timeout=120):
    """Predicts Q8 secondary structure from sequence via API and caches the result."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{seq_id}.ss")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return f.read().strip()

    submit_url = "https://services.healthtech.dtu.dk/cgi-bin/webface2.fcgi"
    payload = {
        'configfile': '/var/www/html/services/NetSurfP-3.0/webface.cf',
        'SEQPASTE': f">{seq_id}\n{sequence}"
    }
    
    try:
        submit_res = requests.post(submit_url, data=payload, timeout=30)
        submit_res.raise_for_status()
        if 'jobid=' not in submit_res.url: return None
            
        job_id = submit_res.url.split('jobid=')[1].split('&')[0]
        result_url = f"https://services.healthtech.dtu.dk/cgi-bin/webface2.fcgi?jobid={job_id}&wait=20"
        json_url = f"https://services.healthtech.dtu.dk/services/NetSurfP-3.0/tmp/{job_id}/results.json"

        start_time = time.time()
        while time.time() - start_time < timeout:
            requests.get(result_url, timeout=10)
            json_check = requests.get(json_url, timeout=10)
            
            if json_check.status_code == 200:
                data = json_check.json()
                q8_string = "".join([residue['q8'] for residue in data[seq_id]['seq']])
                with open(cache_path, 'w') as f:
                    f.write(q8_string)
                return q8_string
            time.sleep(5) 
        return None
    except Exception as e:
        print(f"    [NetSurfP Error] API call failed for {seq_id}: {e}")
        return None


def get_consensus_structure(aligned_sequences, seq_ids, dir_predicted, ss_predictor="netsurfp"):
    """
    Calculates the raw, unfiltered column-by-column majority vote 
    for the secondary structure of the group.
    """
    aligned_ss_strings = []
    
    for seq, seq_id in zip(aligned_sequences, seq_ids):
        ungapped = seq.replace("-", "")
        if not ungapped:
            continue
            
        ss_ungapped = None
        
        # --- TOGGLE LOGIC ---
        if ss_predictor == "esmfold":
            pdb_path = os.path.join(dir_predicted, f"{normalize_id(seq_id)}.pdb")
            if os.path.exists(pdb_path):
                ss_ungapped = get_dssp_q8_from_pdb(pdb_path)
        else:
            # Default to netsurfp
            ss_ungapped = get_netsurfp3_q8_from_sequence(ungapped, normalize_id(seq_id))
            
        if not ss_ungapped or len(ss_ungapped) != len(ungapped):
            continue
            
        # Re-insert gaps based on the original aligned sequence
        ss_gapped = []
        idx = 0
        for char in seq:
            if char == "-":
                ss_gapped.append("-")
            else:
                ss_gapped.append(ss_ungapped[idx])
                idx += 1
        aligned_ss_strings.append("".join(ss_gapped))
        
    if not aligned_ss_strings:
        return "-" * len(aligned_sequences[0]), 0
        
    # Calculate pure column-wise consensus (Majority Vote)
    consensus_ss = []
    seq_length = len(aligned_sequences[0])
    
    for col_idx in range(seq_length):
        col = [ss[col_idx] for ss in aligned_ss_strings if ss[col_idx] != "-"]
        if not col:
            consensus_ss.append("-")
        else:
            most_common = Counter(col).most_common(1)[0][0]
            consensus_ss.append(most_common)
            
    return "".join(consensus_ss), len(aligned_ss_strings)

def draw_biological_ss_track(ax, ss_string):
    """Draws biological shapes, bridging across MSA gaps naturally."""
    blocks = []
    if ss_string:
        current_state = None
        start_idx = None
        last_idx = None
        
        for i, state in enumerate(ss_string):
            if state == '-': 
                continue # Bridge visually over gaps
                
            if current_state is None:
                current_state = state
                start_idx = i
                last_idx = i
            elif state == current_state:
                last_idx = i
            else:
                blocks.append((current_state, start_idx, last_idx))
                current_state = state
                start_idx = i
                last_idx = i
                
        if current_state is not None:
            blocks.append((current_state, start_idx, last_idx))

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(ss_string) - 0.5)
    ax.set_yticks([])
    ax.set_ylabel("Q8", rotation=0, labelpad=15, va='center')
    ax.plot([-0.5, len(ss_string) - 0.5], [0.5, 0.5], color='#D0D0D0', linewidth=2, zorder=1)

    for state, start, end in blocks:
        if state == '-': continue
        color = Q8_COLORS.get(state, '#808080')
        x_start = start - 0.5
        x_end = end + 0.5
        length = x_end - x_start

        if state in ['H', 'G', 'I']:
            x = np.linspace(x_start, x_end, int(max(100, 30 * length)))
            period = min(3.6, length if length > 0 else 1.0)
            taper_len = min(1.0, length / 2)
            taper = np.ones_like(x)
            if taper_len > 0:
                for i, xv in enumerate(x):
                    if xv < x_start + taper_len: taper[i] = 0.5 - 0.5 * np.cos(np.pi * (xv - x_start) / taper_len)
                    elif xv > x_end - taper_len: taper[i] = 0.5 - 0.5 * np.cos(np.pi * (x_end - xv) / taper_len)
            y = 0.5 + 0.35 * np.sin(2 * np.pi * (x - x_start) / period) * taper
            ax.plot(x, y, color=color, linewidth=3.5, solid_capstyle='round', zorder=4,
                    path_effects=[pe.Stroke(linewidth=5.5, foreground='black'), pe.Normal()])
        elif state in ['E', 'B']:
            head_len = min(1.0, length / 2) if length > 0.5 else length
            tail_len = length - head_len
            hw = 0.35 
            tw = 0.20 
            if length > head_len:
                verts = [(x_start, 0.5 - tw), (x_start + tail_len, 0.5 - tw), (x_start + tail_len, 0.5 - hw),
                         (x_end, 0.5), (x_start + tail_len, 0.5 + hw), (x_start + tail_len, 0.5 + tw), (x_start, 0.5 + tw)]
            else:
                verts = [(x_start, 0.5 - hw), (x_end, 0.5), (x_start, 0.5 + hw)]
            ax.add_patch(patches.Polygon(verts, facecolor=color, edgecolor='black', linewidth=1.5, zorder=4))
        else:
            ax.plot([x_start, x_end], [0.5, 0.5], color=color, linewidth=3.5, solid_capstyle='butt', zorder=3,
                    path_effects=[pe.Stroke(linewidth=5.5, foreground='black'), pe.Normal()])

def align_matrices(df1, df2):
    all_chars = sorted(list(set(df1.columns) | set(df2.columns)))
    return df1.reindex(columns=all_chars, fill_value=0), df2.reindex(columns=all_chars, fill_value=0)

def generate_comparative_logos(records, group_a_ids, group_b_ids, dir_predicted, output_path, highlight_threshold=0.6, ss_predictor="netsurfp"):
    print("\n--- Generating Comparative Sequence Logos with Raw Consensus SS ---")
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

        # Get Raw Consensus and structure count
        ss_cons_a, num_a = get_consensus_structure(seqs_a, valid_ids_a, dir_predicted, ss_predictor)
        ss_cons_b, num_b = get_consensus_structure(seqs_b, valid_ids_b, dir_predicted, ss_predictor)

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
        
        # Restore the title format
        ax_logo_a.set_title(f"Group A ({len(seqs_a)} sequences | SS Consensus: {num_a} structures)", fontsize=14, fontweight='bold')
        ax_logo_b.set_title(f"Group B ({len(seqs_b)} sequences | SS Consensus: {num_b} structures)", fontsize=14, fontweight='bold')
        ax_logo_a.set_ylabel("Bits")
        ax_logo_b.set_ylabel("Bits")

        draw_biological_ss_track(ax_ss_a, ss_cons_a)
        draw_biological_ss_track(ax_ss_b, ss_cons_b)

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
        print(f"Saved Comparative Sequence Logos (Raw Consensus): {output_path}")

    except Exception as e:
        print(f"Error generating Logo plot: {e}")
