import os
import json
import torch
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from itertools import groupby
from evaluate_split_options.evaluate_split_options import PhylogeneticPCA

def plot_global_subfamilies(global_cov_ordered_path, subfamilies_json_path, output_dir):
    """
    Plots the global covariance matrix and overlays boxes/lines showing 
    the stable sub-families identified by the mean-shift ANOVA.
    """
    if not os.path.exists(global_cov_ordered_path) or not os.path.exists(subfamilies_json_path):
        print("Missing files for global subfamily visualization. Skipping.")
        return

    # Load data
    df_cov = pd.read_csv(global_cov_ordered_path, index_col=0)
    with open(subfamilies_json_path, 'r') as f:
        subfamilies_dict = json.load(f) # Expected format: {"subfamily_1": ["seq1", "seq2"], ...}

    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # Draw Global Heatmap
    sns.heatmap(df_cov, cmap='viridis', cbar=True, 
                xticklabels=False, yticklabels=False, square=True, ax=ax)

    # Find boundaries for each subfamily based on the tree-ordered index
    ordered_ids = list(df_cov.index)
    
    for sf_name, data in subfamilies_dict.items():
        # Handle the new nested dictionary format (while keeping old runs safe)
        leaves = data.get("leaves", []) if isinstance(data, dict) else data
        
        # Clean leaves to match index format
        clean_leaves = {str(leaf).replace("/", "_") for leaf in leaves}
        
        # Find start and end indices in the global matrix
        indices = [i for i, uid in enumerate(ordered_ids) if str(uid).replace("/", "_") in clean_leaves]
        
        if not indices:
            continue
            
        start = min(indices)
        end = max(indices) + 1 # +1 to cover the last cell
        size = end - start
        
        # Draw a bounding box around the sub-family on the diagonal
        rect = Rectangle((start, start), size, size, fill=False, edgecolor='white', linestyle='-', linewidth=2.5, alpha=0.9)
        ax.add_patch(rect)
        
        # Add Label inside or near the box
        center = start + (size / 2)
        ax.text(center, start - 1, sf_name.replace("_", " ").title(), 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    plt.title("Global Family Covariance\nDivided by Stable Mean-Shift Sub-Families", fontsize=16, pad=20)
    
    output_path = os.path.join(output_dir, "global_mean_shift_subfamilies.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Global sub-families plot saved to: {output_path}")

def plot_global_mean_shift_ppca(global_embeddings_path, global_cov_path, subfamilies_json_path, output_dir):
    """
    Plots the global embeddings in 2D Phylogenetic PCA (pPCA) space, 
    coloring each point by its assigned stable mean-shift sub-family.
    """  
    if not os.path.exists(global_embeddings_path) or not os.path.exists(subfamilies_json_path):
        print("Missing files for global mean-shift pPCA. Skipping.")
        return

    # 1. Load sub-family assignments
    with open(subfamilies_json_path, 'r') as f:
        subfamilies_dict = json.load(f)
        
    id_to_subfamily = {}
    for sf_name, data in subfamilies_dict.items():
        # Handle the new nested dictionary format
        leaves = data.get("leaves", []) if isinstance(data, dict) else data
        
        for leaf in leaves:
            clean_leaf = str(leaf).replace('/', '_')
            id_to_subfamily[clean_leaf] = sf_name

    # 2. Load aligned global embeddings
    try:
        data = torch.load(global_embeddings_path, map_location='cpu')
        embeddings_array = data['embeddings'].cpu().numpy() if hasattr(data['embeddings'], 'cpu') else np.array(data['embeddings'])
        protein_names = data.get('file_names') or data.get('names') or data.get('ids')
    except Exception as e:
        print(f"Error loading global embeddings: {e}")
        return
        
    # 3. Load global covariance matrix (Must align with embeddings!)
    try:
        df_cov = pd.read_csv(global_cov_path, index_col=0)
        cov_matrix = df_cov.values
    except Exception as e:
        print(f"Error loading global covariance matrix: {e}")
        return

    # 4. Assign labels
    labels = []
    for name in protein_names:
        clean_name = str(name).replace('/', '_')
        labels.append(id_to_subfamily.get(clean_name, "Unknown"))
    y = np.array(labels)

    # 5. Fit Phylogenetic PCA
    print(f"Performing Global Phylogenetic PCA on {len(embeddings_array)} proteins...")
    # Force min_components to 2 just for the 2D visualization
    ppca = PhylogeneticPCA(min_components=2, mode='cov') 
    ppca.fit(embeddings_array, cov_matrix)
    
    # Transform to pPCA space
    X_ppca = ppca.transform(embeddings_array)
    
    # 6. Plotting
    plt.figure(figsize=(10, 8))
    unique_labels = sorted([lbl for lbl in set(y) if lbl != "Unknown"], key=lambda x: int(x.split('_')[1]) if '_' in x else x)
    cmap = plt.get_cmap('tab10') 
    
    for i, label in enumerate(unique_labels):
        idx = np.where(y == label)
        display_name = label.replace('_', ' ').title()
        plt.scatter(X_ppca[idx, 0], X_ppca[idx, 1], 
                    alpha=0.8, s=50, label=f"{display_name} (n={len(idx[0])})",
                    color=cmap(i % 10), edgecolors='w', linewidth=0.5)
                    
    plt.title("Global Phylogenetic PCA: Mean-Shift Sub-Families", fontsize=16)
    plt.xlabel("pPC1", fontsize=12)
    plt.ylabel("pPC2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # 7. Save
    output_plot = os.path.join(output_dir, "global_mean_shift_ppca.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Global Mean-Shift pPCA saved to: {output_plot}")
    
def visualize_split_msa_sorted(fasta_path, split_info, sig_split_folder):
    """
    Visualizes an MSA split with hierarchical clustering.
    Handles large datasets by disabling labels and capping image size.
    """
    
    # 1. Load MSA
    try:
        msa_records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print(f"Error: FASTA file {fasta_path} not found.")
        return

    # Map ID -> Sequence (Sanitized)
    id_to_seq = {rec.id.replace('/', '_'): str(rec.seq) for rec in msa_records}
    
    # 2. Define Groups with Robust ID Sanitization
    def get_matching_ids(split_group, id_dict):
        matched = []
        for x in split_group:
            clean_x = str(x).replace('/', '_')
            if clean_x in id_dict:
                matched.append(clean_x)
            elif str(x) in id_dict:
                matched.append(str(x))
        return matched

    group_a_ids = get_matching_ids(split_info['group_a'], id_to_seq)
    group_b_ids = get_matching_ids(split_info['group_b'], id_to_seq)

    print(f"Loaded {len(id_to_seq)} sequences from FASTA.")
    
    if not group_a_ids or not group_b_ids:
        print("Error: One group is empty. Check ID matching.")
        return

    # --- HELPER: Turn sequences into Integer Matrix ---
    def ids_to_matrix(id_list):
        seqs = [list(id_to_seq[i]) for i in id_list]
        if not seqs: return np.array([]), {}, 0
        min_len = min(len(s) for s in seqs)
        matrix = []
        
        all_chars = set(char for s in seqs for char in s)
        char_map = {c: i for i, c in enumerate(sorted(all_chars))}
        
        for s in seqs:
            row = [char_map.get(c, 0) for c in s[:min_len]]
            matrix.append(row)
        return np.array(matrix), char_map, min_len

    # --- HELPER: Cluster and Sort Indices ---
    def get_clustered_order(id_list):
        if len(id_list) < 3: return id_list 
        mat, _, _ = ids_to_matrix(id_list)
        if mat.size == 0: return id_list

        try:
            dist_vec = ssd.pdist(mat, metric='hamming')
            linkage_matrix = sch.linkage(dist_vec, method='ward')
            dendro = sch.dendrogram(linkage_matrix, no_plot=True)
            return [id_list[i] for i in dendro['leaves']]
        except Exception as e:
            print(f"Clustering warning: {e}")
            return id_list

    print("Sorting Group A...")
    sorted_ids_a = get_clustered_order(group_a_ids)
    print("Sorting Group B...")
    sorted_ids_b = get_clustered_order(group_b_ids)

    # 3. Combine for Plotting
    final_ids = sorted_ids_a + sorted_ids_b
    labels = [f"{i} (A)" for i in sorted_ids_a] + [f"{i} (B)" for i in sorted_ids_b]
    
    # Build final plot matrix
    all_seqs_chars = set()
    for i in final_ids:
        all_seqs_chars.update(id_to_seq[i])
    final_char_map = {c: i for i, c in enumerate(sorted(all_seqs_chars))}
    
    seq_len = len(id_to_seq[final_ids[0]])
    plot_matrix = []
    for i in final_ids:
        s = id_to_seq[i][:seq_len]
        row = [final_char_map.get(c, 0) for c in s]
        plot_matrix.append(row)
    plot_data = np.array(plot_matrix)

    # 4. Plotting (OPTIMIZED FOR LARGE DATA)
    num_seqs = len(final_ids)
    
    # Threshold: If more than 250 sequences, switch to "Overview Mode"
    LARGE_DATA_THRESHOLD = 250 
    
    if num_seqs > LARGE_DATA_THRESHOLD:
        print(f"Large dataset detected ({num_seqs} seqs). Switching to condensed overview mode.")
        # Fixed reasonable height for overview (e.g., 10 inches)
        fig_height = 12
        show_labels = False
        dpi_val = 300 # Higher DPI for crisp lines in condensed view
    else:
        # Original logic for small datasets where labels are readable
        fig_height = max(6, num_seqs * 0.25)
        show_labels = True
        dpi_val = 150

    fig, ax = plt.subplots(figsize=(15, fig_height))
    
    cmap = plt.get_cmap('tab20b', len(final_char_map))
    
    # aspect='auto' is CRITICAL for large data to fill the space without distortion
    im = ax.imshow(plot_data, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Draw Split Line
    split_pos = len(sorted_ids_a) - 0.5
    ax.axhline(y=split_pos, color='white', linewidth=2, linestyle='-') 
    ax.axhline(y=split_pos, color='black', linewidth=1, linestyle='--', label='Phylogenetic Split')

    if show_labels:
        ax.set_yticks(np.arange(num_seqs))
        label_fontsize = 10 if num_seqs < 50 else 8
        ax.set_yticklabels(labels, fontsize=label_fontsize)
    else:
        # Hide y-ticks for large data, just show boundary markers
        ax.set_yticks([0, len(sorted_ids_a), num_seqs-1])
        ax.set_yticklabels(["Start (A)", "Split Boundary", "End (B)"])
        ax.tick_params(axis='y', length=0) # Hide tick marks

    ax.set_xlabel("Alignment Position")
    ax.set_title(f"Re-ordered MSA View (n={num_seqs})")

    output_plot = os.path.join(sig_split_folder, "ordered_split_MSA.png")
    
    plt.tight_layout()
    
    # Save logic
    try:
        plt.savefig(output_plot, dpi=dpi_val, bbox_inches='tight')
        print(f"Visualization saved to: {output_plot}")
    except ValueError as e:
        print(f"Error saving image (likely too large): {e}")
        
    plt.close()


def visualize_newick_tree(file_path: str, split_info=None, save_image: bool = True, output_image_path: str = None):
    """
    Reads a phylogenetic tree from a Newick file, visualizes it using biopython and matplotlib,
    and saves the image. Marks the split if split_info is provided.
    """
    try:
        # Load the tree
        tree = Phylo.read(file_path, "newick")

        # --- SPLIT MARKING LOGIC ---
        split_clade = None
        if split_info:
            # A split divides the tree into Group A and Group B. 
            # In a rooted view, one of these groups will correspond exactly to a clade 
            # (the branch leading to it is the split).
            target_a = set(split_info['group_a'])
            target_b = set(split_info['group_b'])

            for clade in tree.find_clades():
                # Get all leaves under this clade
                leaves = set(term.name for term in clade.get_terminals())
                
                # Check if this clade matches either side of the split
                if leaves == target_a or leaves == target_b:
                    split_clade = clade
                    # Mark the branch visually
                    clade.color = 'red'
                    clade.width = 3
                    break
        # ---------------------------

        # Count clades for scaling
        def count_clades(clade):
            count = 1 
            for subclade in clade:
                count += count_clades(subclade)
            return count

        num_clades = count_clades(tree.root)
        figsize = (15, max(6, num_clades / 3)) # Adjusted minimum height

        fig, ax = plt.subplots(figsize=figsize)

        # Custom label function
        def get_label(clade):
            if clade.is_terminal() and clade.name:
                return clade.name
            return ''
        
        # Custom branch label function to show support or "SPLIT"
        def get_branch_label(c):
            if c == split_clade:
                return " <<< SPLIT MARK"
            if c.branch_length is not None:
                return f"{c.branch_length:.2f}"
            return ""

        # Draw the tree
        Phylo.draw(tree, axes=ax, do_show=False,
                   branch_labels=get_branch_label,
                   label_func=get_label)

        ax.set_title(f"Phylogenetic Tree {'(Split Marked)' if split_clade else ''}", fontsize=16)

        # Force updating the font/color of the split branch if needed, 
        # though Biopython handles color attributes automatically in recent versions.

        plt.tight_layout()
        
        if not output_image_path:
            output_image_path = os.path.splitext(file_path)[0] + "_tree_view.png"
            
        if save_image:
            plt.savefig(output_image_path, bbox_inches="tight")
            print(f"Tree visualization saved to '{output_image_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred in tree viz: {e}")


def visualize_embeddings_pca(embeddings_path, split_info, output_plot="pca_split_view.png"):
    """
    Loads PyTorch embeddings, performs PCA, and visualizes them colored by the split.
    Handles name mismatch (Tree uses '/', Embeddings use '_').
    """
    print(f"Loading embeddings from {embeddings_path}...")
    try:
        data = torch.load(embeddings_path)
        # Depending on how it was saved, it might be on GPU or CPU. mapping to cpu is safer.
        if isinstance(data['embeddings'], torch.Tensor):
            embeddings_array = data['embeddings'].cpu().numpy()
        else:
            embeddings_array = data['embeddings'] # Assuming it might already be numpy
            
        protein_names = data['file_names']
    except Exception as e:
        print(f"Error loading embedding file: {e}")
        return

    # Normalize split groups to match embedding naming convention (Replace '/' with '_')
    # We use sets for O(1) lookup
    group_a_ids = {name.replace("/", "_") for name in split_info['group_a']}
    group_b_ids = {name.replace("/", "_") for name in split_info['group_b']}

    filtered_embeddings = []
    labels = []
    
    # Iterate through embedding file names and assign to Group A or B
    for i, name in enumerate(protein_names):
        # Check exact match after normalization
        if name in group_a_ids:
            filtered_embeddings.append(embeddings_array[i])
            labels.append("Group A")
        elif name in group_b_ids:
            filtered_embeddings.append(embeddings_array[i])
            labels.append("Group B")
            
    if not filtered_embeddings:
        print("WARNING: No matching protein names found between Embeddings and Tree Split.")
        print(f"Sample Tree ID (normalized): {list(group_a_ids)[0] if group_a_ids else 'None'}")
        print(f"Sample Embedding ID: {protein_names[0] if protein_names else 'None'}")
        return

    X = np.array(filtered_embeddings)
    y = np.array(labels)
    
    print(f"Extracting top 2 pPCA dimensions for {len(X)} matched proteins...")
    
    if X.shape[1] < 2:
        print("Warning: Embeddings have less than 2 dimensions. Cannot plot 2D pPCA.")
        return
        
    # Since X is already processed by PhylogeneticPCA, the first two columns are our top components
    X_pca = X[:, :2]
    
    # Plotting
    plt.figure(figsize=(8, 8))
    
    # Scatter plot for Group A
    idx_a = np.where(y == "Group A")
    plt.scatter(X_pca[idx_a, 0], X_pca[idx_a, 1], 
                c='red', alpha=0.7, s=50, label=f"Group A (n={len(idx_a[0])})")
    
    # Scatter plot for Group B
    idx_b = np.where(y == "Group B")
    plt.scatter(X_pca[idx_b, 0], X_pca[idx_b, 1], 
                c='blue', alpha=0.7, s=50, label=f"Group B (n={len(idx_b[0])})")
    
    # Cosmetics
    plt.title(f"Phylogenetic PCA of Embeddings")
    plt.xlabel("pPCA Dimension 1")
    plt.ylabel("pPCA Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"PCA visualization saved to: {output_plot}")


def get_group_intervals(ordered_ids, group_a_set, group_b_set):
    """
    Scans the ordered list and identifies contiguous blocks.
    Returns list of tuples: (Label 'A'/'B', StartIndex, EndIndex)
    """
    labels = []
    for uid in ordered_ids:
        if uid in group_a_set: labels.append('A')
        elif uid in group_b_set: labels.append('B')
        else: labels.append('X') # 'X' = Not in either group (if partial split)
        
    intervals = []
    current_idx = 0
    
    # Identify contiguous blocks (e.g. A-A-A -> B-B -> A-A)
    for label, group in groupby(labels):
        length = len(list(group))
        end_idx = current_idx + length
        intervals.append((label, current_idx, end_idx))
        current_idx = end_idx
        
    return intervals

def plot_split_covariance(ordered_cov_path, split_info, sig_split_folder):
    """
    Plots the covariance matrix (preserving input order).
    Draws dashed lines at the boundaries between Group A and Group B.
    """
    # 1. Load Data
    try:
        df_cov = pd.read_csv(ordered_cov_path, index_col=0)
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return

    # 2. Identify Blocks
    group_a_ids = set(split_info.get('group_a', []))
    group_b_ids = set(split_info.get('group_b', []))
    
    # Calculate where the boundaries are in the CURRENT order
    intervals = get_group_intervals(df_cov.index, group_a_ids, group_b_ids)

    # 3. Setup Plot
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Draw Heatmap
    sns.heatmap(df_cov, cmap='viridis', cbar=True, 
                xticklabels=False, yticklabels=False, square=True, ax=ax)

    # 4. Add Dashed Lines and Labels
    # We iterate through the blocks we found
    for label, start, end in intervals:
        
        # A. Draw Lines at the END of the block (Separators)
        # We don't draw a line at the very end of the matrix
        if end < len(df_cov.index):
            # Use white or light grey for visibility on dark viridis
            ax.axvline(x=end, color='white', linestyle='--', linewidth=1.5, alpha=0.9)
            ax.axhline(y=end, color='white', linestyle='--', linewidth=1.5, alpha=0.9)

        # B. Add Labels (Centered on the block)
        # Only label if the block is significant (> 2% of map to avoid clutter)
        size = end - start
        if size > (len(df_cov.index) * 0.02):
            center = start + (size / 2)
            group_name = "Group A" if label == 'A' else "Group B"
            if label == 'X': group_name = "Other"

            # Top Axis Label
            ax.text(center, -0.5, group_name, 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', rotation=45)
            
            # Left Axis Label
            ax.text(-0.5, center, group_name, 
                    ha='right', va='center', fontsize=11, fontweight='bold')

    # 5. Titles and Save
    rank = split_info.get('rank', 'Unknown Rank')
    
    plt.title(f'Covariance Structure\n(Rank: {rank})', fontsize=14, pad=30)
    
    # Save   
    output_path = os.path.join(sig_split_folder, "covariance_with_split_lines.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Covariance plot saved to {output_path}")

def plot_side_by_side_embedding_covariance(folder_path, split_info):
    """
    Plots the Global, Group A, and Group B embedding covariance matrices side-by-side.
    Uses SymLogNorm and zooms into the top principal components for maximum visual clarity.
    """
    # --- 1. Path Deduction ---
    split_name = os.path.basename(folder_path)
    sig_splits_dir = os.path.dirname(folder_path)
    sf_dir = os.path.dirname(sig_splits_dir) 
    embed_dir = os.path.dirname(sf_dir)  # e.g., 'sequence_embeddings'
    protein_outputs_dir = os.path.dirname(embed_dir)
    
    dir_name = os.path.basename(protein_outputs_dir)
    protein_id = dir_name.replace("_outputs", "")
    
    protein_data_root = os.path.dirname(protein_outputs_dir)
    calc_dir = os.path.join(protein_data_root, f"{protein_id}_calculations")
    
    sf_name = os.path.basename(sf_dir)
    embed_name = os.path.basename(embed_dir)
    
    # --- CHANGED: Insert embed_name into the path ---
    full_cov_path = os.path.join(calc_dir, embed_name, sf_name, f"{sf_name}_global_H0_PCA_cov_mat.csv")
    
    child_a_path = os.path.join(folder_path, "calculations", f"embedding_cov_{split_name}_subA.csv")
    child_b_path = os.path.join(folder_path, "calculations", f"embedding_cov_{split_name}_subB.csv")
    output_path = os.path.join(folder_path, "embedding_covariances_comparison.png")

    # --- 2. Setup & Load ---
    try:
        cov_global = pd.read_csv(full_cov_path, index_col=0)
        cov_a = pd.read_csv(child_a_path, index_col=0)
        cov_b = pd.read_csv(child_b_path, index_col=0)
        
        cov_global = cov_global.dropna(axis=1, how='all').dropna(axis=0, how='all')
        cov_a = cov_a.dropna(axis=1, how='all').dropna(axis=0, how='all')
        cov_b = cov_b.dropna(axis=1, how='all').dropna(axis=0, how='all')
    except Exception as e:
        print(f"Error loading embedding covariance matrices: {e}")
        return

    # --- IMPROVEMENT 1: Zoom in on Active Variance ---
    k_max = 50
    k = min(k_max, cov_global.shape[0])
    
    cov_global_sub = cov_global.iloc[:k, :k]
    cov_a_sub = cov_a.iloc[:k, :k]
    cov_b_sub = cov_b.iloc[:k, :k]

    # --- IMPROVEMENT 2: Calculate Frobenius Distance ---
    dist_a = np.linalg.norm(cov_a_sub.values - cov_global_sub.values, 'fro')
    dist_b = np.linalg.norm(cov_b_sub.values - cov_global_sub.values, 'fro')

    # --- IMPROVEMENT 3: SymLogNorm for Visual Resolution ---
    all_values = np.concatenate([
        cov_global_sub.values.flatten(), 
        cov_a_sub.values.flatten(), 
        cov_b_sub.values.flatten()
    ])
    robust_max = np.nanpercentile(np.abs(all_values), 98)
    
    norm = mcolors.SymLogNorm(linthresh=0.01, linscale=1.0, vmin=-robust_max, vmax=robust_max, base=10)
    cmap = 'RdBu_r' 

    # --- 4. Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(cov_global_sub, cmap=cmap, norm=norm, square=True, ax=axes[0], 
                cbar=False, xticklabels=False, yticklabels=False)
    axes[0].set_title(f"Global (H0)\np=Top {k}", fontsize=14)
    
    sns.heatmap(cov_a_sub, cmap=cmap, norm=norm, square=True, ax=axes[1], 
                cbar=False, xticklabels=False, yticklabels=False)
    axes[1].set_title(f"Group A\np=Top {k}\nDistance from H0: {dist_a:.1f}", fontsize=14)
    
    sns.heatmap(cov_b_sub, cmap=cmap, norm=norm, square=True, ax=axes[2], 
                cbar=True, xticklabels=False, yticklabels=False)
    axes[2].set_title(f"Group B\np=Top {k}\nDistance from H0: {dist_b:.1f}", fontsize=14)

    plt.suptitle(f"Embedding Feature Covariances (Top {k} pPCA Dimensions)", fontsize=16, y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def load_matrix(data):
    """
    Helper: Loads data using pandas for robustness.
    """
    if isinstance(data, str):
        try:
            # header=None assumes no column names.
            df = pd.read_csv(data, header=None)
            # Drop columns that are entirely NaN (handling trailing commas)
            df = df.dropna(axis=1, how='all')
            return df.values
        except Exception as e:
            print(f"Error loading {data}: {e}")
            raise
    return data


def plot_variance_spectrum_helper(ax, cov_matrix, label, color=None):
    """
    Helper: Plots the diagonal (Variances) on a log scale.
    """
    cov_matrix = np.array(cov_matrix, dtype=float)
    variances = np.diag(cov_matrix).copy()
    variances[variances <= 0] = np.nan
    
    ax.plot(variances, label=label, marker='.', markersize=2, alpha=0.7, color=color)
    ax.set_yscale('log')

    
def run_variance_analysis(folder_path):
    """
    Automatically deduces paths based on the specific folder structure provided.
    """
    # --- 1. Path Deduction ---
    split_name = os.path.basename(folder_path)

    sig_splits_dir = os.path.dirname(folder_path)
    sf_dir = os.path.dirname(sig_splits_dir) 
    embed_dir = os.path.dirname(sf_dir)
    protein_outputs_dir = os.path.dirname(embed_dir)
    
    dir_name = os.path.basename(protein_outputs_dir)
    protein_id = dir_name.replace("_outputs", "")
    
    print(f"Detected Protein ID: {protein_id}")
    print(f"Detected Split Name: {split_name}")

    protein_data_root = os.path.dirname(protein_outputs_dir)
    calc_dir = os.path.join(protein_data_root, f"{protein_id}_calculations")
    
    sf_name = os.path.basename(sf_dir) 
    embed_name = os.path.basename(embed_dir)
    
    # Insert embed_name into the path ---
    full_cov_path = os.path.join(calc_dir, embed_name, sf_name, f"{sf_name}_global_H0_PCA_cov_mat.csv")
    
    child1_path = os.path.join(folder_path, "calculations", f"embedding_cov_{split_name}_subA.csv")
    child2_path = os.path.join(folder_path, "calculations", f"embedding_cov_{split_name}_subB.csv")
    
    output_path = os.path.join(folder_path, "variance_spectrum.png")

    # --- 2. Setup & Load ---
    if not os.path.exists(full_cov_path):
        print(f"Warning: Could not find global matrix at: {full_cov_path}")
        return

    full_cov = load_matrix(full_cov_path)
    child1_cov = load_matrix(child1_path)
    child2_cov = load_matrix(child2_path)

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_variance_spectrum_helper(ax, full_cov, "Full Model", color='black')
    plot_variance_spectrum_helper(ax, child1_cov, "Child A", color='tab:blue')
    plot_variance_spectrum_helper(ax, child2_cov, "Child B", color='tab:orange')

    ax.set_xlabel("PCA Dimension", fontsize=12)
    ax.set_ylabel("Variance (Log Scale)", fontsize=12)
    ax.set_title(f"Variance Spectrum: {protein_id} / {split_name}", fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
