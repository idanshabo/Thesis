import os
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
from itertools import groupby


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
    
    # 2. Define Groups
    group_a_ids = [x for x in split_info['group_a'] if x in id_to_seq]
    group_b_ids = [x for x in split_info['group_b'] if x in id_to_seq]

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
    ax.set_title(f"Re-ordered MSA View: {split_info.get('node_name', 'Split')} (n={num_seqs})")

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
    
    print(f"Performing PCA on {len(X)} matched proteins...")
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
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
    plt.title(f"PCA of Protein Embeddings (Split Support: {split_info['support']})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
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
    node_name = split_info.get('node_name', 'Unknown Node')
    rank = split_info.get('rank', 'Unknown Rank')
    
    plt.title(f'Covariance Structure: {node_name}\n(Rank: {rank})', fontsize=14, pad=30)
    
    # Save   
    output_path = os.path.join(sig_split_folder, "covariance_with_split_lines.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Covariance plot saved to {output_path}")


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
    # Extract Split Name (e.g., "rank5") directly from the folder name
    split_name = os.path.basename(folder_path)

    # Navigate up to find the Protein Output directory
    # Current: .../{pf}_outputs/splits_evaluations/significant_splits/{split}
    sig_splits_dir = os.path.dirname(folder_path)
    protein_outputs_dir = os.path.dirname(sig_splits_dir) # This ends in {pf}_outputs
    
    # Extract Protein ID by stripping "_outputs"
    # e.g., "PF07361_outputs" -> "PF07361"
    dir_name = os.path.basename(protein_outputs_dir)
    protein_id = dir_name.replace("_outputs", "")
    
    print(f"Detected Protein ID: {protein_id}")
    print(f"Detected Split Name: {split_name}")

    # Navigate to the parallel 'calculations' directory
    # Root is one level up from protein_outputs_dir
    protein_data_root = os.path.dirname(protein_outputs_dir)
    calc_dir = os.path.join(protein_data_root, f"{protein_id}_calculations")
    
    # Define Full File Paths
    full_cov_name = f"{protein_id}_calculations_global_H0_PCA_embeddings_cov_mat.csv"
    full_cov_path = os.path.join(calc_dir, full_cov_name)
    
    child1_path = os.path.join(folder_path, f"calculations/embedding_cov_{split_name}_subA.csv")
    child2_path = os.path.join(folder_path, f"calculations/embedding_cov_{split_name}_subB.csv")
    
    # Visualization Output
    output_path = os.path.join(folder_path, "variance_spectrum.png")

    print(f"Global Cov Path: {full_cov_path}")
    print(f"Child A Path:    {child1_path}")

    # --- 2. Setup & Load ---
    if not os.path.exists(full_cov_path):
        raise FileNotFoundError(f"Could not find global matrix at: {full_cov_path}")

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
    
    print(f"Saved plot to: {output_path}")
