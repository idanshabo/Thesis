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

    viz_dir = os.path.join(sig_split_folder, "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    output_plot = os.path.join(viz_dir, "ordered_split_MSA.png")
    
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


def get_clustered_order(df, ids, method='ward'):
    """
    Performs hierarchical clustering on a subset of the dataframe 
    and returns the IDs in the sorted order.
    """
    # If 0 or 1 item, no sorting needed
    if len(ids) < 2:
        return ids
    
    # Extract the sub-matrix for this group
    sub_df = df.loc[ids, ids]
    
    try:
        # 1. Calculate linkage
        # We use the sub-df itself as features. Rows with similar covariance 
        # profiles across the group will be clustered together.
        # 'ward' minimizes variance within clusters.
        Z = linkage(sub_df, method=method)
        
        # 2. Get the order of leaves (the sorted indices)
        ordered_indices = leaves_list(Z)
        
        # 3. Map back to original IDs
        return sub_df.index[ordered_indices].tolist()
    except Exception as e:
        print(f"Warning: Clustering failed for group (size {len(ids)}). Keeping original order. Error: {e}")
        return ids

def plot_split_covariance(cov_matrix_path, split_info, sig_split_folder, sort_groups=True):
    """
    Generates a heatmap of a covariance matrix sorted by groups defined in split_info.
    
    Parameters:
    - sort_groups (bool): If True, performs hierarchical clustering within Group A 
                          and Group B independently to reveal structure.
    """
    
    # 1. Load the Covariance Matrix
    try:
        df_cov = pd.read_csv(cov_matrix_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: The file at {cov_matrix_path} was not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Extract and Validate IDs
    raw_group_a = split_info.get('group_a', [])
    raw_group_b = split_info.get('group_b', [])

    valid_group_a = [uid for uid in raw_group_a if uid in df_cov.index]
    valid_group_b = [uid for uid in raw_group_b if uid in df_cov.index]

    if not valid_group_a and not valid_group_b:
        print("Error: None of the IDs in split_info were found in the covariance matrix.")
        return

    # 3. Reorder the Groups (Clustering)
    if sort_groups:
        print("Clustering Group A...")
        sorted_group_a = get_clustered_order(df_cov, valid_group_a)
        print("Clustering Group B...")
        sorted_group_b = get_clustered_order(df_cov, valid_group_b)
    else:
        sorted_group_a = valid_group_a
        sorted_group_b = valid_group_b

    # Concatenate the lists: Group A first, then Group B
    ordered_ids = sorted_group_a + sorted_group_b
    
    # Subset and reorder the dataframe
    df_ordered = df_cov.loc[ordered_ids, ordered_ids]

    # 4. Plot Setup
    plt.figure(figsize=(10, 8))
    
    # Draw the heatmap
    sns.heatmap(df_ordered, cmap='viridis', xticklabels=False, yticklabels=False)

    # 5. Add Separation Lines and Labels
    split_index = len(sorted_group_a)
    total_len = len(ordered_ids)

    # Draw white separation lines
    plt.axvline(x=split_index, color='white', linewidth=2, linestyle='-')
    plt.axhline(y=split_index, color='white', linewidth=2, linestyle='-')

    # Add Group Labels
    # X-axis labels (bottom)
    if sorted_group_a:
        plt.text(split_index / 2, total_len + (total_len * 0.02), 
                 f'Group A (n={len(sorted_group_a)})', 
                 ha='center', va='top', fontsize=12, weight='bold')
    if sorted_group_b:
        plt.text(split_index + (len(sorted_group_b) / 2), total_len + (total_len * 0.02), 
                 f'Group B (n={len(sorted_group_b)})', 
                 ha='center', va='top', fontsize=12, weight='bold')

    # Y-axis labels (left)
    if sorted_group_a:
        plt.text(- (total_len * 0.02), split_index / 2, 
                 'Group A', ha='right', va='center', rotation=90, fontsize=12, weight='bold')
    if sorted_group_b:
        plt.text(- (total_len * 0.02), split_index + (len(sorted_group_b) / 2), 
                 'Group B', ha='right', va='center', rotation=90, fontsize=12, weight='bold')

    # 6. Titles and Output
    node_name = split_info.get('node_name', 'Unknown Node')
    rank = split_info.get('rank', 'Unknown Rank')
    
    sort_status = "Clustered" if sort_groups else "Unsorted"
    plt.title(f'Covariance Structure ({sort_status})\nNode: {node_name} | Rank: {rank}', fontsize=14, pad=20)
    plt.tight_layout()

    viz_dir = os.path.join(sig_split_folder, "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Updated filename to reflect sorting
    filename = "proteins_covariance_plot_clustered.png" if sort_groups else "proteins_covariance_plot.png"
    output_path = os.path.join(viz_dir, filename)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Proteins Covariance Plot saved to {output_path}")


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
    Automatically deduces all file paths from the folder_path and runs the analysis.
    Expected Structure: .../{ProteinID}/splits_evaluations/significant_splits/{SplitName}/
    """
    # --- 1. Path Deduction ---
    clean_path = folder_path.rstrip(os.sep)
    split_name = os.path.basename(clean_path) # e.g., "rank5_0.960"
    split_num = split_name.split("_")[0]
    
    # Navigate up 3 levels to find the Protein ID root folder
    # Level 0: .../rank5_0.960
    # Level 1: .../significant_splits
    # Level 2: .../splits_evaluations
    # Level 3: .../PF03869 (Protein Root)
    
    sig_splits_dir = os.path.dirname(clean_path)
    splits_eval_dir = os.path.dirname(sig_splits_dir)
    protein_outputs_dir = os.path.dirname(splits_eval_dir)
    protein_root_dir = os.path.dirname(protein_outputs_dir)

    PF03869_calculations_dir = 
    protein_id = os.path.basename(protein_root_dir) # e.g., "PF03869_calculations"
    
    print(f"Detected Protein ID: {protein_id}")
    
    # Construct paths
    full_cov_name = f"{protein_id}_global_H0_PCA_embeddings_cov_mat.csv"
    full_cov_path = os.path.join(protein_root_dir, full_cov_name)
    
    child1_path = os.path.join(clean_path, f"{split_name}_{split_num}_subA_embeddings_cov_mat.csv")
    child2_path = os.path.join(clean_path, f"{split_name}_{split_num}_subB_embeddings_cov_mat.csv")
    
    # Visualization Output
    viz_dir = os.path.join(clean_path, "visualization")
    output_path = os.path.join(viz_dir, "variance_spectrum.png")

    print(f"Full Cov Path deduced as: {full_cov_path}")

    # --- 2. Setup & Load ---
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

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
