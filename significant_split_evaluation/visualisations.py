import os
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

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
