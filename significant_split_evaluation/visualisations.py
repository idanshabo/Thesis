from Bio import SeqIO, Phylo
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.decomposition import PCA


def visualize_split_msa(fasta_path, split_info, output_plot="msa_split_view.png"):
    """
    Reads a FASTA file, reorders sequences based on the split (Group A then Group B),
    and visualizes the alignment with a horizontal dividing line.
    """
    
    # 1. Load MSA
    try:
        # We use list() to get all records into memory
        msa_records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print(f"Error: FASTA file {fasta_path} not found.")
        return

    # 2. Create dictionaries for fast access
    # Map ID -> Sequence String
    id_to_seq = {rec.id: str(rec.seq) for rec in msa_records}
    
    # 3. Separate records into Group A and Group B
    group_a_ids = split_info['group_a']
    group_b_ids = split_info['group_b']
    
    # Filter: Ensure we only graph sequences that exist in both the Tree and the FASTA
    # (Sometimes tree leaf names differ slightly or one file is a subset of the other)
    sorted_seqs = []
    labels = []
    
    # Add Group A
    count_a = 0
    for gid in group_a_ids:
        if gid in id_to_seq:
            sorted_seqs.append(list(id_to_seq[gid]))
            labels.append(f"{gid} (A)")
            count_a += 1
            
    # Add Group B
    for gid in group_b_ids:
        if gid in id_to_seq:
            sorted_seqs.append(list(id_to_seq[gid]))
            labels.append(f"{gid} (B)")

    if not sorted_seqs:
        print("Error: No matching IDs found between Tree and FASTA.")
        return

    # 4. Convert to Numeric Matrix for Matplotlib
    # We need a fixed length. Assuming aligned FASTA, all lengths should be equal.
    # If not, we truncate/pad to the length of the first sequence for visualization safety.
    seq_len = len(sorted_seqs[0])
    matrix = []
    
    # Simple color mapping for nucleotides/amino acids
    # We hash the character to an integer for coloring
    unique_chars = set(char for seq in sorted_seqs for char in seq)
    char_to_int = {c: i for i, c in enumerate(sorted(unique_chars))}
    
    for seq in sorted_seqs:
        # Ensure length consistency
        trimmed_seq = seq[:seq_len] 
        row = [char_to_int.get(c, 0) for c in trimmed_seq]
        matrix.append(row)
    
    data = np.array(matrix)

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(12, max(4, len(sorted_seqs) * 0.3)))
    
    # Create a custom discrete colormap
    # Using a "tab20" or similar large palette to ensure different bases/AAs get diff colors
    cmap = plt.get_cmap('tab20b', len(unique_chars))
    
    im = ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # 6. Draw the Split Line
    split_y_position = count_a - 0.5
    
    ax.axhline(y=split_y_position, color='red', linewidth=3, linestyle='--', label="Phylogenetic Split")

    # 7. Formatting
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Alignment Position")
    ax.set_title(f"MSA Split Visualization\nSupport: {split_info['support']} | Branch Length: {split_info['length']:.4f}")
    
    # Optional: Add grid for readablity
    ax.grid(False)
    
    # Move legend outside
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05))
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Visualization saved to: {output_plot}")
    # plt.show() # Uncomment if running locally with UI


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
