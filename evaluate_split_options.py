from ete3 import Tree
import os
import torch


def find_candidate_splits(newick_path, k=1, min_support=0.9, min_prop=0.1):
    """
    Finds the top k candidate splits from a phylogenetic tree.

    A "candidate split" is defined as an internal branch with high statistical
    support. The function sorts candidates by branch length, so the top
    candidates represent the deepest, most significant divergences.

    Args:
        newick_path (str): Path to the Newick tree file (e.g., your .tree from FastTree).
        k (int): The number of top splits to return.
        min_support (float): Statistical support threshold (0-1). Branches below
                               this value will be filtered out. FastTree produces
                               SH-like support values (0-1).
        min_size (int): Minimum clade size. Splits that result in a group
                          smaller than this will be filtered out.

    Returns:
        list: A list of dictionaries. Each dictionary represents one "split" and contains:
              {'support': float, 'length': float, 'group_a': set, 'group_b': set}
    """

    # 1. Load the tree and set the root
    try:
        # format=1 helps ete3 correctly parse FastTree support values
        tree = Tree(newick_path, format=1)
    except Exception as e:
        print(f"Error loading tree with format=1: {e}")
        print("Falling back to format=0...")
        tree = Tree(newick_path, format=0) # Fallback

    # 2. Set the root objectively using midpoint rooting (best when no outgroup is known)
    try:
        tree.set_outgroup(tree.get_midpoint_outgroup())
    except Exception as e:
        print(f"Could not midpoint root the tree: {e}")
        print("Proceeding with the tree as-is, but the first split might be arbitrary.")

    all_leaves = set(tree.get_leaf_names())
    total_leaves = len(all_leaves)
    candidate_splits = []

    # 3. Iterate over all internal nodes
    for node in tree.traverse("postorder"):
        if node.is_leaf() or node.is_root():
            continue

        # 4. Filter candidates based on support and size
        support = node.support
        length = node.dist

        if support < min_support:
            continue

        clade_leaves = set(node.get_leaf_names())
        clade_size = len(clade_leaves)
        min_size = min_prop * total_leaves
        # Filter out trivial splits (e.g., a single leaf vs. everyone else)
        if clade_size < min_size or (total_leaves - clade_size) < min_size:
            continue

        # 5. Store the valid candidate split
        # group_a is the clade defined by the current node
        # group_b is everyone else
        group_a_leaves = clade_leaves
        group_b_leaves = all_leaves - group_a_leaves

        split_info = {
            'support': support,
            'length': length,
            'group_a': group_a_leaves,
            'group_b': group_b_leaves,
            'node_name': node.name # For identification
        }
        candidate_splits.append(split_info)

    # 6. Sort candidates: highest branch length first
    candidate_splits.sort(key=lambda x: x['length'], reverse=True)

    # 7. Return the top k candidates
    return candidate_splits[:k]


def split_and_save_tree(original_newick_path, split_info, output_suffix_a="_group_a", output_suffix_b="_group_b"):
    """
    Splits a tree into two subtrees based on a split_info dictionary and saves them.

    Uses the ete3.Tree.prune() method, which keeps only the specified leaves
    and the minimal internal nodes needed to connect them.

    Args:
        original_newick_path (str): Path to the original Newick file.
        split_info (dict): A dictionary from find_candidate_splits, containing
                           {'group_a': set_of_leaves, 'group_b': set_of_leaves}.
        output_suffix_a (str): Suffix to add to the original filename for group A's tree.
        output_suffix_b (str): Suffix to add to the original filename for group B's tree.

    Returns:
        tuple: (path_to_tree_a, path_to_tree_b)
    """
    print(f"\nPerforming split based on {len(split_info['group_a'])} vs {len(split_info['group_b'])} leaves.")

    # 1. Load the original tree (we need a fresh copy for pruning)
    # Use the same format-loading logic as the finder function
    try:
        original_tree = Tree(original_newick_path, format=1)
    except Exception:
        original_tree = Tree(original_newick_path, format=0)

    # We will save in format=1 to preserve support values
    out_format = 1

    # --- Create and save Tree A ---
    tree_a = original_tree.copy()
    # prune() keeps *only* the specified leaves and their common ancestors
    tree_a.prune(split_info['group_a'])

    # --- Create and save Tree B ---
    tree_b = original_tree.copy()
    tree_b.prune(split_info['group_b'])

    # 2. Determine output file paths
    dirname = os.path.dirname(original_newick_path)
    if dirname == "": # Handle case where file is in the current directory
        dirname = "."

    basename = os.path.basename(original_newick_path)
    name, ext = os.path.splitext(basename)

    # Define paths in the same directory as the original
    path_a = os.path.join(dirname, f"{name}{output_suffix_a}{ext}")
    path_b = os.path.join(dirname, f"{name}{output_suffix_b}{ext}")

    # 3. Save the new trees
    tree_a.write(outfile=path_a, format=out_format)
    tree_b.write(outfile=path_b, format=out_format)

    print(f"Saved Group A tree ({len(split_info['group_a'])} leaves) to: {path_a}")
    print(f"Saved Group B tree ({len(split_info['group_b'])} leaves) to: {path_b}")

    return (path_a, path_b)


def split_covariance_matrix(original_cov_path, split_info, output_suffix_a="_group_a", output_suffix_b="_group_b"):
    """
    Splits a covariance matrix and shifts values to the new root,
    BUT preserves original scale (no normalization).
    Includes ID alignment (slash vs underscore) fix.
    """
    print(f"\nProcessing Covariance Matrix (Shift Only, No Rescale): {original_cov_path}")

    # 1. Load
    df = pd.read_csv(original_cov_path, index_col=0)
    df.index = df.index.astype(str)

    # 2. Helper for ID alignment
    def get_valid_ids(tree_leaves, csv_index):
        valid_ids = []
        csv_lookup = set(csv_index)
        for leaf in tree_leaves:
            leaf = str(leaf).strip()
            if leaf in csv_lookup:
                valid_ids.append(leaf)
            else:
                alt_leaf = leaf.replace('/', '_')
                if alt_leaf in csv_lookup:
                    valid_ids.append(alt_leaf)
        return valid_ids

    # 3. Get aligned IDs
    ids_a = get_valid_ids(split_info['group_a'], df.index)
    ids_b = get_valid_ids(split_info['group_b'], df.index)

    if len(ids_a) == 0 or len(ids_b) == 0:
        raise ValueError("Critical Error: 0 matches found after ID alignment.")

    # 4. Function to Process Sub-Matrix (Shift Only)
    def process_submatrix(full_df, subset_ids, group_name):
        # A. Slice
        sub_df = full_df.loc[subset_ids, subset_ids].copy()

        # B. Find the Shift Value
        # The minimum value in the block corresponds to the shared path
        # from the Old Root to the split node.
        shift_val = sub_df.min().min()

        print(f"   [{group_name}] Subtracting background distance: {shift_val:.6f}")

        # C. Apply Shift
        sub_df = sub_df - shift_val

        # D. Clean up floating point noise
        # Sometimes subtraction leaves -0.00000001; we set those to 0.0
        sub_df[sub_df < 0] = 0.0

        return sub_df

    # 5. Process both matrices
    cov_a = process_submatrix(df, ids_a, "Group A")
    cov_b = process_submatrix(df, ids_b, "Group B")

    # 6. Save
    dirname = os.path.dirname(original_cov_path)
    basename = os.path.basename(original_cov_path)
    name, ext = os.path.splitext(basename)

    path_a = os.path.join(dirname, f"{name}{output_suffix_a}{ext}")
    path_b = os.path.join(dirname, f"{name}{output_suffix_b}{ext}")

    cov_a.to_csv(path_a)
    cov_b.to_csv(path_b)

    print(f"Saved Matrix A to: {path_a}")
    print(f"Saved Matrix B to: {path_b}")

    return path_a, path_b


def split_protein_embeddings(original_pt_path, split_info, output_suffix_a="_group_a", output_suffix_b="_group_b"):
    """
    Splits a PyTorch embedding file into two subsets based on tree split info.
    Handles the specific structure: {'embeddings': tensor, 'file_names': list}.
    """
    print(f"\nProcessing Embeddings: {original_pt_path}")

    # 1. Load the original .pt file
    # map_location='cpu' ensures it loads even if you don't have a GPU active right now
    data = torch.load(original_pt_path, map_location='cpu')

    full_tensor = data['embeddings']
    full_names = data['file_names'] # These already have '/' replaced by '_'

    print(f"Original Tensor Shape: {full_tensor.shape}")

    # 2. Create a lookup map for speed: Name -> Index
    # This makes finding indices O(1) instead of O(N)
    name_to_idx = {name: i for i, name in enumerate(full_names)}

    # 3. Helper to find indices for a specific group
    def get_indices_and_names(leaves_set, group_label):
        indices = []
        found_names = []

        missing_count = 0

        for leaf in leaves_set:
            # Normalize the tree leaf name to match the embedding file format
            # The user stated the PT file has '_' instead of '/'
            clean_name = str(leaf).strip().replace('/', '_')

            if clean_name in name_to_idx:
                idx = name_to_idx[clean_name]
                indices.append(idx)
                found_names.append(clean_name)
            else:
                missing_count += 1

        print(f"   [{group_label}] Found {len(indices)} embeddings. (Missing/Mismatch: {missing_count})")
        return torch.tensor(indices, dtype=torch.long), found_names

    # 4. Get indices for both groups
    idx_a, names_a = get_indices_and_names(split_info['group_a'], "Group A")
    idx_b, names_b = get_indices_and_names(split_info['group_b'], "Group B")

    # 5. Slice the Tensor
    # tensor[indices] selects specific rows efficiently
    emb_a = full_tensor[idx_a]
    emb_b = full_tensor[idx_b]

    # 6. Prepare Output Dictionaries
    out_data_a = {'embeddings': emb_a, 'file_names': names_a}
    out_data_b = {'embeddings': emb_b, 'file_names': names_b}

    # 7. Generate Paths
    dirname = os.path.dirname(original_pt_path)
    basename = os.path.basename(original_pt_path)
    name, ext = os.path.splitext(basename)

    path_a = os.path.join(dirname, f"{name}{output_suffix_a}{ext}")
    path_b = os.path.join(dirname, f"{name}{output_suffix_b}{ext}")

    # 8. Save
    torch.save(out_data_a, path_a)
    torch.save(out_data_b, path_b)

    print(f"Saved Embeddings A to: {path_a} {emb_a.shape}")
    print(f"Saved Embeddings B to: {path_b} {emb_b.shape}")

    return path_a, path_b
