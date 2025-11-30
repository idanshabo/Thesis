from ete3 import Tree
import os
import torch
import json
import pandas as pd
import shutil
from sklearn.decomposition import PCA
from estimate_matrix_normal.estimate_matrix_normal import matrix_normal_mle_fixed_u
from align_embeddings_with_covariance import align_embeddings_with_covariance
from evaluate_split_options.utils import load_matrix_tensor, get_log_det, calculate_matrix_normal_ll, calculate_bic_matrix_normal


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


def split_covariance_matrix(original_cov_path, split_info, output_suffix_a="_group_a", output_suffix_b="_group_b", output_dir=None):
    """
    Splits a covariance matrix and shifts values to the new root.
    Saves output to output_dir if provided, otherwise uses original directory.
    """
    #print(f"\nProcessing Covariance Matrix: {original_cov_path}")

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
        print("Error: 0 matches found after ID alignment.")
        return None, None

    # 4. Function to Process Sub-Matrix
    def process_submatrix(full_df, subset_ids):
        sub_df = full_df.loc[subset_ids, subset_ids].copy()
        shift_val = sub_df.min().min()
        sub_df = sub_df - shift_val
        sub_df[sub_df < 0] = 0.0
        return sub_df

    # 5. Process both matrices
    cov_a = process_submatrix(df, ids_a)
    cov_b = process_submatrix(df, ids_b)

    # 6. Determine Output Paths
    basename = os.path.basename(original_cov_path)
    name, ext = os.path.splitext(basename)

    # Logic: Use output_dir if provided, else use original directory
    if output_dir:
        save_dir = output_dir
    else:
        save_dir = os.path.dirname(original_cov_path)

    path_a = os.path.join(save_dir, f"{name}{output_suffix_a}{ext}")
    path_b = os.path.join(save_dir, f"{name}{output_suffix_b}{ext}")

    cov_a.to_csv(path_a)
    cov_b.to_csv(path_b)

    return path_a, path_b


def split_protein_embeddings(original_pt_path, split_info, output_suffix_a="_group_a", output_suffix_b="_group_b", output_dir=None):
    """
    Splits a PyTorch embedding file into two subsets based on tree split info.
    Saves output to output_dir if provided.
    """
    # 1. Load the original .pt file
    data = torch.load(original_pt_path, map_location='cpu')

    full_tensor = data['embeddings']
    full_names = data['file_names'] 

    # 2. Create lookup
    name_to_idx = {name: i for i, name in enumerate(full_names)}

    # 3. Helper to find indices
    def get_indices_and_names(leaves_set):
        indices = []
        found_names = []
        for leaf in leaves_set:
            clean_name = str(leaf).strip().replace('/', '_')
            if clean_name in name_to_idx:
                indices.append(name_to_idx[clean_name])
                found_names.append(clean_name)
        return torch.tensor(indices, dtype=torch.long), found_names

    # 4. Get indices
    idx_a, names_a = get_indices_and_names(split_info['group_a'])
    idx_b, names_b = get_indices_and_names(split_info['group_b'])

    # 5. Slice Tensor
    emb_a = full_tensor[idx_a]
    emb_b = full_tensor[idx_b]

    # 6. Prepare Output
    out_data_a = {'embeddings': emb_a, 'file_names': names_a}
    out_data_b = {'embeddings': emb_b, 'file_names': names_b}

    # 7. Determine Output Paths
    basename = os.path.basename(original_pt_path)
    name, ext = os.path.splitext(basename)

    # Logic: Use output_dir if provided, else use original directory
    if output_dir:
        save_dir = output_dir
    else:
        save_dir = os.path.dirname(original_pt_path)

    path_a = os.path.join(save_dir, f"{name}{output_suffix_a}{ext}")
    path_b = os.path.join(save_dir, f"{name}{output_suffix_b}{ext}")

    # 8. Save
    torch.save(out_data_a, path_a)
    torch.save(out_data_b, path_b)

    return path_a, path_b


def global_standardize_embeddings(full_embeddings, embeddings_list, epsilon=1e-8):
    """
    Calculates the global mean (mu) and standard deviation (std) from the full dataset
    and applies standardization (Z-score) to all matrices in the list.
    
    Args:
        full_embeddings (Tensor): The complete (N_total, p) embedding tensor.
        embeddings_list (list of Tensor): List of sub-tensors to be standardized.
        epsilon (float): Small value to prevent division by zero for dimensions with zero variance.
        
    Returns:
        list of Tensor: The globally standardized tensors.
    """
    print("Global Standardization applied to isolate covariance structure.")
    
    # 1. Calculate Global Statistics
    global_mu = torch.mean(full_embeddings, dim=0)
    # Use torch.std with unbiased=False (population standard deviation for a complete sample)
    global_std = torch.std(full_embeddings, dim=0, unbiased=False)
    
    # Ensure no division by zero
    global_std[global_std == 0] = epsilon
    
    # 2. Apply Standardization to all tensors
    standardized_list = []
    for emb in embeddings_list:
        # Standardization: (X - mu) / sigma
        standardized_list.append((emb - global_mu) / global_std)
        
    return standardized_list


def pca_transform_data(full_tensor_standardized, sub_tensors_standardized, variance_explained=0.9):
    """
    Fits PCA on the full standardized tensor and transforms all tensors to the new, smaller dimension.
    """
    print(f"   -> Applying PCA to capture {variance_explained * 100:.0f}% of variance.")
    
    # Convert to NumPy for scikit-learn PCA
    full_np = full_tensor_standardized.cpu().numpy()
    
    # 1. Fit PCA on the full data
    pca = PCA(n_components=variance_explained)
    pca.fit(full_np)
    
    p_new = pca.n_components_
    print(f"   -> Reduced dimensionality from {full_tensor_standardized.shape[1]} to {p_new}.")

    # 2. Transform the full data
    transformed_full_np = pca.transform(full_np)
    transformed_tensors = [torch.from_numpy(transformed_full_np).float()]

    # 3. Transform the sub-tensors
    for sub_tensor in sub_tensors_standardized:
        sub_np = sub_tensor.cpu().numpy()
        transformed_sub_np = pca.transform(sub_np)
        transformed_tensors.append(torch.from_numpy(transformed_sub_np).float())
        
    return transformed_tensors, p_new


def evaluate_top_splits(tree_path, cov_path, pt_path, output_path, k=5, target_pca_variance=None, standardize=True):
    """
    Evaluates splits using Matrix Normal MLE estimation.
    Fixes name mismatch (/) vs (_) for JSON saving.
    Saves Predicted Embedding Covariance (V) matrices to CSV.
    """
    
    # --- 0. Setup Output Directories ---
    base_eval_dir = os.path.join(output_path, "splits_evaluations")
    sig_splits_dir = os.path.join(base_eval_dir, "significant_splits")
    
    os.makedirs(base_eval_dir, exist_ok=True)
    os.makedirs(sig_splits_dir, exist_ok=True)

    # --- Initial Load and Dimensions ---
    data_full = torch.load(pt_path, map_location='cpu')
    emb_raw_full = data_full['embeddings'].float()
    
    # Ensure we get the full list of names
    all_names = data_full.get('file_names') or data_full.get('names') or data_full.get('ids')
    
    N_total, p_initial = emb_raw_full.shape
    p_current = p_initial
    print(f"Initial Data Dimensions: N={N_total}, p={p_initial}")

    dir_out = os.path.dirname(pt_path)

    # --- PHASE 1: Data Alignment and Standardization ---
    print("\n" + "="*40)
    print("PHASE 1: Alignment & Standardization")
    print("="*40)
    
    aligned_full_path = os.path.join(dir_out, "aligned_global_embeddings.pt")
    emb_tensor_full = align_embeddings_with_covariance(cov_path, pt_path, aligned_full_path).float()
    
    if standardize:
        print("Applying Global Standardization (Z-score)...")
        emb_standardized_full_raw, = global_standardize_embeddings(emb_tensor_full, [emb_tensor_full])
    else:
        print("Skipping Standardization (Using raw aligned embeddings)...")
        emb_standardized_full_raw = emb_tensor_full
    
    # --- PHASE 2: Dimensionality Reduction (PCA) ---
    if target_pca_variance is not None:
        print("\n" + "="*40)
        print("PHASE 2: Dimensionality Reduction (PCA)")
        print("="*40)
        
        [emb_transformed_full_raw], p_current = pca_transform_data(
            emb_standardized_full_raw, [], target_pca_variance
        )
        print(f"New Dimension (p'): {p_current}")
    else:
        emb_transformed_full_raw = emb_standardized_full_raw

    # --- PHASE 3: Global Baseline (H0) Estimation ---
    print("\n" + "="*40)
    print("PHASE 3: Global Baseline (H0)")
    print("="*40)

    print(f"Running Matrix Normal MLE for Global Model...")
    _, v_path_full, _ = matrix_normal_mle_fixed_u(
        X=[emb_transformed_full_raw], 
        U_path=cov_path, 
        name_comments='_global_H0_PCA' if target_pca_variance else '_global_H0'
    )
    
    u_tensor_full = load_matrix_tensor(cov_path)
    v_tensor_full = load_matrix_tensor(v_path_full) 
    
    ll_global = calculate_matrix_normal_ll(N_total, p_current, u_tensor_full, v_tensor_full)
    bic_global = calculate_bic_matrix_normal(ll_global, N_total, p_current, num_models=1)
    
    print(f"Global LL: {ll_global:.2f}")
    print(f"Global BIC: {bic_global:.2f}")

    # --- PHASE 4: Split Testing (H1) ---
    print("\n" + "="*40)
    print("PHASE 4: Split Testing (H1)")
    print("="*40)
    
    if target_pca_variance is not None:
        pca = PCA(n_components=target_pca_variance)
        pca.fit(emb_standardized_full_raw.cpu().numpy())
    
    candidates = find_candidate_splits(tree_path, k=k, min_support=0.8, min_prop=0.1)
    results = []

    for i, split in enumerate(candidates):
        rank = i + 1
        node_name = split.get('node_name', f'Node_{i}')
        
        # Sanitize node name for folder creation
        safe_node_name = node_name.replace("/", "_").replace(" ", "")
        
        print(f"\n--- Candidate {rank}: {node_name} (Len: {split.get('length', 0):.4f}) ---")

        # 1. Create Initial Sub-Directory
        split_folder_name = f"rank{rank}_{safe_node_name}"
        split_dir = os.path.join(base_eval_dir, split_folder_name)
        os.makedirs(split_dir, exist_ok=True)

        suffix_a = f"_rank{rank}_subA"
        suffix_b = f"_rank{rank}_subB"

        # A. Generate Split Files
        cov_a, cov_b = split_covariance_matrix(cov_path, split, suffix_a, suffix_b, output_dir=split_dir)
        pt_a, pt_b = split_protein_embeddings(pt_path, split, suffix_a, suffix_b, output_dir=split_dir)

        if cov_a is None:
            print("Skipping due to alignment error.")
            continue

        # B. Align Subsets
        aligned_path_a = os.path.join(split_dir, f"aligned{suffix_a}.pt")
        aligned_path_b = os.path.join(split_dir, f"aligned{suffix_b}.pt")
        
        emb_tensor_a = align_embeddings_with_covariance(cov_a, pt_a, aligned_path_a).float()
        emb_tensor_b = align_embeddings_with_covariance(cov_b, pt_b, aligned_path_b).float()

        # C. Standardization
        if standardize:
            emb_standardized_a_raw, emb_standardized_b_raw = global_standardize_embeddings(
                emb_tensor_full, [emb_tensor_a, emb_tensor_b]
            )
        else:
            emb_standardized_a_raw, emb_standardized_b_raw = emb_tensor_a, emb_tensor_b

        # D. PCA
        if target_pca_variance is not None:
            emb_transformed_a_raw = torch.from_numpy(pca.transform(emb_standardized_a_raw.cpu().numpy())).float()
            emb_transformed_b_raw = torch.from_numpy(pca.transform(emb_standardized_b_raw.cpu().numpy())).float()
        else:
            emb_transformed_a_raw, emb_transformed_b_raw = emb_standardized_a_raw, emb_standardized_b_raw

        # E. MLE
        print("   Running MLE for Sub-trees")
        _, v_path_a, _ = matrix_normal_mle_fixed_u([emb_transformed_a_raw], cov_a, suffix_a, output_dir=split_dir)
        _, v_path_b, _ = matrix_normal_mle_fixed_u([emb_transformed_b_raw], cov_b, suffix_b, output_dir=split_dir)

        # F. Calculate BIC and Save V Matrices
        u_tensor_a = load_matrix_tensor(cov_a)
        u_tensor_b = load_matrix_tensor(cov_b)
        v_tensor_a = load_matrix_tensor(v_path_a)
        v_tensor_b = load_matrix_tensor(v_path_b)
        
        # --- NEW CODE BLOCK: SAVE EMBEDDING COVARIANCE MATRICES ---
        v_out_a = os.path.join(split_dir, f"embedding_cov{suffix_a}.csv")
        v_out_b = os.path.join(split_dir, f"embedding_cov{suffix_b}.csv")
        
        # Convert tensors to numpy and save as CSV
        pd.DataFrame(v_tensor_a.cpu().numpy()).to_csv(v_out_a)
        pd.DataFrame(v_tensor_b.cpu().numpy()).to_csv(v_out_b)
        print(f"   -> Saved Embedding Covariance matrices to {split_dir}")
        # ----------------------------------------------------------

        n_a, n_b = u_tensor_a.shape[0], u_tensor_b.shape[0]

        ll_a = calculate_matrix_normal_ll(n_a, p_current, u_tensor_a, v_tensor_a)
        ll_b = calculate_matrix_normal_ll(n_b, p_current, u_tensor_b, v_tensor_b)
        
        ll_split = ll_a + ll_b
        bic_split = calculate_bic_matrix_normal(ll_split, N_total, p_current, num_models=2)
        
        delta_bic = bic_global - bic_split
        is_sig = bic_split < bic_global
        
        print(f"   Split LL:  {ll_split:.2f}")
        print(f"   Split BIC: {bic_split:.2f}")
        print(f"   Delta BIC: {delta_bic:.2f} [{'SIGNIFICANT' if is_sig else 'NO'}]")

        # --- G. Handle Significance (Move Folder & Save JSON) ---
        if is_sig:
            print(f"   -> SIGNIFICANT! Moving folder to: {sig_splits_dir}")
            
            new_split_dir = os.path.join(sig_splits_dir, split_folder_name)
            
            if os.path.exists(new_split_dir):
                shutil.rmtree(new_split_dir)
            shutil.move(split_dir, new_split_dir)
            
            # Update split_dir variable for JSON logic below
            split_dir = new_split_dir
            
            # Generate JSON
            raw_group_a = split.get('taxa') or split.get('leaves') or split.get('group_a')
            
            if raw_group_a and all_names and len(all_names) > 0:
                group_a_names = [name.replace("/", "_") for name in raw_group_a]
                set_a = set(group_a_names)
                group_b_names = [x for x in all_names if x not in set_a]
                
                split_data_out = {
                    "rank": rank,
                    "node_name": node_name,
                    "support": split.get('support', 0.0),
                    "delta_bic": delta_bic,
                    "group_a": group_a_names,
                    "group_b": group_b_names,
                    "folder_path": split_dir
                }
                
                json_filename = f"split_rank{rank}_{safe_node_name}.json"
                json_path = os.path.join(split_dir, json_filename)
                
                with open(json_path, 'w') as f:
                    json.dump(split_data_out, f, indent=4)
                print(f"   -> JSON saved to {json_path}")
            else:
                print("   [!] Warning: Could not extract leaf names to save JSON.")

        results.append({
            'rank': rank,
            'node': node_name,
            'bic': bic_split,
            'delta': delta_bic,
            'sig': is_sig,
            'folder': split_dir 
        })

    # --- 4. Summary ---
    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    print(f"{'Rank':<5} | {'Node':<15} | {'Delta BIC':<15} | {'Result':<10}")
    print("-" * 40)
    for res in results:
        print(f"{res['rank']:<5} | {res['node']:<15} | {res['delta']:<15.2f} | {'YES' if res['sig'] else 'NO'}")

    return results
