from ete3 import Tree
import os
import torch
import json
import pandas as pd
import numpy as np
import shutil
from sklearn.decomposition import PCA
from estimate_matrix_normal.estimate_matrix_normal import matrix_normal_mle_fixed_u
from utils.align_embeddings_with_covariance import align_embeddings_with_covariance
from evaluate_split_options.utils import load_matrix_tensor, get_log_det, calculate_matrix_normal_ll, calculate_bic_matrix_normal


def calculate_jaccard(set1, set2):
    """Calculates Jaccard Index: Intersection / Union"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0: return 0.0
    return intersection / union


def is_split_redundant(current_split_sets, history_splits, threshold=0.85):
    """
    Checks if current split is similar to any split in history.
    """
    cur_a, cur_b = current_split_sets

    for idx, (hist_a, hist_b) in enumerate(history_splits):
        # Check Direct Match (A vs A)
        sim_direct = calculate_jaccard(cur_a, hist_a)
        
        # Check Inverse Match (A vs B) - because {A,B} is the same split as {B,A}
        sim_inverse = calculate_jaccard(cur_a, hist_b)
        
        if sim_direct >= threshold:
            return True, f"Too similar to previous Significant Split #{idx+1} (Direct match: {sim_direct:.2%})"
        
        if sim_inverse >= threshold:
            return True, f"Too similar to previous Significant Split #{idx+1} (Inverse match: {sim_inverse:.2%})"
            
    return False, ""


def find_candidate_splits(newick_path, k=1, min_support=0.9, min_prop=0.1):
    """
    Finds the top k candidate splits from a phylogenetic tree.
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


def pca_transform_data(full_tensor_standardized, sub_tensors_standardized, min_variance=None, min_components=None):
    """
    Fits PCA. If both min_variance and min_components are provided,
    it selects the number of components that satisfies BOTH conditions.
    """
    # Convert to NumPy
    full_np = full_tensor_standardized.cpu().numpy()
    n_samples, n_features = full_np.shape
    
    final_n_components = None

    # --- LOGIC TO DETERMINE N_COMPONENTS ---
    if min_variance is not None and min_components is not None:
        print(f"   -> Calculating components to satisfy Variance >= {min_variance*100:.0f}% AND Count >= {min_components}...")
        
        # 1. Fit a temporary PCA on all available components to check variance profile
        max_possible = min(n_samples, n_features)
        pca_temp = PCA(n_components=max_possible)
        pca_temp.fit(full_np)
        
        # 2. Calculate cumulative variance
        cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        
        # 3. Find index where variance is met
        n_needed_for_var = np.searchsorted(cumsum_var, min_variance) + 1
        
        # 4. Take the maximum of the two requirements
        final_n_components = max(n_needed_for_var, min_components)
        
        # 5. Safety cap: cannot exceed available features
        final_n_components = min(final_n_components, max_possible)
        
        print(f"      - Components for {min_variance*100:.0f}% variance: {n_needed_for_var}")
        print(f"      - Hard floor: {min_components}")
        print(f"      - Selected: {final_n_components}")

    elif min_variance is not None:
        final_n_components = min_variance 
        print(f"   -> Target Variance: {min_variance*100:.0f}%")

    elif min_components is not None:
        final_n_components = min_components
        print(f"   -> Target Count: {min_components}")
        
    else:
        final_n_components = 0.99 
    
    # --- FINAL FIT & TRANSFORM ---
    pca = PCA(n_components=final_n_components)
    pca.fit(full_np)
    
    p_new = pca.n_components_
    
    # 1. Transform full data
    transformed_full_np = pca.transform(full_np)
    transformed_tensors = [torch.from_numpy(transformed_full_np).float()]

    # 2. Transform sub-tensors
    for sub_tensor in sub_tensors_standardized:
        sub_np = sub_tensor.cpu().numpy()
        transformed_sub_np = pca.transform(sub_np)
        transformed_tensors.append(torch.from_numpy(transformed_sub_np).float())
        
    return transformed_tensors, p_new


def evaluate_top_splits(tree_path, cov_path, pt_path, output_path, k=5, 
                        pca_min_variance=None, pca_min_components=None, 
                        standardize=True, similarity_threshold=0.85):
    """
    Evaluates splits using Matrix Normal MLE estimation.
    Includes logic to skip splits that are highly similar to previously identified significant splits.
    """
    
    # --- 0. Setup Output Directories ---
    base_eval_dir = output_path
    sig_splits_dir = os.path.join(base_eval_dir, "significant_splits")
    non_sig_splits_dir = os.path.join(base_eval_dir, "non_significant_splits")
    
    os.makedirs(base_eval_dir, exist_ok=True)
    os.makedirs(sig_splits_dir, exist_ok=True)
    os.makedirs(non_sig_splits_dir, exist_ok=True)

    # --- Initial Load ---
    data_full = torch.load(pt_path, map_location='cpu')
    emb_raw_full = data_full['embeddings'].float()
    all_names = data_full.get('file_names') or data_full.get('names') or data_full.get('ids')
    
    N_total, p_initial = emb_raw_full.shape
    p_current = p_initial
    print(f"Initial Data Dimensions: N={N_total}, p={p_initial}")

    dir_out = os.path.dirname(pt_path)

    # --- PHASE 1: Alignment & Standardization ---
    print("\n" + "="*40)
    print("PHASE 1: Alignment & Standardization")
    print("="*40)
    
    aligned_full_path = os.path.join(dir_out, "aligned_global_embeddings.pt")
    emb_tensor_full = align_embeddings_with_covariance(cov_path, pt_path, aligned_full_path).float()
    
    if standardize:
        print("Applying Global Standardization (Z-score)...")
        emb_standardized_full_raw, = global_standardize_embeddings(emb_tensor_full, [emb_tensor_full])
    else:
        print("Skipping Standardization...")
        emb_standardized_full_raw = emb_tensor_full
    
    # --- PHASE 2: Dimensionality Reduction (PCA) ---
    if pca_min_variance is not None or pca_min_components is not None:
        print("\n" + "="*40)
        print("PHASE 2: Dimensionality Reduction (PCA)")
        print("="*40)
        
        [emb_transformed_full_raw], p_current = pca_transform_data(
            emb_standardized_full_raw, [], 
            min_variance=pca_min_variance, 
            min_components=pca_min_components
        )
        print(f"Final Dimension (p'): {p_current}")
    else:
        emb_transformed_full_raw = emb_standardized_full_raw

    # --- PHASE 3: Global Baseline (H0) ---
    print("\n" + "="*40)
    print("PHASE 3: Global Baseline (H0)")
    print("="*40)

    name_comment = '_global_H0'
    if pca_min_variance or pca_min_components:
        name_comment += '_PCA'

    _, v_path_full, _ = matrix_normal_mle_fixed_u(
        X=[emb_transformed_full_raw], 
        U_path=cov_path, 
        name_comments=name_comment
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
    
    pca_for_splits = None
    if pca_min_variance is not None or pca_min_components is not None:
        pca_for_splits = PCA(n_components=p_current)
        pca_for_splits.fit(emb_standardized_full_raw.cpu().numpy())
    
    candidates = find_candidate_splits(tree_path, k=k, min_support=0.8, min_prop=0.1)
    results = []
    
    significant_split_history = [] 

    for i, split in enumerate(candidates):
        rank = i + 1
        node_name = split.get('node_name', f'Node_{i}')
        
        print(f"\n--- Candidate {rank}: {node_name} (Len: {split.get('length', 0):.4f}) ---")

        current_sets = (split['group_a'], split['group_b'])
        is_redundant, reason = is_split_redundant(current_sets, significant_split_history, threshold=similarity_threshold)
        
        if is_redundant:
            print(f"   [SKIP] {reason}")
            results.append({
                'rank': rank, 'node': node_name, 'bic': None, 'delta': 0, 
                'sig': False, 'folder': None, 'note': 'Skipped (Redundant)'
            })
            continue

        split_folder_name = f"rank{rank}"
        split_dir = os.path.join(base_eval_dir, split_folder_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # NEW: Create calculations folder
        calc_dir = os.path.join(split_dir, "calculations")
        os.makedirs(calc_dir, exist_ok=True)

        suffix_a = f"_rank{rank}_subA"
        suffix_b = f"_rank{rank}_subB"

        # CHANGED: output_dir is now calc_dir
        cov_a, cov_b = split_covariance_matrix(cov_path, split, suffix_a, suffix_b, output_dir=calc_dir)
        pt_a, pt_b = split_protein_embeddings(pt_path, split, suffix_a, suffix_b, output_dir=calc_dir)

        if cov_a is None: continue

        # CHANGED: Aligned files go to calc_dir
        aligned_path_a = os.path.join(calc_dir, f"aligned{suffix_a}.pt")
        aligned_path_b = os.path.join(calc_dir, f"aligned{suffix_b}.pt")
        
        emb_tensor_a = align_embeddings_with_covariance(cov_a, pt_a, aligned_path_a).float()
        emb_tensor_b = align_embeddings_with_covariance(cov_b, pt_b, aligned_path_b).float()

        if standardize:
            emb_standardized_a_raw, emb_standardized_b_raw = global_standardize_embeddings(
                emb_tensor_full, [emb_tensor_a, emb_tensor_b]
            )
        else:
            emb_standardized_a_raw, emb_standardized_b_raw = emb_tensor_a, emb_tensor_b

        # Apply PCA
        if pca_for_splits is not None:
            emb_transformed_a_raw = torch.from_numpy(pca_for_splits.transform(emb_standardized_a_raw.cpu().numpy())).float()
            emb_transformed_b_raw = torch.from_numpy(pca_for_splits.transform(emb_standardized_b_raw.cpu().numpy())).float()
        else:
            emb_transformed_a_raw, emb_transformed_b_raw = emb_standardized_a_raw, emb_standardized_b_raw

        # Run MLE
        print("   Running MLE for Sub-trees")
        # CHANGED: output_dir is now calc_dir
        _, v_path_a, _ = matrix_normal_mle_fixed_u(X=[emb_transformed_a_raw], U_path=cov_a, name_comments=suffix_a, output_dir=calc_dir)
        _, v_path_b, _ = matrix_normal_mle_fixed_u(X=[emb_transformed_b_raw], U_path=cov_b, name_comments=suffix_b, output_dir=calc_dir)

        # Calculate Stats
        u_tensor_a = load_matrix_tensor(cov_a)
        u_tensor_b = load_matrix_tensor(cov_b)
        v_tensor_a = load_matrix_tensor(v_path_a)
        v_tensor_b = load_matrix_tensor(v_path_b)
        
        # CHANGED: Save embedding covariances to calc_dir
        pd.DataFrame(v_tensor_a.cpu().numpy()).to_csv(os.path.join(calc_dir, f"embedding_cov{suffix_a}.csv"))
        pd.DataFrame(v_tensor_b.cpu().numpy()).to_csv(os.path.join(calc_dir, f"embedding_cov{suffix_b}.csv"))

        ll_split = calculate_matrix_normal_ll(u_tensor_a.shape[0], p_current, u_tensor_a, v_tensor_a) + \
                   calculate_matrix_normal_ll(u_tensor_b.shape[0], p_current, u_tensor_b, v_tensor_b)
        
        bic_split = calculate_bic_matrix_normal(ll_split, N_total, p_current, num_models=2)
        delta_bic = bic_global - bic_split
        is_sig = bic_split < bic_global
        
        print(f"   Delta BIC: {delta_bic:.2f} [{'SIGNIFICANT' if is_sig else 'NO'}]")

        if is_sig:
            print(f"   -> SIGNIFICANT! Moving folder.")
            new_split_dir = os.path.join(sig_splits_dir, split_folder_name)
            if os.path.exists(new_split_dir): shutil.rmtree(new_split_dir)
            shutil.move(split_dir, new_split_dir)
            split_dir = new_split_dir
            
            significant_split_history.append((split['group_a'], split['group_b']))
            
            # Save JSON
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
                
                json_filename = f"split_rank{rank}.json"
                # Keep JSON in the root of the split folder for easy access
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

    print("\n" + "="*40)
    print("Moving Non-Significant Splits")
    print("="*40)
    for res in results:
        # If not significant, not skipped (folder exists), and still in the base folder
        if not res['sig'] and res['folder'] is not None:
            # Check if it currently resides in the base eval dir (meaning it wasn't moved to significant)
            if os.path.dirname(res['folder']) == base_eval_dir:
                split_name = os.path.basename(res['folder'])
                dest_path = os.path.join(non_sig_splits_dir, split_name)
                
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                
                try:
                    shutil.move(res['folder'], dest_path)
                    res['folder'] = dest_path # Update path in results
                except Exception as e:
                    print(f"Error moving {split_name} to non_significant_splits: {e}")

    # Summary
    print("\n" + "="*40 + "\nFINAL SUMMARY\n" + "="*40)
    for res in results:
        note = res.get('note', '')
        if note:
            print(f"{res['rank']:<5} | {res['node']:<15} | {note}")
        else:
            print(f"{res['rank']:<5} | {res['node']:<15} | {res['delta']:<15.2f} | {'YES' if res['sig'] else 'NO'}")

    return results
