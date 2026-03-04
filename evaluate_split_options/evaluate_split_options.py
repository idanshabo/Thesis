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
from evaluate_split_options.lrt_statistics import compute_gls_operators, compute_mle_and_lrt, simulate_null_data, add_jitter

def calculate_jaccard(set1, set2):
    """Calculates Jaccard Index: Intersection / Union"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0: return 0.0
    return intersection / union


def is_split_redundant(current_split_sets, history_splits, threshold=0.85):
    """
    Checks if current split is similar to any previously accepted split.
    """
    cur_a, cur_b = current_split_sets

    for idx, (hist_a, hist_b) in enumerate(history_splits):
        # Check Direct Match (A vs A)
        sim_direct = calculate_jaccard(cur_a, hist_a)
        
        # Check Inverse Match (A vs B)
        sim_inverse = calculate_jaccard(cur_a, hist_b)
        
        if sim_direct >= threshold:
            return True, f"Too similar to accepted Split #{idx+1} (Direct match: {sim_direct:.2%})"
        
        if sim_inverse >= threshold:
            return True, f"Too similar to accepted Split #{idx+1} (Inverse match: {sim_inverse:.2%})"
            
    return False, ""


def find_candidate_splits(newick_path, k=None, min_support=0.8, min_prop=0.1):
    """
    Finds candidate splits from a phylogenetic tree, ordered bottom-up (leaves to root).
    Filters out splits where the smallest group is less than `min_prop` (e.g., 10%) of total leaves.
    """
    # 1. Load the tree and set the root
    try:
        tree = Tree(newick_path, format=1)
    except Exception as e:
        print(f"Error loading tree with format=1: {e}")
        print("Falling back to format=0...")
        tree = Tree(newick_path, format=0)

    # 2. Set the root objectively using midpoint rooting
    try:
        tree.set_outgroup(tree.get_midpoint_outgroup())
    except Exception as e:
        print(f"Could not midpoint root the tree: {e}")

    all_leaves = set(tree.get_leaf_names())
    total_leaves = len(all_leaves)
    candidate_splits = []

    # 3. Iterate over all internal nodes
    # "postorder" inherently visits child nodes (leaves) before parent nodes (root)
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
        
        # Enforce the 10% rule: filter out splits where the smaller part is < 10%
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
            'node_name': node.name 
        }
        candidate_splits.append(split_info)

    # Note: Sorting by length was removed here to preserve the bottom-up order!

    # 6. Return candidates (cap at k if specified, otherwise return all)
    if k is not None:
        return candidate_splits[:k]
    return candidate_splits

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
    

class PhylogeneticPCA:
    def __init__(self, min_variance=None, min_components=None, mode='cov'):
        """
        mode: 'cov' for standard pPCA, 'corr' to standardize to unit evolutionary variance.
        """
        self.min_variance = min_variance
        self.min_components = min_components
        self.mode = mode
        self.a = None          # Phylogenetic mean
        self.V = None          # Eigenvectors
        self.std_diag = None   # For correlation mode scaling
        self.final_n_components = None

    def fit(self, X, C):
        """
        X: numpy array (n_samples, n_features)
        C: numpy array (n_samples, n_samples) - Phylogenetic covariance matrix
        """
        n, m = X.shape
        
        # 1. Compute inverse of C (using pseudo-inverse for numerical stability)
        invC = np.linalg.pinv(C)
        
        # 2. Compute vector of ancestral states (phylogenetic mean 'a')
        one = np.ones((n, 1))
        term1 = 1.0 / (one.T @ invC @ one)[0, 0]
        term2 = one.T @ invC @ X
        self.a = (term1 * term2).T  # Shape: (m, 1)
        
        # Center X using the phylogenetic mean
        X_centered = X - (one @ self.a.T)
        
        # 3. Compute evolutionary VCV matrix 'R'
        R = (X_centered.T @ invC @ X_centered) / (n - 1)
        
        # Convert to correlation matrix if standardized variance is requested
        if self.mode == 'corr':
            self.std_diag = np.sqrt(np.diag(R))
            # Standardize X
            X_centered = X_centered / self.std_diag
            # Change R to correlation matrix
            R = R / np.outer(self.std_diag, self.std_diag)
            
        # 4. Eigendecomposition of R
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine n_components based on your existing logic
        total_var = np.sum(np.maximum(eigenvalues, 0))
        explained_variance_ratio = np.maximum(eigenvalues, 0) / total_var
        cumsum_var = np.cumsum(explained_variance_ratio)
        
        final_comp = m 
        if self.min_variance is not None and self.min_components is not None:
            n_needed = np.searchsorted(cumsum_var, self.min_variance) + 1
            final_comp = max(n_needed, self.min_components)
        elif self.min_variance is not None:
            final_comp = np.searchsorted(cumsum_var, self.min_variance) + 1
        elif self.min_components is not None:
            final_comp = self.min_components
            
        self.final_n_components = min(final_comp, min(n, m))
        print(f"   -> pPCA selected {self.final_n_components} components.")
        
        # Save truncated eigenvectors
        self.V = eigenvectors[:, :self.final_n_components]

    def transform(self, X):
        """ Projects new data into the pPCA space using global phylogenetic mean and eigenvectors. """
        k = X.shape[0]
        one = np.ones((k, 1))
        
        # Center using the previously fitted phylogenetic mean
        X_centered = X - (one @ self.a.T)
        
        if self.mode == 'corr':
            X_centered = X_centered / self.std_diag
            
        # Compute scores in the rotated space
        return X_centered @ self.V


def evaluate_top_splits(tree_path, cov_path, pt_path, output_path, k=None, 
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

    # --- PHASE 1: Alignment ---
    print("\n" + "="*40)
    print("PHASE 1: Alignment (Standardization is now handled by pPCA)")
    print("="*40)
    
    aligned_full_path = os.path.join(dir_out, "aligned_global_embeddings.pt")
    emb_tensor_full = align_embeddings_with_covariance(cov_path, pt_path, aligned_full_path).float()
    
    # Load the aligned covariance matrix 'C' early so pPCA can use it
    u_tensor_full = load_matrix_tensor(cov_path)
    
    # --- PHASE 2: Dimensionality Reduction (pPCA) ---
    pca_for_splits = None
    if pca_min_variance is not None or pca_min_components is not None:
        print("\n" + "="*40)
        print("PHASE 2: Dimensionality Reduction (Phylogenetic PCA)")
        print("="*40)
        
        # Determine mode based on your 'standardize' flag
        p_mode = 'corr' if standardize else 'cov'
        
        pca_for_splits = PhylogeneticPCA(
            min_variance=pca_min_variance, 
            min_components=pca_min_components, 
            mode=p_mode
        )
        
        # Fit the pPCA using both the embeddings and the phylogenetic covariance matrix
        pca_for_splits.fit(emb_tensor_full.cpu().numpy(), u_tensor_full.cpu().numpy())
        
        # Transform the full dataset
        transformed_full_np = pca_for_splits.transform(emb_tensor_full.cpu().numpy())
        emb_transformed_full_raw = torch.from_numpy(transformed_full_np).float()
        
        p_current = pca_for_splits.final_n_components
        print(f"Final Dimension (p'): {p_current}")
    else:
        emb_transformed_full_raw = emb_tensor_full

    # --- PHASE 3: LRT Split Testing and Parametric Bootstrap ---
    print("\n" + "="*40)
    print("PHASE 3: LRT Split Testing (H1) & Bootstrap")
    print("="*40)
    
    raw_candidates = find_candidate_splits(tree_path, k=k, min_support=0.8, min_prop=0.1)
    
    candidates = []
    accepted_split_sets = []
    
    print(f"Found {len(raw_candidates)} raw candidate splits meeting size and support criteria.")
    
    for split in raw_candidates:
        current_sets = (split['group_a'], split['group_b'])
        is_redundant, reason = is_split_redundant(current_sets, accepted_split_sets, threshold=similarity_threshold)
        if not is_redundant:
            candidates.append(split)
            accepted_split_sets.append(current_sets)
            
    print(f"--> Kept {len(candidates)} unique splits for LRT evaluation.\n")

    # 1. Setup Global Null Model Parameters for Bootstrap
    # Get global U and global X
    U_global = load_matrix_tensor(cov_path).float()
    X_global = emb_transformed_full_raw.float()
    n_global, p_global = X_global.shape
    
    U_inv_g, P_g, t1_g, t2_g = compute_gls_operators(U_global)
    mu_hat_global = (t1_g @ t2_g @ X_global).squeeze() # shape (p,)
    
    # Global V_hat under H0
    S_global_H0 = X_global.T @ P_g @ X_global
    V_hat_global = S_global_H0 / n_global

    # Save the global V_hat matrix for downstream visualization
    family_name = os.path.basename(tree_path).split('.')[0]
    calc_dir = os.path.dirname(tree_path)
    global_cov_filename = f"{family_name}_calculations_global_H0_PCA_embeddings_cov_mat.csv"
    pd.DataFrame(V_hat_global.cpu().numpy()).to_csv(os.path.join(calc_dir, global_cov_filename))
    
    # Cholesky factors for bootstrap simulation
    # forcing symmetry
    U_global_sym = (U_global + U_global.T) / 2.0
    V_hat_global_sym = (V_hat_global + V_hat_global.T) / 2.0
    
    L_U = torch.linalg.cholesky(add_jitter(U_global_sym))                      
    L_V = torch.linalg.cholesky(add_jitter(V_hat_global_sym))

    # 2. Precompute Operators and Calculate Observed Lambda for all candidates
    split_data = []
    df_global_index = pd.read_csv(cov_path, index_col=0).index.astype(str)

    def get_valid_indices(tree_leaves, csv_index):
        valid_idx = []
        csv_lookup = list(csv_index)
        for leaf in tree_leaves:
            leaf = str(leaf).strip()
            if leaf in csv_lookup:
                valid_idx.append(csv_lookup.index(leaf))
            else:
                alt_leaf = leaf.replace('/', '_')
                if alt_leaf in csv_lookup:
                    valid_idx.append(csv_lookup.index(alt_leaf))
        return valid_idx

    print("   Precomputing operators and calculating observed statistics...")
    for i, split in enumerate(candidates):
        idx_A = get_valid_indices(split['group_a'], df_global_index)
        idx_B = get_valid_indices(split['group_b'], df_global_index)
        
        if not idx_A or not idx_B:
            continue
            
        # Extract sub-matrices directly from global U
        U_A = U_global[idx_A][:, idx_A]
        U_B = U_global[idx_B][:, idx_B]
        
        X_A = X_global[idx_A]
        X_B = X_global[idx_B]
        
        # Precompute projection operators
        _, P_A, _, _ = compute_gls_operators(U_A)
        _, P_B, _, _ = compute_gls_operators(U_B)
        
        # Calculate observed Lambda AND capture the V matrices
        lambda_obs, V_A, V_B = compute_mle_and_lrt(
            X_A, X_B, P_A, P_B, len(idx_A), len(idx_B), return_matrices=True
        )
        
        split_data.append({
            'rank': i + 1,
            'node_name': split.get('node_name', f'Node_{i}'),
            'split_dict': split,
            'idx_A': idx_A,
            'idx_B': idx_B,
            'P_A': P_A,
            'P_B': P_B,
            'n_A': len(idx_A),
            'n_B': len(idx_B),
            'lambda_obs': lambda_obs.item(),
            'support': split.get('support', 0.0),
            'V_A': V_A.cpu().numpy(),  # Store as numpy for easy saving later
            'V_B': V_B.cpu().numpy()
        })

    # 3. Parametric Bootstrap (Westfall-Young)
    C_replicates = 10000  # Number of bootstrap replicates.
    print(f"\n   Running Parametric Bootstrap with {C_replicates} replicates...")
    max_lambdas_null = []

    for c in range(C_replicates):
        if c % 50 == 0:
            print(f"      Bootstrap iteration {c}/{C_replicates}")
            
        # Simulate data under H0
        X_sim = simulate_null_data(n_global, p_global, mu_hat_global, L_U, L_V)
        
        lambda_sims = []
        for sd in split_data:
            X_A_sim = X_sim[sd['idx_A']]
            X_B_sim = X_sim[sd['idx_B']]
            
            lam_sim = compute_mle_and_lrt(X_A_sim, X_B_sim, sd['P_A'], sd['P_B'], sd['n_A'], sd['n_B'])
            lambda_sims.append(lam_sim.item())
            
        # Record the maximum Lambda across all splits for this replicate
        max_lambdas_null.append(max(lambda_sims))

    max_lambdas_null = np.array(max_lambdas_null)

    # 4. Calculate p-values and output
    alpha_level = 0.05
    results = []

    for sd in split_data:
        # Adjusted p-value: proportion of null max-lambdas >= observed lambda
        # Adding 1 to numerator and denominator prevents p=0
        count_exceed = np.sum(max_lambdas_null >= sd['lambda_obs'])
        p_adj = (count_exceed + 1) / (C_replicates + 1)
        
        is_sig = p_adj <= alpha_level
        
        print(f"   Split {sd['rank']} ({sd['node_name']}): Lambda_obs={sd['lambda_obs']:.2f}, p_adj={p_adj:.4f} [{'SIGNIFICANT' if is_sig else 'NO'}]")
        
        # Determine folder structure based on significance
        split_folder_name = f"rank{sd['rank']}"
        dest_dir = sig_splits_dir if is_sig else non_sig_splits_dir
        split_dir = os.path.join(dest_dir, split_folder_name)
        
        # Clean up directory if it exists from a previous run, then create
        if os.path.exists(split_dir): shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        
        if is_sig:
            print(f"   -> SIGNIFICANT! Saving matrices and JSON.")
            
            # --- 1. Reconstruct and Save Matrices for Phase 5 ---
            calc_dir = os.path.join(split_dir, "calculations")
            os.makedirs(calc_dir, exist_ok=True)
            
            # Save V_A and V_B directly from memory
            pd.DataFrame(sd['V_A']).to_csv(os.path.join(calc_dir, f"embedding_cov_rank{sd['rank']}_subA.csv"))
            pd.DataFrame(sd['V_B']).to_csv(os.path.join(calc_dir, f"embedding_cov_rank{sd['rank']}_subB.csv"))
            
            # Extract and Save U_A and U_B with original string IDs
            names_A = df_global_index[sd['idx_A']]
            names_B = df_global_index[sd['idx_B']]
            
            U_A_np = U_global[sd['idx_A']][:, sd['idx_A']].cpu().numpy()
            U_B_np = U_global[sd['idx_B']][:, sd['idx_B']].cpu().numpy()
            
            basename = os.path.splitext(os.path.basename(cov_path))[0]
            pd.DataFrame(U_A_np, index=names_A, columns=names_A).to_csv(
                os.path.join(calc_dir, f"{basename}_rank{sd['rank']}_subA.csv")
            )
            pd.DataFrame(U_B_np, index=names_B, columns=names_B).to_csv(
                os.path.join(calc_dir, f"{basename}_rank{sd['rank']}_subB.csv")
            )
            
            # --- 2. Save JSON ---
            split = sd['split_dict']
            raw_group_a = split.get('taxa') or split.get('leaves') or split.get('group_a')
            if raw_group_a and all_names and len(all_names) > 0:
                group_a_names = [name.replace("/", "_") for name in raw_group_a]
                set_a = set(group_a_names)
                group_b_names = [x for x in all_names if x not in set_a]
                
                split_data_out = {
                    "rank": sd['rank'],
                    "node_name": sd['node_name'],
                    "support": split.get('support', 0.0),
                    "lambda_obs": sd['lambda_obs'],
                    "p_adj": p_adj,
                    "group_a": group_a_names,
                    "group_b": group_b_names,
                    "folder_path": split_dir
                }
                
                json_filename = f"split_rank{sd['rank']}.json"
                json_path = os.path.join(split_dir, json_filename)
                
                with open(json_path, 'w') as f:
                    json.dump(split_data_out, f, indent=4)
                print(f"   -> JSON saved to {json_path}")
            else:
                print("   [!] Warning: Could not extract leaf names to save JSON.")

        results.append({
            'rank': sd['rank'],
            'node': sd['node_name'],
            'lambda': sd['lambda_obs'],
            'p_adj': p_adj,
            'sig': is_sig,
            'folder': split_dir 
        })

    # Summary
    print("\n" + "="*40 + "\nFINAL SUMMARY\n" + "="*40)
    for res in results:
        print(f"{res['rank']:<5} | {res['node']:<15} | L: {res['lambda']:<10.2f} | p: {res['p_adj']:<6.4f} | {'YES' if res['sig'] else 'NO'}")

    # Define the counts based on your lists
    raw_splits_count = len(raw_candidates)
    unique_splits_count = len(candidates)

    return results, raw_splits_count, unique_splits_count, p_current
