from ete3 import Tree
import os
import torch
import json
import pandas as pd
import numpy as np
import shutil
from Bio import SeqIO
from sklearn.decomposition import PCA
from estimate_matrix_normal.estimate_matrix_normal import matrix_normal_mle_fixed_u
from utils.align_embeddings_with_covariance import align_embeddings_with_covariance
from evaluate_split_options.utils import load_matrix_tensor, get_log_det, calculate_matrix_normal_ll, calculate_bic_matrix_normal
from evaluate_split_options.lrt_statistics import compute_gls_operators, compute_mle_and_lrt, simulate_null_data, add_jitter
from evaluate_split_options.recursive_tree_traversal import find_candidate_splits_from_node, recursive_mean_split

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


def evaluate_top_splits(tree_path, cov_path, pt_path, output_path, calc_dir, fasta_path, k=None, 
                        pca_min_variance=None, pca_min_components=None, 
                        standardize=True, similarity_threshold=0.85,
                        anova_alpha=0.05, anova_permutations=999):
    """
    Evaluates splits using a Two-Stage Procedure:
    1. Recursive Mean Shift Testing (Phylogenetic ANOVA) to find stable sub-families.
    2. Local pPCA + Covariance Shift Testing (LRT & Bootstrap) on each sub-family.
    """
    
    # --- PHASE 1: Initial Load & Alignment ---
    print("\n" + "="*40)
    print("PHASE 1: Global Alignment (No pPCA yet)")
    print("="*40)
    
    # Load Tree
    try:
        tree = Tree(tree_path, format=1)
    except Exception:
        tree = Tree(tree_path, format=0)
    try:
        tree.set_outgroup(tree.get_midpoint_outgroup())
    except: pass
    
    # Load and Align Embeddings to the Covariance Matrix
    data_full = torch.load(pt_path, map_location='cpu')
    all_names = data_full.get('file_names') or data_full.get('names') or data_full.get('ids')
    
    aligned_full_path = os.path.join(os.path.dirname(pt_path), "aligned_global_embeddings.pt")
    emb_tensor_full = align_embeddings_with_covariance(cov_path, pt_path, aligned_full_path).float()
    
    C_global = load_matrix_tensor(cov_path).float()
    df_global_index = list(pd.read_csv(cov_path, index_col=0).index.astype(str))
    
    print(f"Loaded {len(df_global_index)} aligned sequences with {emb_tensor_full.shape[1]} dimensions.")

    # --- PHASE 2: Recursive Mean Shift Testing ---
    print("\n" + "="*40)
    print("PHASE 2: Recursive Mean Shift Testing (Phylogenetic ANOVA)")
    print("="*40)
    
    stable_subfamilies = recursive_mean_split(
        tree_node=tree, 
        Y_global=emb_tensor_full, 
        C_global=C_global, 
        global_names=df_global_index, 
        min_prop=0.1, 
        alpha=anova_alpha, 
        n_permutations=anova_permutations
    )
    
    print(f"\n=> Divided family into {len(stable_subfamilies)} stable sub-families based on global mean shifts.")

    # Tracking variables to return to the pipeline
    all_results = []
    total_raw_splits = 0
    total_unique_splits = 0
    final_p_dims = {}

    # --- PHASE 3 & 4: Sub-Family Processing ---
    for sf_idx, subfamily in enumerate(stable_subfamilies, 1):
        sf_node = subfamily['node']
        sf_leaves = list(subfamily['leaves'])
        sf_indices = subfamily['indices']
        n_sf = len(sf_leaves)
        
        print("\n" + "="*40)
        print(f"PROCESSING SUB-FAMILY {sf_idx}/{len(stable_subfamilies)} ({n_sf} leaves)")
        print("="*40)
        
        # 1. Setup Directories
        # --> OUTPUT DIR (for results.json, plots, and significant splits)
        out_sf_dir = os.path.join(output_path, f"subfamily_{sf_idx}")
        sf_sig_dir = os.path.join(out_sf_dir, "significant_splits")
        sf_non_sig_dir = os.path.join(out_sf_dir, "non_significant_splits")
        os.makedirs(sf_sig_dir, exist_ok=True)
        os.makedirs(sf_non_sig_dir, exist_ok=True)
        
        # --> CALCULATION DIR (for .fasta, .tree, and .csv matrices)
        calc_sf_dir = os.path.join(calc_dir, f"subfamily_{sf_idx}")
        os.makedirs(calc_sf_dir, exist_ok=True)
        
        # 2. Extract and Save Local Assets
        # Save Tree
        sf_tree_path = os.path.join(calc_sf_dir, f"subfamily_{sf_idx}.tree")
        sf_node.write(outfile=sf_tree_path)
        print(f"   -> Saved physical tree to {sf_tree_path}")
        
        # Save Cropped FASTA
        global_records = list(SeqIO.parse(fasta_path, "fasta"))
        sf_records = [rec for rec in global_records if str(rec.id).replace('/', '_') in sf_leaves or str(rec.id) in sf_leaves]
        sf_fasta_path = os.path.join(calc_sf_dir, f"subfamily_{sf_idx}.fasta")
        SeqIO.write(sf_records, sf_fasta_path, "fasta")
        print(f"   -> Saved cropped FASTA to {sf_fasta_path}")

        # Extract Local Matrices (Embeddings and Covariance)
        idx_tensor = torch.tensor(sf_indices, dtype=torch.long, device=emb_tensor_full.device)
        Y_local = emb_tensor_full[idx_tensor]
        U_local = C_global[idx_tensor][:, idx_tensor]
        
        # Save Shifted Local Covariance Matrix
        U_local_shifted = U_local - torch.min(U_local)
        sf_cov_df = pd.DataFrame(U_local_shifted.cpu().numpy(), index=sf_leaves, columns=sf_leaves)
        sf_cov_path = os.path.join(calc_sf_dir, f"subfamily_{sf_idx}_cov_mat.csv")
        sf_cov_df.to_csv(sf_cov_path)
        print(f"   -> Saved local covariance matrix to {sf_cov_path}")
        
        if n_sf < 10:
            print(f"   -> Sub-family {sf_idx} too small for meaningful covariance testing. Skipping.")
            continue
            
        # --- PHASE 3: Local pPCA ---
        print("   -> Running Local pPCA...")
        p_current = Y_local.shape[1]
        X_sf = Y_local
        
        if pca_min_variance is not None or pca_min_components is not None:
            p_mode = 'corr' if standardize else 'cov'
            pca_sf = PhylogeneticPCA(min_variance=pca_min_variance, min_components=pca_min_components, mode=p_mode)
            pca_sf.fit(Y_local.cpu().numpy(), U_local.cpu().numpy())
            
            X_sf = torch.from_numpy(pca_sf.transform(Y_local.cpu().numpy())).float()
            p_current = pca_sf.final_n_components
            
        final_p_dims[f"subfamily_{sf_idx}"] = p_current
        print(f"   -> Local dimension reduced to {p_current}")

        # --- PHASE 4: Local Covariance Shift Test ---
        print("   -> Finding candidate covariance splits...")
        raw_candidates = find_candidate_splits_from_node(sf_node, k=k, min_support=0.8, min_prop=0.1)
        total_raw_splits += len(raw_candidates)
        
        candidates = []
        accepted_split_sets = []
        for split in raw_candidates:
            current_sets = (split['group_a'], split['group_b'])
            is_redundant, reason = is_split_redundant(current_sets, accepted_split_sets, threshold=similarity_threshold)
            if not is_redundant:
                candidates.append(split)
                accepted_split_sets.append(current_sets)
                
        total_unique_splits += len(candidates)
        
        if not candidates:
            print("   -> No valid candidate splits found in this sub-family.")
            continue
            
        # Global Null parameters FOR THIS SUB-FAMILY
        U_inv_g, P_g, t1_g, t2_g = compute_gls_operators(U_local)
        mu_hat_sf = (t1_g @ t2_g @ X_sf).squeeze()
        V_hat_sf = (X_sf.T @ P_g @ X_sf) / n_sf
        
        U_local_sym = (U_local + U_local.T) / 2.0
        V_hat_sf_sym = (V_hat_sf + V_hat_sf.T) / 2.0
        L_U = torch.linalg.cholesky(add_jitter(U_local_sym))                      
        L_V = torch.linalg.cholesky(add_jitter(V_hat_sf_sym))
        
        # Calculate Observed Lambda
        split_data = []
        for i, split in enumerate(candidates):
            # Map leaf names directly to local indices (0 to n_sf-1)
            local_idx_A = [sf_leaves.index(name) for name in split['group_a'] if name in sf_leaves]
            local_idx_B = [sf_leaves.index(name) for name in split['group_b'] if name in sf_leaves]
            
            U_A = U_local[local_idx_A][:, local_idx_A]
            U_B = U_local[local_idx_B][:, local_idx_B]
            X_A, X_B = X_sf[local_idx_A], X_sf[local_idx_B]
            
            _, P_A, _, _ = compute_gls_operators(U_A)
            _, P_B, _, _ = compute_gls_operators(U_B)
            
            lambda_obs, V_A, V_B = compute_mle_and_lrt(X_A, X_B, P_A, P_B, len(local_idx_A), len(local_idx_B), return_matrices=True)
            
            split_data.append({
                'rank': i + 1, 'node_name': split['node_name'], 'split_dict': split,
                'idx_A': local_idx_A, 'idx_B': local_idx_B, 'P_A': P_A, 'P_B': P_B,
                'lambda_obs': lambda_obs.item(), 'V_A': V_A.cpu().numpy(), 'V_B': V_B.cpu().numpy()
            })
            
        # Parametric Bootstrap
        C_replicates = 1000  # Default to 1000 for standard testing, adjust as needed
        print(f"   -> Bootstrapping Covariance LRT ({C_replicates} replicates)...")
        max_lambdas_null = []
        for c in range(C_replicates):
            X_sim = simulate_null_data(n_sf, p_current, mu_hat_sf, L_U, L_V)
            lambda_sims = [compute_mle_and_lrt(X_sim[sd['idx_A']], X_sim[sd['idx_B']], sd['P_A'], sd['P_B'], len(sd['idx_A']), len(sd['idx_B'])).item() for sd in split_data]
            max_lambdas_null.append(max(lambda_sims))
            
        max_lambdas_null = np.array(max_lambdas_null)
        
        # Save Results
        for sd in split_data:
            p_adj = (np.sum(max_lambdas_null >= sd['lambda_obs']) + 1) / (C_replicates + 1)
            is_sig = p_adj <= 0.05
            
            print(f"      Split {sd['rank']} ({sd['node_name']}): L_obs={sd['lambda_obs']:.2f}, p={p_adj:.4f} [{'SIG' if is_sig else 'NO'}]")
            
            dest_dir = sf_sig_dir if is_sig else sf_non_sig_dir
            split_dir = os.path.join(dest_dir, f"rank{sd['rank']}")
            if os.path.exists(split_dir): shutil.rmtree(split_dir)
            os.makedirs(split_dir, exist_ok=True)
            
            if is_sig:
                calc_dir = os.path.join(split_dir, "calculations")
                os.makedirs(calc_dir, exist_ok=True)
                pd.DataFrame(sd['V_A']).to_csv(os.path.join(calc_dir, f"embedding_cov_rank{sd['rank']}_subA.csv"))
                pd.DataFrame(sd['V_B']).to_csv(os.path.join(calc_dir, f"embedding_cov_rank{sd['rank']}_subB.csv"))
                
                # Extract original names for JSON
                group_a_names = [sf_leaves[idx] for idx in sd['idx_A']]
                group_b_names = [sf_leaves[idx] for idx in sd['idx_B']]
                
                split_data_out = {
                    "subfamily": sf_idx, "rank": sd['rank'], "node_name": sd['node_name'],
                    "lambda_obs": sd['lambda_obs'], "p_adj": p_adj,
                    "group_a": group_a_names, "group_b": group_b_names, "folder_path": split_dir
                }
                with open(os.path.join(split_dir, f"split_rank{sd['rank']}.json"), 'w') as f:
                    json.dump(split_data_out, f, indent=4)
                    
            all_results.append({
                'subfamily': sf_idx, 'rank': sd['rank'], 'node': sd['node_name'],
                'lambda': sd['lambda_obs'], 'p_adj': p_adj, 'sig': is_sig, 'folder': split_dir 
            })
    # Save a summary JSON of all subfamilies for the macro-visualization
    subfamilies_summary = {}
    for sf_idx, subfamily in enumerate(stable_subfamilies, 1):
        subfamilies_summary[f"subfamily_{sf_idx}"] = list(subfamily['leaves'])
        
    with open(os.path.join(output_path, "subfamilies_summary.json"), 'w') as f:
        json.dump(subfamilies_summary, f, indent=4)

    return all_results, total_raw_splits, total_unique_splits, final_p_dims
