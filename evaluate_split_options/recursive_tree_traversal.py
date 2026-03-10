import random
import numpy as np
from itertools import combinations
import torch
from evaluate_split_options.phylogenetic_anova import phylogenetic_anova_rrpp
from ete3 import Tree

def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Applies the Benjamini-Hochberg FDR correction.
    Returns a boolean array of significance and the adjusted p-values.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Calculate BH critical values: (i / m) * alpha
    ranks = np.arange(1, n + 1)
    critical_values = (ranks / n) * alpha
    
    # Find the largest p-value that is less than or equal to its critical value
    significant_flags = sorted_p <= critical_values
    max_sig_index = np.max(np.where(significant_flags)[0]) if np.any(significant_flags) else -1
    
    # Create the final boolean array for significance
    is_significant = np.zeros(n, dtype=bool)
    if max_sig_index >= 0:
        # Everything up to the max_sig_index is considered significant under FDR
        is_significant[sorted_indices[:max_sig_index + 1]] = True
        
    # Calculate adjusted p-values (q-values)
    # q = min(p * m / i, 1.0), evaluated from right to left to ensure monotonicity
    adjusted_p = np.zeros(n)
    min_q = 1.0
    for i in range(n - 1, -1, -1):
        idx = sorted_indices[i]
        q_val = min(1.0, sorted_p[i] * n / ranks[i])
        min_q = min(min_q, q_val)
        adjusted_p[idx] = min_q
        
    return is_significant, adjusted_p

def evaluate_splits_adaptively(candidates, Y_local, C_local, current_leaves_list, 
                               anova_alpha=0.05, pass1_perms=999, pass2_perms=9999):
    """
    Evaluates candidate splits using a 2-pass adaptive RRPP to save time, 
    then applies an FDR correction to prevent multiple testing bias.
    """
    total_candidates = len(candidates)
    if total_candidates == 0:
        return None, 1.0, -1.0
        
    p_values = np.ones(total_candidates)
    F_values = np.zeros(total_candidates)
    
    # --- PASS 1: Low Resolution (Identify Promising Splits) ---
    promising_indices = []
    for i, split in enumerate(candidates):
        local_idx_a = [idx for idx, name in enumerate(current_leaves_list) if name in split['group_a']]
        local_idx_b = [idx for idx, name in enumerate(current_leaves_list) if name in split['group_b']]
        
        if not local_idx_a or not local_idx_b:
            continue
            
        F_obs, p_val = phylogenetic_anova_rrpp(
            Y_local, C_local, local_idx_a, local_idx_b, n_permutations=pass1_perms
        )
        p_values[i] = p_val
        F_values[i] = F_obs
        
        # If the split is even remotely close to significant, keep it for Pass 2
        # A threshold of 0.15 is generous enough to prevent false negatives at this stage
        if p_val <= 0.15: 
            promising_indices.append((i, local_idx_a, local_idx_b))

    # --- PASS 2: High Resolution (For FDR Correction) ---
    for i, local_idx_a, local_idx_b in promising_indices:
        F_obs, p_val = phylogenetic_anova_rrpp(
            Y_local, C_local, local_idx_a, local_idx_b, n_permutations=pass2_perms
        )
        p_values[i] = p_val
        F_values[i] = F_obs

    # --- Multiple Testing Correction (FDR) ---
    is_sig, adjusted_p = benjamini_hochberg_correction(p_values, alpha=anova_alpha)
    
    # Find the BEST split among those that survived the FDR correction
    best_split = None
    best_adj_p = 1.0
    best_F = -1.0
    
    for i in range(total_candidates):
        if is_sig[i]: # Only consider splits that survived FDR
            # Break ties using the F-statistic
            if adjusted_p[i] < best_adj_p or (adjusted_p[i] == best_adj_p and F_values[i] > best_F):
                best_adj_p = adjusted_p[i]
                best_F = F_values[i]
                best_split = candidates[i]
                best_split['raw_p'] = p_values[i]
                
    return best_split, best_adj_p, best_F
                                   
def get_induced_branch_length(tree, leaf_subset, node_leaves_cache=None):
    """
    Calculates the exact branch length for the subtree induced by a subset of leaves.
    """
    if len(leaf_subset) <= 1:
        return 0.0
        
    induced_length = 0.0
    for node in tree.traverse():
        if node.is_root():
            continue
            
        if node_leaves_cache is not None:
            node_leaves = node_leaves_cache[node]
        else:
            node_leaves = set(node.get_leaf_names())
            
        in_subset = len(node_leaves.intersection(leaf_subset))
        out_subset = len(leaf_subset) - in_subset
        
        # If the edge separates elements of the subset, it connects them in the induced tree
        if in_subset > 0 and out_subset > 0:
            induced_length += getattr(node, "dist", 0.0)
            
    return induced_length


def find_candidate_splits_from_node(node, tree_alpha=0.1, min_absolute_size=20, k=None):
    """
    Helper to find candidate splits directly from an ete3 TreeNode.
    Evaluates splits based on absolute size, and the alpha rule 
    (tree_alpha) applied to BOTH proportional species count and induced branch length.
    Support condition has been removed.
    """
    all_leaves = set(node.get_leaf_names())
    total_leaves = len(all_leaves)
    
    # If the entire sub-tree is already too small, don't even try to split it
    if total_leaves < min_absolute_size * 2: 
        return []
        
    total_tree_length = sum(getattr(n, "dist", 0.0) for n in node.traverse() if not n.is_root())
    
    # Pre-cache leaves for massive speedup during branch length calculations
    node_leaves_cache = {n: set(n.get_leaf_names()) for n in node.traverse() if not n.is_root()}
        
    candidates = []
    
    for child in node.traverse("postorder"):
        if child.is_leaf() or child == node:
            continue
            
        clade_leaves = node_leaves_cache[child]
        clade_size = len(clade_leaves)
        group_b_leaves = all_leaves - clade_leaves
        
        # 1. Enforce Leaf Count Rule
        min_allowed_by_prop = tree_alpha * total_leaves
        actual_min_allowed = max(min_allowed_by_prop, min_absolute_size)
        
        if clade_size < actual_min_allowed or (total_leaves - clade_size) < actual_min_allowed:
            continue
            
        # 2. Enforce Branch Length Rule (tree_alpha alpha)
        clade_A_len = get_induced_branch_length(node, clade_leaves, node_leaves_cache)
        clade_B_len = get_induced_branch_length(node, group_b_leaves, node_leaves_cache)
        
        if clade_A_len < tree_alpha * total_tree_length or clade_B_len < tree_alpha * total_tree_length:
            continue
            
        candidates.append({
            'node': child,
            'group_a': clade_leaves,
            'group_b': group_b_leaves,
            'length': getattr(child, "dist", 0.0),
            'node_name': getattr(child, "name", "Unnamed")
        })
        
    # Apply the k limit if specified
    if k is not None:
        candidates = candidates[:k]
        
    return candidates

def recursive_mean_split(tree_node, Y_global, C_global, global_names, tree_alpha=0.1, anova_alpha=0.05, n_permutations=999, id_to_seq=None, split_history=None):
    """
    Recursively divides a phylogenetic tree into stable sub-families based on 
    significant mean shifts (Phylogenetic ANOVA).
    
    Returns a list of dictionaries, each representing a stable sub-family.
    """
    if split_history is None:
        split_history = []
        
    # 1. Fix the order of current leaves to ensure indices align perfectly
    current_leaves_list = list(tree_node.get_leaf_names())
    
    # Helper to map leaf names to the global indices in our tensors
    def get_global_indices(leaf_list):
        idx = []
        for leaf in leaf_list:
            leaf_str = str(leaf).strip()
            if leaf_str in global_names:
                idx.append(global_names.index(leaf_str))
            else:
                alt_leaf = leaf_str.replace('/', '_')
                if alt_leaf in global_names:
                    idx.append(global_names.index(alt_leaf))
        return idx
        
    current_global_indices = get_global_indices(current_leaves_list)
    
    # 2. Extract local Y and C matrices for the current clade
    idx_tensor = torch.tensor(current_global_indices, dtype=torch.long, device=Y_global.device)
    Y_local = Y_global[idx_tensor]
    
    # Slice rows and columns to get the local covariance matrix
    C_local = C_global[idx_tensor][:, idx_tensor]
    
    # Shift to the local root and clamp floating-point noise
    C_local = torch.clamp(C_local - torch.min(C_local), min=0.0)
    
    # 3. Find candidate splits in this clade
    print(f"\n   -> Analyzing Clade with {len(current_leaves_list)} sequences for mean shifts...")

    norm_branch_len = 0.0
    sim_pct = 100.0
    
    if id_to_seq is not None:
        # Calculate Branch Length
        total_dist = sum([n.dist for n in tree_node.traverse() if n != tree_node])
        norm_branch_len = total_dist / max(len(current_leaves_list), 1)

        # Calculate Sequence Similarity
        seqs = []
        for leaf in current_leaves_list:
            clean_leaf = str(leaf).replace('/', '_')
            if clean_leaf in id_to_seq:
                seqs.append(id_to_seq[clean_leaf])
            elif str(leaf) in id_to_seq:
                seqs.append(id_to_seq[str(leaf)])

        if len(seqs) >= 2:
            pairs = list(combinations(range(len(seqs)), 2))
            if len(pairs) > 500:  # Cap for speed during deep recursion
                pairs = random.sample(pairs, 500)
            
            total_sim = 0
            for i, j in pairs:
                s1, s2 = seqs[i], seqs[j]
                
                # Get lengths without gaps
                len1 = len(s1.replace('-', ''))
                len2 = len(s2.replace('-', ''))
                denom = min(len1, len2)
                
                if denom == 0:
                    continue
                
                # Count column-by-column matches, ignoring gap-to-gap
                matches = sum(1 for a, b in zip(s1, s2) if a == b and a != '-')
                total_sim += (matches / denom) * 100.0
                
            sim_pct = total_sim / len(pairs) if pairs else 0.0

        print(f"      sequence similarity is {sim_pct:.2f}%")
        print(f"      normalized_total_branch_length is {norm_branch_len:.4f}")

    candidates = find_candidate_splits_from_node(tree_node, tree_alpha=tree_alpha)

    print(f"      Found {len(candidates)} valid candidate splits to evaluate.")
    if not candidates:
        # BASE CASE: No valid candidate splits found. This is a stable sub-family.
        print(f"      [=] Clade of {len(current_leaves_list)} sequences is stable (No valid splits meet size/alpha criteria).")
        return [{'node': tree_node, 'leaves': set(current_leaves_list), 'indices': current_global_indices, 
                 'sim_pct': sim_pct, 'norm_branch_len': norm_branch_len, 'split_history': split_history}]
        
    # 4. Evaluate candidates using the Adaptive Phylogenetic ANOVA + FDR
    best_split, best_p, best_F = evaluate_splits_adaptively(
        candidates, Y_local, C_local, current_leaves_list, 
        anova_alpha=anova_alpha, pass1_perms=999, pass2_perms=9999
    )
            
    # 5. Recursive Step
    if best_split is not None:
        node_name = best_split['node_name']
        size_parent = len(current_leaves_list)
        size_A = len(best_split['group_a'])
        size_B = len(best_split['group_b'])
        
        print(f"      [!] SIGNIFICANT SPLIT ACCEPTED (FDR Corrected):")
        print(f"          Parent Clade ({size_parent}) --> Group A ({size_A}) & Group B ({size_B})")
        print(f"          Stats: Node '{node_name}' | adj_p={best_p:.5f} (raw_p={best_split.get('raw_p', 'N/A'):.5f}), F={best_F:.2f}")

        new_history = split_history + [f"node {node_name}"]
        
        # Group A
        node_A = tree_node.copy()
        node_A.prune([str(leaf) for leaf in best_split['group_a']], preserve_branch_length=True)
        
        # Group B
        node_B = tree_node.copy()
        node_B.prune([str(leaf) for leaf in best_split['group_b']], preserve_branch_length=True)
        
        stable_A = recursive_mean_split(node_A, Y_global, C_global, global_names, tree_alpha, anova_alpha, n_permutations, id_to_seq)
        stable_B = recursive_mean_split(node_B, Y_global, C_global, global_names, tree_alpha, anova_alpha, n_permutations, id_to_seq)
        return stable_A + stable_B
        
    else:
        print(f"      [=] Clade of {len(current_leaves_list)} sequences is stable (No splits survived FDR correction).")
        return [{'node': tree_node, 'leaves': set(current_leaves_list), 'indices': current_global_indices, 
                 'sim_pct': sim_pct, 'norm_branch_len': norm_branch_len,
                 'split_history': split_history}]
