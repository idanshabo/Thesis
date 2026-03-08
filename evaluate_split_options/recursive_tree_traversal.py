import random
from itertools import combinations
import torch
from evaluate_split_options.phylogenetic_anova import phylogenetic_anova_rrpp
from ete3 import Tree

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

def recursive_mean_split(tree_node, Y_global, C_global, global_names, tree_alpha=0.1, anova_alpha=0.05, n_permutations=999, id_to_seq=None):
    """
    Recursively divides a phylogenetic tree into stable sub-families based on 
    significant mean shifts (Phylogenetic ANOVA).
    
    Returns a list of dictionaries, each representing a stable sub-family.
    """
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
                matches = sum(1 for a, b in zip(s1, s2) if a == b)
                total_sim += matches / max(len(s1), 1)
            sim_pct = (total_sim / len(pairs)) * 100.0

        print(f"      sequence similarity is {sim_pct:.2f}%")
        print(f"      normalized_total_branch_length is {norm_branch_len:.4f}")

    candidates = find_candidate_splits_from_node(tree_node, tree_alpha=tree_alpha)
    
    if not candidates:
        # BASE CASE: No valid candidate splits found. This is a stable sub-family.
        return [{'node': tree_node, 'leaves': set(current_leaves_list), 'indices': current_global_indices, 
                 'sim_pct': sim_pct, 'norm_branch_len': norm_branch_len}] # <-- ADDED METRICS
        
    # 4. Evaluate candidates using the Phylogenetic ANOVA
    best_p = 1.0
    best_F = -1.0
    best_split = None
    
    for split in candidates:
        # Map the split groups to local indices (0 to len(current_leaves)-1) for the ANOVA
        local_idx_a = [i for i, name in enumerate(current_leaves_list) if name in split['group_a']]
        local_idx_b = [i for i, name in enumerate(current_leaves_list) if name in split['group_b']]
        
        if not local_idx_a or not local_idx_b:
            continue
            
        F_obs, p_val = phylogenetic_anova_rrpp(
            Y_local, C_local, local_idx_a, local_idx_b, n_permutations=n_permutations
        )
        
        # Track the most significant split
        if p_val < best_p or (p_val == best_p and F_obs > best_F):
            best_p = p_val
            best_F = F_obs
            best_split = split
            
    # 5. Recursive Step
    if best_split and best_p <= anova_alpha:
        node_name = best_split['node_name']
        size_parent = len(current_leaves_list)
        size_A = len(best_split['group_a'])
        size_B = len(best_split['group_b'])
        
        print(f"      [!] SIGNIFICANT SPLIT ACCEPTED:")
        print(f"          Parent Clade ({size_parent}) --> Group A ({size_A}) & Group B ({size_B})")
        print(f"          Stats: Node '{node_name}' | p={best_p:.4f}, F={best_F:.2f}")
        
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
        print(f"      [=] Clade of {len(current_leaves_list)} sequences is stable (No further mean shifts).")
        return [{'node': tree_node, 'leaves': set(current_leaves_list), 'indices': current_global_indices, 
                 'sim_pct': sim_pct, 'norm_branch_len': norm_branch_len}] # <-- ADDED METRICS
