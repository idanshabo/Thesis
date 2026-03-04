import torch
from evaluate_split_options.phylogenetic_anova import phylogenetic_anova_rrpp
from ete3 import Tree

def find_candidate_splits_from_node(node, min_support=0.8, min_prop=0.1):
    """
    Helper to find candidate splits directly from an ete3 TreeNode.
    Evaluates splits within the current clade based on support and size.
    """
    all_leaves = set(node.get_leaf_names())
    total_leaves = len(all_leaves)
    candidates = []
    
    for child in node.traverse("postorder"):
        # Skip leaves and the root of the current sub-tree
        if child.is_leaf() or child == node:
            continue
            
        # Handle nodes that might be missing support values
        support = getattr(child, "support", 1.0) 
        if support < min_support:
            continue
            
        clade_leaves = set(child.get_leaf_names())
        clade_size = len(clade_leaves)
        min_size = min_prop * total_leaves
        
        # Enforce the size rule for the current sub-tree
        if clade_size < min_size or (total_leaves - clade_size) < min_size:
            continue
            
        candidates.append({
            'node': child,
            'group_a': clade_leaves,
            'group_b': all_leaves - clade_leaves,
            'support': support,
            'node_name': getattr(child, "name", "Unnamed")
        })
        
    return candidates

def recursive_mean_split(tree_node, Y_global, C_global, global_names, min_prop=0.1, alpha=0.05, n_permutations=999):
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
    
    # 3. Find candidate splits in this clade
    candidates = find_candidate_splits_from_node(tree_node, min_support=0.8, min_prop=min_prop)
    
    if not candidates:
        # BASE CASE: No valid candidate splits found. This is a stable sub-family.
        return [{'node': tree_node, 'leaves': set(current_leaves_list), 'indices': current_global_indices}]
        
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
    if best_split and best_p <= alpha:
        node_name = best_split['node_name']
        print(f"   -> Significant mean shift at node '{node_name}' (p={best_p:.4f}, F={best_F:.2f}). Splitting tree...")
        
        # Physically detach Group A's node from the tree. 
        # This elegantly leaves `tree_node` containing ONLY Group B.
        node_A = best_split['node'].detach() 
        node_B = tree_node 
        
        # Recurse on both new sub-trees
        stable_A = recursive_mean_split(node_A, Y_global, C_global, global_names, min_prop, alpha, n_permutations)
        stable_B = recursive_mean_split(node_B, Y_global, C_global, global_names, min_prop, alpha, n_permutations)
        
        return stable_A + stable_B
        
    else:
        # BASE CASE: No split was statistically significant.
        print(f"   -> Clade with {len(current_leaves_list)} leaves is stable (no mean shifts).")
        return [{'node': tree_node, 'leaves': set(current_leaves_list), 'indices': current_global_indices}]
