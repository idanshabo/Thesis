import random
import numpy as np
from itertools import combinations

def calc_exact_msa_similarity_in_memory(group_leaves, id_to_seq):
    seqs = []
    for leaf in group_leaves:
        clean_leaf = str(leaf).replace('/', '_')
        if clean_leaf in id_to_seq:
            seqs.append(id_to_seq[clean_leaf])
        elif str(leaf) in id_to_seq:
            seqs.append(id_to_seq[str(leaf)])

    n_seqs = len(seqs)
    if n_seqs < 2: 
        return 100.0

    if n_seqs <= 500:
        pairs = list(combinations(seqs, 2))
    else:
        pairs = []
        for _ in range(10000):
            i, j = random.sample(range(n_seqs), 2)
            pairs.append((seqs[i], seqs[j]))
            
    total_sim = 0.0
    for seq1, seq2 in pairs:
        len1 = len(seq1.replace('-', ''))
        len2 = len(seq2.replace('-', ''))
        denom = min(len1, len2)
        if denom == 0: continue
            
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
        total_sim += (matches / denom) * 100.0
        
    return total_sim / len(pairs) if pairs else 0.0

def generate_robust_k_partition(tree_node, target_k):
    """
    Randomly selects K-1 internal nodes to partition the tree into exactly K valid clades.
    This safely bypasses strict global size limits to allow for large K values.
    """
    if target_k <= 1:
        return [list(tree_node.get_leaf_names())]

    # Get all internal nodes that form a clade of at least 2 leaves
    valid_nodes = []
    for n in tree_node.traverse():
        if not n.is_root() and not n.is_leaf():
            if len(n.get_leaves()) >= 2:
                valid_nodes.append(n)
                
    if len(valid_nodes) < target_k - 1:
        return None # Literally not enough branches in the tree to make K groups!

    # Try to find a valid K-partition
    for _ in range(2000):
        sampled_nodes = random.sample(valid_nodes, target_k - 1)
        
        groups = {n: [] for n in sampled_nodes}
        groups["root"] = []
        
        # Assign every leaf to its lowest selected ancestor
        for leaf in tree_node.get_leaves():
            assigned = False
            curr = leaf.up
            while curr is not None:
                if curr in sampled_nodes:
                    groups[curr].append(leaf.name)
                    assigned = True
                    break
                curr = curr.up
            if not assigned:
                groups["root"].append(leaf.name)
                
        # Filter out empty groups (happens if one node completely encompasses another)
        non_empty_groups = [g for g in groups.values() if len(g) > 0]
        
        # If we successfully isolated exactly K non-empty phylogenetic groups, we are done!
        if len(non_empty_groups) == target_k:
            return non_empty_groups
            
    return None 

def evaluate_strict_branch_baselines(tree_node, id_to_seq, target_k=2, min_absolute_size=10, num_trials=100):
    """
    Evaluates baseline homogeneity by partitioning the tree into exactly K subfamilies.
    """
    all_leaves = set(tree_node.get_leaf_names())
    total_size = len(all_leaves)
    
    baseline_results = {'sim_pct': [], 'group_sizes': []}
    successful_trials = 0

    for _ in range(num_trials):
        # Generate a random, valid phylogenetic K-partition
        partition = generate_robust_k_partition(tree_node, target_k)
        
        if not partition:
            continue
            
        successful_trials += 1
        
        # Calculate size-weighted average similarity for this K-partition
        trial_sim = 0.0
        sizes = []
        for group in partition:
            group_size = len(group)
            sizes.append(group_size)
            if id_to_seq:
                group_sim = calc_exact_msa_similarity_in_memory(group, id_to_seq)
                trial_sim += (group_sim * group_size)
                
        baseline_results['sim_pct'].append(trial_sim / total_size)
        baseline_results['group_sizes'].append(sizes)

    if successful_trials == 0:
        return None

    return {
        'mean_random_sim_pct': np.mean(baseline_results['sim_pct']),
        'total_valid_edges_tested': successful_trials,
        'example_group_sizes': baseline_results['group_sizes'][0] 
    }
