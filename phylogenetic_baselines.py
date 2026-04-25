import random
import numpy as np
from itertools import combinations
from evaluate_split_options.recursive_tree_traversal import get_induced_branch_length

def calc_exact_msa_similarity_in_memory(group_leaves, id_to_seq):
    """
    Replicates the exact math from MetadataTracker.calc_and_add_sequence_similarity 
    but runs in-memory using the id_to_seq dictionary to avoid disk I/O bottlenecks.
    """
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

    # Match the MetadataTracker's sampling threshold exactly
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
        
        if denom == 0:
            continue
            
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
        total_sim += (matches / denom) * 100.0
        
    return total_sim / len(pairs) if pairs else 0.0


def evaluate_strict_branch_baselines(tree_node, id_to_seq, tree_alpha=0.1, min_absolute_size=20, num_trials=100):
    """
    Evaluates baseline homogeneity using biologically valid, arbitrary branch cuts
    that strictly adhere to the algorithm's size and branch length constraints.
    """
    all_leaves = set(tree_node.get_leaf_names())
    total_size = len(all_leaves)
    total_tree_length = sum(getattr(n, "dist", 0.0) for n in tree_node.traverse() if not n.is_root())
    
    # Calculate the exact minimum allowed size
    min_allowed_by_prop = tree_alpha * total_size
    actual_min_allowed = max(min_allowed_by_prop, min_absolute_size)
    
    valid_splits = []
    node_leaves_cache = {n: set(n.get_leaf_names()) for n in tree_node.traverse() if not n.is_root()}
    
    # 1. Collect all valid internal edges that pass the strict rules
    for child in tree_node.traverse("postorder"):
        if child.is_leaf() or child == tree_node:
            continue
            
        clade_leaves = node_leaves_cache[child]
        clade_size = len(clade_leaves)
        group_b_leaves = all_leaves - clade_leaves
        
        # Rule 1: Leaf Count & Percentage Constraints
        if clade_size < actual_min_allowed or (total_size - clade_size) < actual_min_allowed:
            continue
            
        # Rule 2: Induced Branch Length Constraints
        clade_A_len = get_induced_branch_length(tree_node, clade_leaves, node_leaves_cache)
        clade_B_len = get_induced_branch_length(tree_node, group_b_leaves, node_leaves_cache)
        
        if clade_A_len < tree_alpha * total_tree_length or clade_B_len < tree_alpha * total_tree_length:
            continue
            
        valid_splits.append((clade_leaves, group_b_leaves))
            
    if not valid_splits:
        return None # No alternative valid branches exist under these strict constraints

    # 2. Randomly sample from the valid candidate pool
    sampled_splits = valid_splits
    if len(valid_splits) > num_trials:
        sampled_splits = random.sample(valid_splits, num_trials)

    baseline_results = {'sim_pct': [], 'branch_len': []}

    # 3. Calculate metrics for these comparable, strict phylogenetic cuts
    for group_A, group_B in sampled_splits:
        size_A, size_B = len(group_A), len(group_B)
        
        # Branch Lengths
        len_A = get_induced_branch_length(tree_node, group_A, node_leaves_cache)
        len_B = get_induced_branch_length(tree_node, group_B, node_leaves_cache)
        norm_len_A = len_A / max(size_A, 1)
        norm_len_B = len_B / max(size_B, 1)
        
        avg_rand_branch_len = ((norm_len_A * size_A) + (norm_len_B * size_B)) / total_size
        baseline_results['branch_len'].append(avg_rand_branch_len)

        # Sequence Similarity
        if id_to_seq:
            sim_A = calc_exact_msa_similarity_in_memory(group_A, id_to_seq)
            sim_B = calc_exact_msa_similarity_in_memory(group_B, id_to_seq)
            avg_rand_sim = ((sim_A * size_A) + (sim_B * size_B)) / total_size
            baseline_results['sim_pct'].append(avg_rand_sim)

    return {
        'mean_random_sim_pct': np.mean(baseline_results['sim_pct']) if baseline_results['sim_pct'] else None,
        'mean_random_branch_len': np.mean(baseline_results['branch_len']),
        'total_valid_edges_tested': len(sampled_splits)
    }
