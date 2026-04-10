"""
Tree-weighted Adjusted Rand Index (tw-ARI) for comparing two tree-consistent
partitions of the same phylogeny.

Standard ARI overestimates agreement because both partitions are constrained
to the same tree structure. This module adjusts for that by weighting leaf
pairs by their informativeness (how likely they are to be split by a random
tree-consistent partition) and computing the expected agreement under a
tree-aware null model.

For general graphs, a graph-constrained ARI is intractable (#P-complete).
For trees, it is exactly solvable because C(n-1, k-1) tree-consistent
partitions exist and can be uniformly sampled by choosing random edges.

References:
    - Hubert & Arabie (1985), "Comparing Partitions", J. Classification
    - Hall & Colijn (2019), MBE — uniform sampling of tree-consistent partitions
    - Poulin & Theberge (2020), IEEE TPAMI — graph-aware partition measures
"""

import numpy as np
from scipy.special import comb
from itertools import combinations


def get_eligible_edges(tree, tau=0.1):
    """
    Get internal edges that pass the tau threshold for candidate splits.

    An edge (defined by its child node) is eligible if both sides have
    >= tau fraction of total leaves AND >= tau fraction of total branch length.

    Parameters
    ----------
    tree : ete3.Tree
        Rooted phylogenetic tree.
    tau : float
        Threshold fraction (default 0.1).

    Returns
    -------
    eligible_nodes : list of ete3.TreeNode
        Internal nodes whose parent edges are eligible.
    """
    all_leaves = tree.get_leaves()
    n_total = len(all_leaves)
    total_branch = sum(n.dist for n in tree.traverse() if n != tree)

    if total_branch == 0 or n_total < 2:
        return []

    eligible = []
    for node in tree.traverse("postorder"):
        if node == tree or node.is_leaf():
            continue

        n_desc = len(node.get_leaves())
        n_other = n_total - n_desc

        if min(n_desc, n_other) < tau * n_total:
            continue

        desc_leaves_set = set(l.name for l in node.get_leaves())
        desc_branch = sum(
            n.dist for n in node.traverse() if n != node
        )
        other_branch = total_branch - desc_branch - node.dist
        if min(desc_branch, other_branch) < tau * total_branch:
            continue

        eligible.append(node)

    return eligible


def _get_ancestors(node):
    """Return set of all ancestor nodes from node to root (inclusive)."""
    ancestors = set()
    current = node
    while current is not None:
        ancestors.add(current)
        current = current.up
    return ancestors


def compute_path_eligible_edges(tree, leaf_names, eligible_nodes):
    """
    For each pair of leaves, count how many eligible edges lie on the
    path between them.

    An eligible edge (defined by node v) is on the path from leaf i to
    leaf j if and only if v is an ancestor of exactly one of them.

    Parameters
    ----------
    tree : ete3.Tree
        Rooted phylogenetic tree.
    leaf_names : list of str
        Ordered leaf names.
    eligible_nodes : list of ete3.TreeNode
        Eligible internal nodes (from get_eligible_edges).

    Returns
    -------
    m_matrix : numpy.ndarray
        (n, n) matrix where m_matrix[i,j] = number of eligible edges
        on the path from leaf i to leaf j.
    """
    n = len(leaf_names)
    name_to_idx = {name: i for i, name in enumerate(leaf_names)}

    # For each eligible node, precompute which leaves descend from it
    eligible_desc = []
    for node in eligible_nodes:
        desc_set = set()
        for leaf in node.get_leaves():
            if leaf.name in name_to_idx:
                desc_set.add(name_to_idx[leaf.name])
        eligible_desc.append(desc_set)

    m_matrix = np.zeros((n, n), dtype=np.int32)

    for desc_set in eligible_desc:
        # This edge separates desc_set from its complement
        # It's on the path between i and j iff exactly one of them is in desc_set
        desc_arr = np.zeros(n, dtype=bool)
        for idx in desc_set:
            desc_arr[idx] = True

        # Outer product: desc_arr[i] XOR desc_arr[j] means edge is on their path
        for i in range(n):
            for j in range(i + 1, n):
                if desc_arr[i] != desc_arr[j]:
                    m_matrix[i, j] += 1
                    m_matrix[j, i] += 1

    return m_matrix


def _partition_to_label_array(partition, leaf_names):
    """Convert a partition (list of sets) to integer label array."""
    name_to_label = {}
    for label, group in enumerate(partition):
        for name in group:
            name_to_label[name] = label
    return np.array([name_to_label.get(name, -1) for name in leaf_names])


def compute_tw_ari(partition_1, partition_2, tree, tau=0.1):
    """
    Compute tree-weighted Adjusted Rand Index between two tree-consistent
    partitions of the same phylogeny.

    Parameters
    ----------
    partition_1 : list of sets of str
        First partition, e.g. [{'A','B'}, {'C','D','E'}].
    partition_2 : list of sets of str
        Second partition.
    tree : ete3.Tree
        Rooted phylogenetic tree (same tree for both partitions).
    tau : float
        Threshold for eligible edges (default 0.1).

    Returns
    -------
    result : dict
        Keys: 'tw_ari', 'wri_observed', 'wri_expected', 'standard_ari',
              'n_eligible_edges', 'k1', 'k2', 'n_leaves',
              'n_informative_pairs', 'n_total_pairs'.
    """
    # Find common leaves
    leaves_1 = set().union(*partition_1)
    leaves_2 = set().union(*partition_2)
    common = sorted(leaves_1 & leaves_2)
    n = len(common)

    if n < 2:
        return {
            'tw_ari': float('nan'), 'wri_observed': float('nan'),
            'wri_expected': float('nan'), 'standard_ari': float('nan'),
            'n_eligible_edges': 0, 'k1': len(partition_1),
            'k2': len(partition_2), 'n_leaves': n,
            'n_informative_pairs': 0, 'n_total_pairs': 0,
        }

    k1 = len(partition_1)
    k2 = len(partition_2)

    # Build label arrays for common leaves
    labels_1 = _partition_to_label_array(partition_1, common)
    labels_2 = _partition_to_label_array(partition_2, common)

    # Standard ARI for comparison
    try:
        from sklearn.metrics import adjusted_rand_score
        standard_ari = float(adjusted_rand_score(labels_1, labels_2))
    except ImportError:
        standard_ari = float('nan')

    # Get eligible edges
    eligible_nodes = get_eligible_edges(tree, tau)
    N_e = len(eligible_nodes)

    if N_e < 1:
        return {
            'tw_ari': float('nan'), 'wri_observed': float('nan'),
            'wri_expected': float('nan'), 'standard_ari': standard_ari,
            'n_eligible_edges': N_e, 'k1': k1, 'k2': k2, 'n_leaves': n,
            'n_informative_pairs': 0, 'n_total_pairs': n * (n - 1) // 2,
        }

    # Compute path eligible edges
    m_matrix = compute_path_eligible_edges(tree, common, eligible_nodes)

    # Use average k for p_ij computation
    k_avg = (k1 + k2) / 2.0
    k_for_comb = max(1, int(round(k_avg)))

    if N_e < k_for_comb - 1:
        return {
            'tw_ari': float('nan'), 'wri_observed': float('nan'),
            'wri_expected': float('nan'), 'standard_ari': standard_ari,
            'n_eligible_edges': N_e, 'k1': k1, 'k2': k2, 'n_leaves': n,
            'n_informative_pairs': 0, 'n_total_pairs': n * (n - 1) // 2,
        }

    # Precompute p_ij, w_ij, and agreement for all pairs
    denom = comb(N_e, k_for_comb - 1, exact=False)
    if denom == 0:
        denom = 1e-300

    sum_w = 0.0
    sum_w_agree = 0.0
    sum_w_expected = 0.0
    n_informative = 0

    for i in range(n):
        for j in range(i + 1, n):
            m_ij = m_matrix[i, j]
            remaining = N_e - m_ij
            if remaining < 0 or remaining < k_for_comb - 1:
                p_ij = 0.0
            else:
                p_ij = comb(remaining, k_for_comb - 1, exact=False) / denom

            w_ij = 2.0 * p_ij * (1.0 - p_ij)

            if w_ij < 1e-10:
                continue

            n_informative += 1
            sum_w += w_ij

            # Do partitions agree on this pair?
            same_in_1 = (labels_1[i] == labels_1[j])
            same_in_2 = (labels_2[i] == labels_2[j])
            agree = 1.0 if same_in_1 == same_in_2 else 0.0
            sum_w_agree += w_ij * agree

            # Expected agreement under null
            expected_agree = p_ij ** 2 + (1.0 - p_ij) ** 2
            sum_w_expected += w_ij * expected_agree

    if sum_w < 1e-10:
        return {
            'tw_ari': float('nan'), 'wri_observed': float('nan'),
            'wri_expected': float('nan'), 'standard_ari': standard_ari,
            'n_eligible_edges': N_e, 'k1': k1, 'k2': k2, 'n_leaves': n,
            'n_informative_pairs': n_informative,
            'n_total_pairs': n * (n - 1) // 2,
        }

    wri_obs = sum_w_agree / sum_w
    wri_exp = sum_w_expected / sum_w

    if abs(1.0 - wri_exp) < 1e-10:
        tw_ari = float('nan')
    else:
        tw_ari = (wri_obs - wri_exp) / (1.0 - wri_exp)

    return {
        'tw_ari': tw_ari,
        'wri_observed': wri_obs,
        'wri_expected': wri_exp,
        'standard_ari': standard_ari,
        'n_eligible_edges': N_e,
        'k1': k1,
        'k2': k2,
        'n_leaves': n,
        'n_informative_pairs': n_informative,
        'n_total_pairs': n * (n - 1) // 2,
    }


if __name__ == "__main__":
    # Simple test with a small tree
    from ete3 import Tree

    # Build a balanced tree: ((A,B),(C,D),(E,F));
    t = Tree("(((A:1,B:1):1,(C:1,D:1):1):1,(E:1,F:1):1);")
    t.set_outgroup(t.get_midpoint_outgroup())

    # Two partitions that mostly agree
    p1 = [{'A', 'B'}, {'C', 'D'}, {'E', 'F'}]
    p2 = [{'A', 'B'}, {'C', 'D', 'E'}, {'F'}]

    result = compute_tw_ari(p1, p2, t, tau=0.05)
    print("Test result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
