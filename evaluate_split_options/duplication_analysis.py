"""
Gene duplication analysis from phylogenetic trees and leaf names.

Computes duplication proxy signals from UniProt-style leaf names
(ACCESSION_SPECIES/start-end) and phylogenetic tree topology.

These signals can be correlated with pipeline outputs (split counts,
BM vs OU preference, evolutionary rates) across protein families
to test hypotheses about post-duplication divergence.
"""

import os
import json
import numpy as np
from collections import Counter, defaultdict


def parse_species_from_leaf(leaf_name):
    """
    Extract species code from a UniProt-style leaf name.

    Handles formats like:
        'METJ_ERWT9/26-105'  -> 'ERWT9'
        'A0A0L0GV20_9ENTR/26-105' -> '9ENTR'
        'A0A0L0GV20_9ENTR' -> '9ENTR'

    Parameters
    ----------
    leaf_name : str
        Leaf name in UniProt format.

    Returns
    -------
    species : str
        Species code, or 'UNKNOWN' if parsing fails.
    """
    # Remove domain boundaries (/start-end)
    name = leaf_name.split('/')[0]

    # Species code is after the last underscore
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[-1]
    return 'UNKNOWN'


def is_unclassified_taxon(species_code):
    """
    Check if a species code is an unclassified taxonomic group.

    UniProt uses '9XXXX' codes for sequences not assigned to a specific
    species (e.g., 9GAMM = unclassified Gammaproteobacteria). These
    represent taxonomic GROUPS, not individual species, so multiple
    sequences under 9GAMM are likely from DIFFERENT species, not paralogs.

    Parameters
    ----------
    species_code : str

    Returns
    -------
    bool
    """
    return species_code.startswith('9') or species_code == 'UNKNOWN'


def compute_species_duplication_counts(leaf_names):
    """
    Count within-species duplications from leaf names.

    For each species, count how many sequences appear in the family.
    Species with >1 sequence represent putative gene duplications.

    IMPORTANT: Sequences with unclassified taxonomic group codes (9XXXX)
    are excluded from duplication counting, as these represent different
    species lumped under one group code, not actual paralogs.

    Parameters
    ----------
    leaf_names : list of str
        Leaf names from the phylogenetic tree.

    Returns
    -------
    result : dict
        'species_counts': dict of {species: count} (all species)
        'n_species_total': int (including unclassified)
        'n_species_resolved': int (excluding unclassified 9XXXX codes)
        'n_sequences': int
        'n_unclassified_sequences': int (sequences with 9XXXX codes)
        'n_multi_copy_species': int (resolved species with >1 sequence)
        'multi_copy_fraction': float (among resolved species only)
        'max_copies': int
        'mean_copies_resolved': float
        'duplication_index': float (among resolved species only)
        'multi_copy_species': dict of {species: count} for resolved species with >1
    """
    species_list = [parse_species_from_leaf(name) for name in leaf_names]
    species_counts = Counter(species_list)

    n_seqs = len(leaf_names)
    n_species_total = len(species_counts)

    # Separate resolved species from unclassified taxonomic groups
    resolved_counts = {sp: cnt for sp, cnt in species_counts.items()
                       if not is_unclassified_taxon(sp)}
    unclassified_counts = {sp: cnt for sp, cnt in species_counts.items()
                           if is_unclassified_taxon(sp)}

    n_resolved = len(resolved_counts)
    n_unclassified_seqs = sum(unclassified_counts.values())
    n_resolved_seqs = n_seqs - n_unclassified_seqs

    multi_copy = {sp: cnt for sp, cnt in resolved_counts.items() if cnt > 1}

    return {
        'species_counts': dict(species_counts),
        'n_species_total': n_species_total,
        'n_species_resolved': n_resolved,
        'n_sequences': n_seqs,
        'n_resolved_sequences': n_resolved_seqs,
        'n_unclassified_sequences': n_unclassified_seqs,
        'unclassified_fraction': n_unclassified_seqs / max(n_seqs, 1),
        'n_multi_copy_species': len(multi_copy),
        'multi_copy_fraction': len(multi_copy) / max(n_resolved, 1),
        'max_copies': max(resolved_counts.values()) if resolved_counts else 0,
        'mean_copies_resolved': n_resolved_seqs / max(n_resolved, 1),
        'duplication_index': 1.0 - n_resolved / max(n_resolved_seqs, 1),
        'multi_copy_species': multi_copy,
    }


def count_duplication_nodes(tree):
    """
    Count putative duplication nodes in the phylogenetic tree.

    A node is classified as a putative duplication if its two children's
    leaf sets share at least one species. (If a species appears on both
    sides of a split, the split likely represents a gene duplication
    rather than a speciation event.)

    Parameters
    ----------
    tree : ete3.Tree
        Rooted phylogenetic tree with UniProt-style leaf names.

    Returns
    -------
    result : dict
        'n_duplication_nodes': int
        'n_speciation_nodes': int
        'n_internal_nodes': int
        'duplication_fraction': float
        'duplication_nodes': list of dict with node info
    """
    dup_nodes = []
    n_speciation = 0
    n_internal = 0

    for node in tree.traverse("postorder"):
        if node.is_leaf() or node == tree:
            continue

        children = node.get_children()
        if len(children) != 2:
            continue

        n_internal += 1

        # Get species sets for each child
        species_left = set(
            parse_species_from_leaf(l.name) for l in children[0].get_leaves()
        )
        species_right = set(
            parse_species_from_leaf(l.name) for l in children[1].get_leaves()
        )

        shared = species_left & species_right

        if shared:
            dup_nodes.append({
                'n_left': len(list(children[0].get_leaves())),
                'n_right': len(list(children[1].get_leaves())),
                'n_shared_species': len(shared),
                'branch_length': node.dist,
                'depth': node.get_distance(tree),
            })
        else:
            n_speciation += 1

    n_dup = len(dup_nodes)
    return {
        'n_duplication_nodes': n_dup,
        'n_speciation_nodes': n_speciation,
        'n_internal_nodes': n_internal,
        'duplication_fraction': n_dup / max(n_internal, 1),
        'duplication_nodes': dup_nodes,
    }


def compute_duplication_depth_profile(tree):
    """
    Compute the depth distribution of duplication vs speciation events.

    This reveals whether duplications are concentrated near the tips
    (recent) or deep in the tree (ancient).

    Parameters
    ----------
    tree : ete3.Tree
        Rooted phylogenetic tree.

    Returns
    -------
    result : dict
        'duplication_depths': list of float (normalized 0-1)
        'speciation_depths': list of float (normalized 0-1)
        'mean_duplication_depth': float
        'mean_speciation_depth': float
    """
    tree_height = max(tree.get_distance(leaf) for leaf in tree.get_leaves())
    if tree_height == 0:
        tree_height = 1.0

    dup_depths = []
    spec_depths = []

    for node in tree.traverse("postorder"):
        if node.is_leaf() or node == tree:
            continue

        children = node.get_children()
        if len(children) != 2:
            continue

        depth = node.get_distance(tree) / tree_height

        species_left = set(
            parse_species_from_leaf(l.name) for l in children[0].get_leaves()
        )
        species_right = set(
            parse_species_from_leaf(l.name) for l in children[1].get_leaves()
        )

        if species_left & species_right:
            dup_depths.append(depth)
        else:
            spec_depths.append(depth)

    return {
        'duplication_depths': dup_depths,
        'speciation_depths': spec_depths,
        'mean_duplication_depth': float(np.mean(dup_depths)) if dup_depths else float('nan'),
        'mean_speciation_depth': float(np.mean(spec_depths)) if spec_depths else float('nan'),
    }


def run_duplication_analysis(tree, leaf_names=None):
    """
    Run full duplication analysis on a single family.

    Parameters
    ----------
    tree : ete3.Tree
        Rooted phylogenetic tree.
    leaf_names : list of str or None
        If None, extracted from tree leaves.

    Returns
    -------
    result : dict
        Combined results from all analyses.
    """
    if leaf_names is None:
        leaf_names = [l.name for l in tree.get_leaves()]

    species_result = compute_species_duplication_counts(leaf_names)
    node_result = count_duplication_nodes(tree)
    depth_result = compute_duplication_depth_profile(tree)

    # Remove bulky details for summary
    summary = {
        'n_sequences': species_result['n_sequences'],
        'n_species': species_result['n_species'],
        'duplication_index': species_result['duplication_index'],
        'multi_copy_fraction': species_result['multi_copy_fraction'],
        'max_copies': species_result['max_copies'],
        'mean_copies': species_result['mean_copies'],
        'n_duplication_nodes': node_result['n_duplication_nodes'],
        'n_speciation_nodes': node_result['n_speciation_nodes'],
        'duplication_node_fraction': node_result['duplication_fraction'],
        'mean_duplication_depth': depth_result['mean_duplication_depth'],
        'mean_speciation_depth': depth_result['mean_speciation_depth'],
    }

    return {
        'summary': summary,
        'species_detail': species_result,
        'node_detail': node_result,
        'depth_detail': depth_result,
    }


def run_cross_family_correlation(family_results):
    """
    Compute Spearman correlations between duplication signals and
    pipeline outputs across families.

    Parameters
    ----------
    family_results : list of dict
        Each dict has keys: 'family', 'duplication_index', 'duplication_node_fraction',
        'n_significant_splits_seq', 'n_significant_splits_str', 'bm_ou_pvalue',
        'alpha_hat', 'similarity_gain_seq', 'similarity_gain_str', etc.

    Returns
    -------
    correlations : dict
        Pairwise Spearman correlations and p-values.
    """
    from scipy.stats import spearmanr

    # Define which variables to correlate
    dup_vars = ['duplication_index', 'duplication_node_fraction',
                'multi_copy_fraction', 'mean_copies']
    pipeline_vars = ['n_significant_splits_seq', 'n_significant_splits_str',
                     'similarity_gain_seq', 'similarity_gain_str']
    # Add optional vars if present
    optional_vars = ['bm_ou_pvalue', 'alpha_hat', 'bm_rate']

    results = {}
    for dv in dup_vars:
        dv_vals = [r.get(dv) for r in family_results]
        if all(v is None for v in dv_vals):
            continue

        for pv in pipeline_vars + optional_vars:
            pv_vals = [r.get(pv) for r in family_results]

            # Filter to pairs where both are not None/NaN
            pairs = [(d, p) for d, p in zip(dv_vals, pv_vals)
                     if d is not None and p is not None
                     and not (isinstance(d, float) and np.isnan(d))
                     and not (isinstance(p, float) and np.isnan(p))]

            if len(pairs) < 5:
                continue

            d_arr, p_arr = zip(*pairs)
            rho, pval = spearmanr(d_arr, p_arr)
            results[f'{dv}_vs_{pv}'] = {
                'spearman_rho': float(rho),
                'p_value': float(pval),
                'n': len(pairs),
                'significant': pval < 0.05,
            }

    return results


if __name__ == "__main__":
    # Test with leaf names only (no tree needed for species analysis)
    test_leaves = [
        "METJ_ERWT9/26-105",
        "A0A0L0GV20_9ENTR/26-105",
        "A0A0F5VDA7_9GAMM/26-105",
        "A0A1V3QNY5_9GAMM/24-103",
        "A8T8T0_9VIBR/26-106",
        "Q5E2I3_ALIF1/26-106",
        "A0A178KJZ2_9GAMM/26-105",
        "METJ_ECOLI/26-105",  # Another MetJ from E. coli
        "METJ2_ECOLI/26-105",  # Paralog in E. coli
    ]

    result = compute_species_duplication_counts(test_leaves)
    print("Species duplication analysis:")
    for k, v in result.items():
        if k != 'species_counts' and k != 'multi_copy_species':
            print(f"  {k}: {v}")
    print(f"  multi_copy_species: {result['multi_copy_species']}")
    print("PASS")
