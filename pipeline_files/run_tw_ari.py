"""
CLI tool for computing tree-weighted ARI between two pipeline output directories
(e.g., sequence_embeddings vs structure_embeddings for the same family).

Usage:
    python pipeline_files/run_tw_ari.py \
        --dir_a PF01340_outputs/sequence_embeddings/ \
        --dir_b PF01340_outputs/structure_embeddings/ \
        --tree PF01340_calculations/PF01340.tree \
        --tau 0.1 \
        --output tw_ari_PF01340.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ete3 import Tree
from evaluate_split_options.tree_weighted_ari import compute_tw_ari


def load_partition_from_summary(summary_path):
    """
    Load Phase 1 partition from a subfamilies_summary.json file.

    Returns a list of sets of leaf names.
    """
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    partition = []
    for key, members in summary.items():
        if key.startswith('subfamily_') and isinstance(members, list):
            partition.append(set(str(m) for m in members))

    return partition


def main():
    parser = argparse.ArgumentParser(
        description="Compute tree-weighted ARI between two pipeline outputs")

    parser.add_argument('--dir_a', type=str, required=True,
                        help="First pipeline output directory (e.g., sequence_embeddings/)")
    parser.add_argument('--dir_b', type=str, required=True,
                        help="Second pipeline output directory (e.g., structure_embeddings/)")
    parser.add_argument('--tree', type=str, required=True,
                        help="Path to Newick tree file")
    parser.add_argument('--tau', type=float, default=0.1,
                        help="Threshold for eligible edges (default: 0.1)")
    parser.add_argument('--output', type=str, default=None,
                        help="Output JSON path (default: stdout)")

    args = parser.parse_args()

    # Load tree
    print(f"Loading tree from {args.tree}")
    tree = Tree(args.tree)
    # Midpoint rooting (matching pipeline convention)
    outgroup = tree.get_midpoint_outgroup()
    if outgroup:
        tree.set_outgroup(outgroup)

    print(f"  {len(tree.get_leaves())} leaves, "
          f"{len([n for n in tree.traverse() if not n.is_leaf() and n != tree])} internal nodes")

    # Load partitions
    summary_a = os.path.join(args.dir_a, 'subfamilies_summary.json')
    summary_b = os.path.join(args.dir_b, 'subfamilies_summary.json')

    if not os.path.exists(summary_a):
        print(f"Error: {summary_a} not found")
        sys.exit(1)
    if not os.path.exists(summary_b):
        print(f"Error: {summary_b} not found")
        sys.exit(1)

    print(f"Loading partition A from {summary_a}")
    partition_a = load_partition_from_summary(summary_a)
    print(f"  {len(partition_a)} groups, "
          f"{sum(len(g) for g in partition_a)} total leaves")

    print(f"Loading partition B from {summary_b}")
    partition_b = load_partition_from_summary(summary_b)
    print(f"  {len(partition_b)} groups, "
          f"{sum(len(g) for g in partition_b)} total leaves")

    # Compute tw-ARI
    print(f"\nComputing tw-ARI (tau={args.tau})...")
    result = compute_tw_ari(partition_a, partition_b, tree, tau=args.tau)

    # Add metadata
    result['dir_a'] = args.dir_a
    result['dir_b'] = args.dir_b
    result['tree_path'] = args.tree
    result['tau'] = args.tau

    # Print summary
    print(f"\n{'='*60}")
    print(f"  tw-ARI:        {result['tw_ari']:.4f}")
    print(f"  Standard ARI:  {result['standard_ari']:.4f}")
    print(f"  WRI observed:  {result['wri_observed']:.4f}")
    print(f"  WRI expected:  {result['wri_expected']:.4f}")
    print(f"  Eligible edges: {result['n_eligible_edges']}")
    print(f"  Informative pairs: {result['n_informative_pairs']} / {result['n_total_pairs']}")
    print(f"  k1={result['k1']}, k2={result['k2']}, n={result['n_leaves']}")
    print(f"{'='*60}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
