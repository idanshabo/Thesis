"""
CLI tool for computing pairwise residue couplings and differential
couplings between phylogenetic clades.

This is a phylogenetically-corrected analogue of DCA/PSICOV that operates
in PLM embedding space, with the added capability of comparing coupling
patterns between two clades.

Outputs:
  - L x L position-pair coupling matrix
  - Top coupled position pairs
  - 20 x 20 AA-pair coupling matrices for top position pairs (DCA-style)
  - Region-level coupling (helix-helix, sheet-sheet, etc.) using DSSP/Q8
  - Spectral clusters of positions (data-driven module discovery)
  - For differential mode: same outputs but for the difference C^A - C^B

Usage examples:

Single-group analysis:
    python pipeline_files/run_pairwise_couplings.py \\
        --mode single \\
        --per_residue_dir CALCULATIONS/embeddings_sequence/per_residue/ \\
        --msa CALCULATIONS/family.fasta \\
        --tree CALCULATIONS/family.tree \\
        --predicted_dir CALCULATIONS/predicted_structures/ \\
        --output couplings_single.json

Differential analysis (between two clades):
    python pipeline_files/run_pairwise_couplings.py \\
        --mode differential \\
        --per_residue_dir CALCULATIONS/embeddings_sequence/per_residue/ \\
        --msa CALCULATIONS/family.fasta \\
        --tree CALCULATIONS/family.tree \\
        --split_json OUTPUTS/sequence_embeddings/subfamily_1/significant_splits/rank1/split_rank1.json \\
        --predicted_dir CALCULATIONS/predicted_structures/ \\
        --output couplings_differential.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ete3 import Tree

from utils.phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix
from evaluate_split_options.pairwise_couplings import (
    run_pairwise_coupling_analysis,
    run_differential_coupling_analysis,
    get_aa_embeddings_from_data,
    load_per_residue_embeddings,
    get_consensus_ss_from_pipeline,
)


def load_msa_fasta(fasta_path):
    """Load aligned MSA in FASTA format."""
    sequences = {}
    name = None
    seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    sequences[name] = "".join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name:
            sequences[name] = "".join(seq)
    return sequences


def compute_u_inv_from_tree(tree_path, sequence_names):
    """Compute U^{-1} for the given sequence names from the phylogenetic tree."""
    tree = Tree(tree_path, format=1)
    outgroup = tree.get_midpoint_outgroup()
    if outgroup:
        tree.set_outgroup(outgroup)

    U, leaf_names = tree_to_covariance_matrix(tree)
    U = np.asarray(U, dtype=np.float64)

    # Reorder to match sequence_names
    name_to_idx = {n: i for i, n in enumerate(leaf_names)}
    indices = [name_to_idx[n] for n in sequence_names if n in name_to_idx]
    U_sub = U[np.ix_(indices, indices)]
    # Add small jitter for numerical stability
    U_sub += 1e-6 * np.eye(U_sub.shape[0])
    U_inv = np.linalg.inv(U_sub)

    valid_names = [n for n in sequence_names if n in name_to_idx]
    return U_inv, valid_names


def main():
    parser = argparse.ArgumentParser(
        description="Phylogenetically-corrected residue-residue couplings"
    )

    parser.add_argument('--mode', choices=['single', 'differential'],
                        required=True,
                        help="single: analyze one group; differential: compare two groups")
    parser.add_argument('--per_residue_dir', required=True,
                        help="Directory with per-residue embedding .pt files")
    parser.add_argument('--msa', required=True,
                        help="Aligned MSA in FASTA format")
    parser.add_argument('--tree', required=True,
                        help="Phylogenetic tree in Newick format")

    # Single mode
    parser.add_argument('--sequence_names', nargs='+', default=None,
                        help="(single mode) sequence names to analyze; default: all in MSA")

    # Differential mode
    parser.add_argument('--split_json', default=None,
                        help="(differential mode) path to split_rankN.json with group_a/group_b")

    # Optional structural annotation
    parser.add_argument('--predicted_dir', default=None,
                        help="Directory with ESMFold-predicted .pdb files for region aggregation")

    # AA embeddings
    parser.add_argument('--aa_embeddings_npy', default=None,
                        help="Path to cached AA embeddings .npy. If missing, will compute from data.")

    # Tunables
    parser.add_argument('--n_top', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--return_full', action='store_true',
                        help="Compute the full L x L x p x p tensor (memory intensive)")

    parser.add_argument('--output', required=True, help="Output JSON path")

    args = parser.parse_args()

    print(f"Loading MSA from {args.msa}")
    msa_sequences = load_msa_fasta(args.msa)
    print(f"  {len(msa_sequences)} sequences in MSA")

    # Determine the sequence names to use
    if args.mode == 'single':
        if args.sequence_names is None:
            seq_names = list(msa_sequences.keys())
        else:
            seq_names = args.sequence_names
        print(f"  Analyzing {len(seq_names)} sequences")

        print(f"Loading tree from {args.tree}")
        U_inv, valid_names = compute_u_inv_from_tree(args.tree, seq_names)
        print(f"  U_inv shape: {U_inv.shape}")

        # AA embeddings
        aa_emb = None
        if args.aa_embeddings_npy and os.path.exists(args.aa_embeddings_npy):
            aa_emb = np.load(args.aa_embeddings_npy)
        else:
            print("Computing AA embeddings from data (averaging)...")
            per_res = load_per_residue_embeddings(args.per_residue_dir,
                                                   valid_names)
            if per_res:
                aa_emb = get_aa_embeddings_from_data(per_res, msa_sequences,
                                                      valid_names)
                if args.aa_embeddings_npy:
                    np.save(args.aa_embeddings_npy, aa_emb)

        # Consensus secondary structure
        consensus_ss = None
        if args.predicted_dir:
            print(f"Computing consensus SS from {args.predicted_dir}")
            try:
                aligned_seqs = [msa_sequences[n] for n in valid_names if n in msa_sequences]
                seq_ids = [n for n in valid_names if n in msa_sequences]
                consensus_ss, n_used = get_consensus_ss_from_pipeline(
                    aligned_seqs, seq_ids, args.predicted_dir
                )
                print(f"  Used {n_used} structures, "
                      f"consensus length: {len(consensus_ss)}")
            except Exception as e:
                print(f"  Failed to compute consensus SS: {e}")

        print("Running pairwise coupling analysis...")
        result = run_pairwise_coupling_analysis(
            per_residue_dir=args.per_residue_dir,
            sequence_names=valid_names,
            msa_sequences=msa_sequences,
            U_inv=U_inv,
            aa_embeddings=aa_emb,
            consensus_ss=consensus_ss,
            n_top=args.n_top,
            n_clusters=args.n_clusters,
            return_full=args.return_full or (aa_emb is not None),
        )

        result['mode'] = 'single'

    else:  # differential mode
        if not args.split_json:
            print("Error: --split_json required for differential mode")
            sys.exit(1)

        print(f"Loading split from {args.split_json}")
        with open(args.split_json) as f:
            split = json.load(f)

        group_a = split.get('group_a', [])
        group_b = split.get('group_b', [])
        print(f"  Group A: {len(group_a)} sequences")
        print(f"  Group B: {len(group_b)} sequences")

        print(f"Loading tree from {args.tree}")
        U_inv_a, valid_a = compute_u_inv_from_tree(args.tree, group_a)
        U_inv_b, valid_b = compute_u_inv_from_tree(args.tree, group_b)

        # AA embeddings
        aa_emb = None
        if args.aa_embeddings_npy and os.path.exists(args.aa_embeddings_npy):
            aa_emb = np.load(args.aa_embeddings_npy)
        else:
            print("Computing AA embeddings from data (averaging)...")
            all_names = list(set(valid_a) | set(valid_b))
            per_res = load_per_residue_embeddings(args.per_residue_dir,
                                                   all_names)
            if per_res:
                aa_emb = get_aa_embeddings_from_data(per_res, msa_sequences,
                                                      all_names)
                if args.aa_embeddings_npy:
                    np.save(args.aa_embeddings_npy, aa_emb)

        # Consensus SS for both groups
        consensus_ss_a = consensus_ss_b = None
        if args.predicted_dir:
            print("Computing consensus SS for both groups...")
            try:
                aligned_a = [msa_sequences[n] for n in valid_a if n in msa_sequences]
                aligned_b = [msa_sequences[n] for n in valid_b if n in msa_sequences]
                consensus_ss_a, _ = get_consensus_ss_from_pipeline(
                    aligned_a, valid_a, args.predicted_dir
                )
                consensus_ss_b, _ = get_consensus_ss_from_pipeline(
                    aligned_b, valid_b, args.predicted_dir
                )
            except Exception as e:
                print(f"  Failed to compute consensus SS: {e}")

        print("Running differential coupling analysis...")
        result = run_differential_coupling_analysis(
            per_residue_dir=args.per_residue_dir,
            group_a_names=valid_a,
            group_b_names=valid_b,
            msa_sequences=msa_sequences,
            U_inv_a=U_inv_a,
            U_inv_b=U_inv_b,
            aa_embeddings=aa_emb,
            consensus_ss_a=consensus_ss_a,
            consensus_ss_b=consensus_ss_b,
            n_top=args.n_top,
            n_clusters=args.n_clusters,
            return_full=args.return_full or (aa_emb is not None),
        )

        result['mode'] = 'differential'
        result['split_json'] = args.split_json

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")

    # Print top results
    if 'top_pairs' in result:
        print(f"\nTop {min(10, len(result['top_pairs']))} coupled position pairs:")
        for pair in result['top_pairs'][:10]:
            print(f"  ({pair['i']}, {pair['j']}): {pair['coupling']:.4f}")

    if 'top_differential_pairs' in result:
        print(f"\nTop {min(10, len(result['top_differential_pairs']))} differentially-coupled pairs:")
        for pair in result['top_differential_pairs'][:10]:
            print(f"  ({pair['i']}, {pair['j']}): {pair['coupling']:.4f}")


if __name__ == "__main__":
    main()
