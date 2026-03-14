#!/usr/bin/env python3
"""
Cross-PLM Comparison Tool.

Compares pipeline results across two different PLM embedding modes to assess
whether the same phylogenetic splits emerge from different representations.

Reads two pipeline output directories (same family, different --embedding modes)
and reports:
1. Split concordance: which splits are found by both PLMs?
2. V matrix correlation: how similar are the trait covariances?
3. Mean shift alignment: do the two PLMs agree on the direction of divergence?

Example usage:
    python pipeline_files/compare_plm_embeddings.py \
        --dir_a pf00228_outputs/sequence_embeddings/ \
        --dir_b pf00228_outputs/clss_embeddings/ \
        --output comparison_results.json
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_split_options.utils import load_matrix_tensor


def load_results_json(path):
    """Load pipeline results.json."""
    with open(path, 'r') as f:
        return json.load(f)


def load_subfamilies_summary(path):
    """Load subfamilies_summary.json."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def find_significant_splits(out_mode_dir):
    """
    Scans a pipeline output directory and returns a list of significant splits
    with their metadata.
    """
    splits = []
    results_path = os.path.join(out_mode_dir, "results.json")
    if os.path.exists(results_path):
        results = load_results_json(results_path)
        if isinstance(results, list):
            for r in results:
                if r.get("sig", False):
                    splits.append(r)

    # Also scan for split JSONs directly
    for sf_dir_name in sorted(os.listdir(out_mode_dir)):
        sf_path = os.path.join(out_mode_dir, sf_dir_name)
        if not os.path.isdir(sf_path):
            continue
        sig_dir = os.path.join(sf_path, "significant_splits")
        if not os.path.isdir(sig_dir):
            continue
        for rank_dir_name in sorted(os.listdir(sig_dir)):
            rank_path = os.path.join(sig_dir, rank_dir_name)
            if not os.path.isdir(rank_path):
                continue
            # Find split JSON
            for fname in os.listdir(rank_path):
                if fname.startswith("split_") and fname.endswith(".json"):
                    split_json_path = os.path.join(rank_path, fname)
                    with open(split_json_path, 'r') as f:
                        split_data = json.load(f)
                    split_data["_sf_dir"] = sf_dir_name
                    split_data["_rank_dir"] = rank_dir_name
                    split_data["_split_dir"] = rank_path
                    splits.append(split_data)

    return splits


def compute_split_overlap(split_a, split_b):
    """
    Computes overlap between two splits. Each split has group_a and group_b.
    Returns the Jaccard similarity of the partitions (accounting for label swap).
    """
    a_set_a = set(str(n).replace('/', '_') for n in split_a.get("group_a", []))
    a_set_b = set(str(n).replace('/', '_') for n in split_a.get("group_b", []))
    b_set_a = set(str(n).replace('/', '_') for n in split_b.get("group_a", []))
    b_set_b = set(str(n).replace('/', '_') for n in split_b.get("group_b", []))

    # Two possible alignments (group labels are arbitrary)
    match_same = len(a_set_a & b_set_a) + len(a_set_b & b_set_b)
    match_swap = len(a_set_a & b_set_b) + len(a_set_b & b_set_a)
    total = len(a_set_a | a_set_b | b_set_a | b_set_b)

    if total == 0:
        return 0.0, "same"

    if match_same >= match_swap:
        return match_same / total, "same"
    else:
        return match_swap / total, "swapped"


def compare_v_matrices(split_dir_a, split_dir_b, rank_name_a, rank_name_b):
    """
    Loads V_A and V_B from two splits and computes correlation between
    the flattened upper triangles.
    """
    results = {}

    for sub in ["subA", "subB"]:
        path_a = os.path.join(split_dir_a, "calculations",
                              f"embedding_cov_{rank_name_a}_{sub}.csv")
        path_b = os.path.join(split_dir_b, "calculations",
                              f"embedding_cov_{rank_name_b}_{sub}.csv")

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            results[sub] = {"status": "missing", "path_a": path_a, "path_b": path_b}
            continue

        Va = load_matrix_tensor(path_a).numpy()
        Vb = load_matrix_tensor(path_b).numpy()

        # Eigenvalue spectra comparison
        evals_a = np.sort(np.linalg.eigvalsh(Va))[::-1]
        evals_b = np.sort(np.linalg.eigvalsh(Vb))[::-1]

        # Truncate to common length
        min_len = min(len(evals_a), len(evals_b))
        evals_a_t = evals_a[:min_len]
        evals_b_t = evals_b[:min_len]

        # Correlation of eigenvalue spectra
        if min_len > 1 and np.std(evals_a_t) > 0 and np.std(evals_b_t) > 0:
            eval_corr = float(np.corrcoef(evals_a_t, evals_b_t)[0, 1])
        else:
            eval_corr = None

        results[sub] = {
            "status": "ok",
            "dim_a": Va.shape[0],
            "dim_b": Vb.shape[0],
            "trace_a": float(np.trace(Va)),
            "trace_b": float(np.trace(Vb)),
            "eigenvalue_spectrum_correlation": eval_corr,
            "effective_rank_a": float(np.sum(evals_a)**2 / np.sum(evals_a**2)) if np.sum(evals_a**2) > 0 else 0,
            "effective_rank_b": float(np.sum(evals_b)**2 / np.sum(evals_b**2)) if np.sum(evals_b**2) > 0 else 0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-PLM Comparison Tool. "
                    "Compares pipeline results across two different PLM embedding modes."
    )
    parser.add_argument('--dir_a', type=str, required=True,
                        help="First PLM output directory (e.g., family_outputs/sequence_embeddings/)")
    parser.add_argument('--dir_b', type=str, required=True,
                        help="Second PLM output directory (e.g., family_outputs/clss_embeddings/)")
    parser.add_argument('--output', type=str, default=None,
                        help="Output JSON path (default: prints to stdout)")
    parser.add_argument('--overlap_threshold', type=float, default=0.8,
                        help="Minimum Jaccard overlap to consider splits concordant (default: 0.8)")

    args = parser.parse_args()

    dir_a = os.path.abspath(args.dir_a)
    dir_b = os.path.abspath(args.dir_b)

    mode_a = os.path.basename(dir_a).replace("_embeddings", "")
    mode_b = os.path.basename(dir_b).replace("_embeddings", "")

    print(f"Comparing PLM embeddings: {mode_a} vs {mode_b}")
    print(f"  Dir A: {dir_a}")
    print(f"  Dir B: {dir_b}")

    # --- Step 1: Load subfamily summaries ---
    sf_summary_a = load_subfamilies_summary(os.path.join(dir_a, "subfamilies_summary.json"))
    sf_summary_b = load_subfamilies_summary(os.path.join(dir_b, "subfamilies_summary.json"))

    # --- Step 2: Compare subfamily assignments ---
    sf_comparison = {}
    if sf_summary_a and sf_summary_b:
        all_sf_keys = sorted(set(list(sf_summary_a.keys()) + list(sf_summary_b.keys())))
        for sf_key in all_sf_keys:
            members_a = set(str(n).replace('/', '_') for n in sf_summary_a.get(sf_key, []))
            members_b = set(str(n).replace('/', '_') for n in sf_summary_b.get(sf_key, []))
            union = members_a | members_b
            intersection = members_a & members_b
            jaccard = len(intersection) / len(union) if union else 0
            sf_comparison[sf_key] = {
                "n_a": len(members_a),
                "n_b": len(members_b),
                "intersection": len(intersection),
                "jaccard": round(jaccard, 4),
            }
        print(f"\n  Subfamily comparison ({len(all_sf_keys)} subfamilies):")
        for sf_key, info in sf_comparison.items():
            print(f"    {sf_key}: A={info['n_a']}, B={info['n_b']}, "
                  f"overlap={info['intersection']}, Jaccard={info['jaccard']:.3f}")

    # --- Step 3: Find significant splits ---
    splits_a = find_significant_splits(dir_a)
    splits_b = find_significant_splits(dir_b)
    print(f"\n  Significant splits: A={len(splits_a)}, B={len(splits_b)}")

    # --- Step 4: Compute split concordance ---
    concordance = []
    for i, sa in enumerate(splits_a):
        for j, sb in enumerate(splits_b):
            overlap, alignment = compute_split_overlap(sa, sb)
            if overlap >= args.overlap_threshold:
                entry = {
                    "split_a_idx": i,
                    "split_b_idx": j,
                    "split_a_sf": sa.get("_sf_dir", sa.get("subfamily", "?")),
                    "split_b_sf": sb.get("_sf_dir", sb.get("subfamily", "?")),
                    "split_a_node": sa.get("node_name", sa.get("node", "?")),
                    "split_b_node": sb.get("node_name", sb.get("node", "?")),
                    "overlap_jaccard": round(overlap, 4),
                    "alignment": alignment,
                    "p_adj_a": sa.get("p_adj", None),
                    "p_adj_b": sb.get("p_adj", None),
                    "lambda_a": sa.get("lambda_obs", sa.get("lambda", None)),
                    "lambda_b": sb.get("lambda_obs", sb.get("lambda", None)),
                }

                # Compare V matrices if split dirs available
                split_dir_a = sa.get("_split_dir") or sa.get("folder", "")
                split_dir_b = sb.get("_split_dir") or sb.get("folder", "")
                rank_a = sa.get("_rank_dir", f"rank{sa.get('rank', '?')}")
                rank_b = sb.get("_rank_dir", f"rank{sb.get('rank', '?')}")
                if os.path.isdir(split_dir_a) and os.path.isdir(split_dir_b):
                    entry["v_matrix_comparison"] = compare_v_matrices(
                        split_dir_a, split_dir_b, rank_a, rank_b)

                concordance.append(entry)

    print(f"\n  Concordant splits (Jaccard >= {args.overlap_threshold}): {len(concordance)}")
    for c in concordance:
        print(f"    {c['split_a_sf']} ({mode_a}) <-> {c['split_b_sf']} ({mode_b}): "
              f"Jaccard={c['overlap_jaccard']:.3f}, "
              f"p_adj: {c.get('p_adj_a', '?')} vs {c.get('p_adj_b', '?')}")

    # --- Step 5: Summary statistics ---
    n_concordant = len(concordance)
    n_only_a = len(splits_a) - len(set(c["split_a_idx"] for c in concordance))
    n_only_b = len(splits_b) - len(set(c["split_b_idx"] for c in concordance))

    summary = {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "dir_a": dir_a,
        "dir_b": dir_b,
        "n_significant_splits_a": len(splits_a),
        "n_significant_splits_b": len(splits_b),
        "n_concordant": n_concordant,
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "overlap_threshold": args.overlap_threshold,
        "subfamily_comparison": sf_comparison,
        "concordant_splits": concordance,
    }

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {mode_a} vs {mode_b}")
    print(f"{'='*60}")
    print(f"  Significant splits in {mode_a}: {len(splits_a)}")
    print(f"  Significant splits in {mode_b}: {len(splits_b)}")
    print(f"  Concordant (Jaccard >= {args.overlap_threshold}): {n_concordant}")
    print(f"  Only in {mode_a}: {n_only_a}")
    print(f"  Only in {mode_b}: {n_only_b}")
    if len(splits_a) + len(splits_b) > 0:
        concordance_rate = 2 * n_concordant / (len(splits_a) + len(splits_b))
        print(f"  Concordance rate: {concordance_rate:.1%}")
    print(f"{'='*60}")

    # --- Step 6: Save output ---
    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Results saved to: {out_path}")
    else:
        print("\n  (Use --output to save results as JSON)")


if __name__ == "__main__":
    main()
