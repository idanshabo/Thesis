#!/usr/bin/env python3
"""
BM vs OU Model Comparison Tool.

Standalone CLI tool that compares Brownian Motion (BM) and Ornstein-Uhlenbeck (OU)
phylogenetic models on protein language model embeddings.

Uses outputs already produced by the main pipeline (covariance matrix CSV,
embeddings .pt file). Does not modify any existing pipeline files.

Example usage:
    python pipeline_files/compare_bm_ou.py \
        --cov pf00228_calculations/pf00228_cov_mat_tree_ordered.csv \
        --embeddings pf00228_calculations/embeddings_sequence/normalized_mean_embeddings_matrix.pt \
        --pca_components 40 --plot
"""

import os
import sys
import argparse
import json
import math
import torch
import numpy as np

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.decomposition import PCA
from evaluate_split_options.ou_model import compare_bm_ou, bm_to_ou_covariance, profile_log_likelihood
from evaluate_split_options.utils import load_matrix_tensor
from utils.save_results_json import save_results_json


def load_embeddings(path):
    """
    Loads embeddings from a .pt file. Handles the two formats used by the pipeline:
    - Dict with 'embeddings' key (from create_normalized_mean_embeddings_matrix)
    - Raw tensor (from aligned_global_embeddings.pt)
    """
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict):
        if 'embeddings' in data:
            return data['embeddings'].float()
        elif 'data' in data:
            return data['data'].float()
        else:
            raise ValueError(f"Embeddings file {path} is a dict but has no 'embeddings' or 'data' key. "
                             f"Keys found: {list(data.keys())}")
    elif isinstance(data, torch.Tensor):
        return data.float()
    else:
        raise ValueError(f"Unknown embeddings format in {path}: {type(data)}")


def print_summary(result, cov_name, p_raw, pca_k, pca_var_explained):
    """Prints a formatted summary table to stdout."""
    print()
    print("=" * 60)
    print("  BM vs OU Model Comparison")
    print("=" * 60)
    print(f"  Input: {cov_name}")
    print(f"  Taxa (n): {result['n']}  |  Traits raw: {p_raw}  |  PCA (p): {pca_k}")
    print(f"  PCA variance explained: {pca_var_explained * 100:.1f}%")
    print("-" * 60)
    print(f"  BM   Log-Likelihood:  {result['ll_bm']:.2f}")
    print(f"  OU   Log-Likelihood:  {result['ll_ou']:.2f}   (alpha_hat = {result['alpha_hat']:.4f})")
    print("-" * 60)
    print(f"  LRT statistic:    {result['lrt_statistic']:.2f}   "
          f"(df={result['lrt_df']},  p = {result['lrt_pvalue']:.2e})")
    print(f"  AIC  BM: {result['aic_bm']:.2f}   |   OU: {result['aic_ou']:.2f}   "
          f"-->  {result['preferred_model_aic']} preferred")
    print(f"  BIC  BM: {result['bic_bm']:.2f}   |   OU: {result['bic_ou']:.2f}   "
          f"-->  {result['preferred_model_bic']} preferred")
    print("=" * 60)


def generate_plots(X, U_BM, result, output_dir):
    """Generates diagnostic plots: alpha profile and eigenvalue comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    alpha_hat = result["alpha_hat"]

    # --- Plot 1: Alpha Profile ---
    alphas = np.logspace(-4, np.log10(50), 100)
    lls = []
    for a in alphas:
        try:
            U_OU = bm_to_ou_covariance(U_BM, a)
            ll = profile_log_likelihood(X, U_OU)
            lls.append(ll)
        except Exception:
            lls.append(float('nan'))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(np.log10(alphas), lls, 'b-', linewidth=1.5, label='OU profile LL')
    ax.axhline(y=result["ll_bm"], color='r', linestyle='--', linewidth=1.2, label='BM LL')
    ax.axvline(x=np.log10(alpha_hat), color='g', linestyle=':', linewidth=1.2,
               label=f'alpha_hat = {alpha_hat:.4f}')
    ax.set_xlabel('log10(alpha)')
    ax.set_ylabel('Profile Log-Likelihood')
    ax.set_title('Profile Log-Likelihood: BM vs OU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'bm_ou_alpha_profile.png'), dpi=150)
    plt.close(fig)

    # --- Plot 2: Eigenvalue Comparison ---
    U_OU = bm_to_ou_covariance(U_BM, alpha_hat)
    eig_bm = torch.linalg.eigvalsh(U_BM.double()).numpy()[::-1]
    eig_ou = torch.linalg.eigvalsh(U_OU).numpy()[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, len(eig_bm) + 1), eig_bm, 'r-o', markersize=3, label='BM eigenvalues')
    ax.plot(range(1, len(eig_ou) + 1), eig_ou, 'b-o', markersize=3, label='OU eigenvalues')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Covariance Eigenvalue Spectrum (alpha_hat = {alpha_hat:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'bm_ou_eigenvalue_comparison.png'), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="BM vs OU Model Comparison Tool. "
                    "Compares Brownian Motion and Ornstein-Uhlenbeck models for "
                    "phylogenetic trait evolution on protein language model embeddings."
    )

    parser.add_argument('--cov', type=str, required=True,
                        help="Path to the BM covariance matrix CSV "
                             "(e.g., pf00228_cov_mat_tree_ordered.csv)")
    parser.add_argument('--embeddings', type=str, required=True,
                        help="Path to the embeddings .pt file "
                             "(e.g., normalized_mean_embeddings_matrix.pt)")
    parser.add_argument('--pca_components', type=int, default=None,
                        help="Number of standard PCA components. Default: min(n-2, 50)")
    parser.add_argument('--alpha_min', type=float, default=1e-4,
                        help="Lower bound for OU alpha search (default: 1e-4)")
    parser.add_argument('--alpha_max', type=float, default=50.0,
                        help="Upper bound for OU alpha search (default: 50.0)")
    parser.add_argument('--output', type=str, default=None,
                        help="Path for the output JSON file. "
                             "Default: <cov_dir>/bm_ou_comparison.json")
    parser.add_argument('--plot', action='store_true',
                        help="Generate diagnostic plots (alpha profile, eigenvalue comparison)")

    args = parser.parse_args()

    # --- Step 1: Load Inputs ---
    print("Loading inputs...")
    U_BM = load_matrix_tensor(args.cov).double()
    n = U_BM.shape[0]

    X_raw = load_embeddings(args.embeddings)
    p_raw = X_raw.shape[1]

    # Validate dimensions
    if X_raw.shape[0] != n:
        print(f"  [WARNING] Embeddings have {X_raw.shape[0]} rows but covariance has {n}. "
              f"Using first {n} rows of embeddings.")
        X_raw = X_raw[:n]

    if n < 5:
        print(f"  [ERROR] Too few taxa (n={n}) for meaningful model comparison. Need at least 5.")
        sys.exit(1)

    if n < 20:
        print(f"  [WARNING] Small sample size (n={n}). Results may be unreliable.")

    # --- Step 2: Standard PCA ---
    if args.pca_components is not None:
        k = min(args.pca_components, n - 2, p_raw)
    else:
        k = min(n - 2, 50, p_raw)

    print(f"Applying standard PCA: {p_raw} -> {k} components...")
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X_raw.numpy())
    pca_var_explained = float(pca.explained_variance_ratio_.sum())
    X = torch.from_numpy(X_pca).double()
    print(f"  Variance explained: {pca_var_explained * 100:.1f}%")

    # --- Step 3: Run Comparison ---
    print("Running BM vs OU comparison...")
    result = compare_bm_ou(X, U_BM, alpha_bounds=(args.alpha_min, args.alpha_max))

    # --- Step 4: Augment with Metadata ---
    result["p_raw"] = p_raw
    result["pca_components_used"] = k
    result["pca_variance_explained"] = pca_var_explained
    result["input_cov_path"] = os.path.abspath(args.cov)
    result["input_embeddings_path"] = os.path.abspath(args.embeddings)

    # --- Step 5: Save JSON ---
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.cov)), "bm_ou_comparison.json")
    save_results_json(result, output_path)

    # --- Step 6: Print Summary ---
    cov_name = os.path.basename(args.cov)
    print_summary(result, cov_name, p_raw, k, pca_var_explained)
    print(f"  Results saved to: {output_path}")

    # --- Step 7: Optional Plots ---
    if args.plot:
        print("Generating diagnostic plots...")
        output_dir = os.path.dirname(output_path)
        generate_plots(X, U_BM, result, output_dir)

    print("=" * 60)


if __name__ == "__main__":
    main()
