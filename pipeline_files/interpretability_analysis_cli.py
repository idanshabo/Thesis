#!/usr/bin/env python3
"""
Interpretability Analysis Tool.

Standalone CLI tool that analyzes the meaning of phylogenetic split results:
1. Mean shift direction between groups A and B
2. Covariance eigendecomposition and comparison
3. Projection from PCA space back to original embedding coordinates

Reads from existing pipeline output directories. Does not modify any existing files.

Example usage:
    python pipeline_files/interpretability_analysis_cli.py \
        --split_dir pf00228_outputs/sequence_embeddings/subfamily_1/significant_splits/rank1/ \
        --embeddings pf00228_calculations/embeddings_sequence/normalized_mean_embeddings_matrix.pt \
        --pca_components 100 --pca_variance 0.99 --plot
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_split_options.interpretability_analysis import run_full_interpretability
from evaluate_split_options.utils import load_matrix_tensor
from utils.save_results_json import save_results_json


def load_embeddings(path):
    """Loads embeddings from a .pt file (same as compare_bm_ou.py)."""
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict):
        if 'embeddings' in data:
            return data['embeddings'].float(), data.get('file_names') or data.get('names') or data.get('ids')
        elif 'data' in data:
            return data['data'].float(), data.get('file_names') or data.get('names') or data.get('ids')
        else:
            raise ValueError(f"Embeddings file {path} has no 'embeddings' or 'data' key. "
                             f"Keys: {list(data.keys())}")
    elif isinstance(data, torch.Tensor):
        return data.float(), None
    else:
        raise ValueError(f"Unknown embeddings format in {path}: {type(data)}")


def auto_detect_paths(split_dir):
    """
    Auto-detects file paths from the split directory structure.

    Expected structure:
        .../family_outputs/mode_embeddings/subfamily_N/significant_splits/rankM/
    """
    split_dir = os.path.abspath(split_dir)
    rank_name = os.path.basename(split_dir)                          # rankM
    sig_splits_dir = os.path.dirname(split_dir)                      # significant_splits/
    subfamily_dir = os.path.dirname(sig_splits_dir)                  # subfamily_N/
    output_mode_dir = os.path.dirname(subfamily_dir)                 # mode_embeddings/
    outputs_dir = os.path.dirname(output_mode_dir)                   # family_outputs/
    base_dir = os.path.dirname(outputs_dir)                          # parent

    sf_name = os.path.basename(subfamily_dir)                        # subfamily_N
    family = os.path.basename(outputs_dir).replace("_outputs", "")   # family name
    calc_dir = os.path.join(base_dir, f"{family}_calculations")

    ppca_dir = os.path.join(calc_dir, sf_name, "ppca_loadings")

    paths = {
        "rank_name": rank_name,
        "sf_name": sf_name,
        "family": family,
        "calc_dir": calc_dir,
        "split_json": os.path.join(split_dir, f"split_{rank_name}.json"),
        "V_A": os.path.join(split_dir, "calculations", f"embedding_cov_{rank_name}_subA.csv"),
        "V_B": os.path.join(split_dir, "calculations", f"embedding_cov_{rank_name}_subB.csv"),
        "V_pooled": os.path.join(calc_dir, sf_name, f"{sf_name}_global_H0_PCA_cov_mat.csv"),
        "U_local": os.path.join(calc_dir, sf_name, f"{sf_name}_cov_mat.csv"),
        "U_global": os.path.join(calc_dir, f"{family}_cov_mat_tree_ordered.csv"),
        "sf_summary": os.path.join(output_mode_dir, "subfamilies_summary.json"),
        "ppca_dir": ppca_dir,
    }
    return paths


def print_summary(result, metadata):
    """Prints a formatted summary table to stdout."""
    ms = result["analysis_1_mean_shift"]
    cv = result["analysis_2_covariance_decomposition"]
    pj = result["analysis_3_projection_to_original"]

    print()
    print("=" * 60)
    print("  Interpretability Analysis")
    print("=" * 60)
    print(f"  Split: {metadata['sf_name']} / {metadata['rank_name']} "
          f"(node: {metadata.get('node_name', '?')})")
    print(f"  Groups: A (n={metadata['n_A']})  vs  B (n={metadata['n_B']})")
    print(f"  PCA dims: {pj['ppca_n_components_recovered']} (from {pj['p_original']} original)")
    print("-" * 60)
    print("  MEAN SHIFT ANALYSIS")
    print(f"    ||delta_mu||_pca = {ms['delta_mu_magnitude_pca']:.4f}")

    top_dims = ms["top_contributing_dims_pca"][:5]
    top_vals = ms["top_contributing_values_pca"][:5]
    dim_strs = [f"[{d}] {v*100:.1f}%" for d, v in zip(top_dims, top_vals)]
    print(f"    Top PCA dims:  {', '.join(dim_strs)}")

    if "delta_mu_magnitude_original" in ms:
        print(f"    ||delta_mu||_original = {ms['delta_mu_magnitude_original']:.4f}")
        top_o = ms["top_original_dims_by_mean_shift"][:5]
        top_ov = ms["top_original_dims_mean_shift_values"][:5]
        orig_strs = [f"[{d}] {v:.4f}" for d, v in zip(top_o, top_ov)]
        print(f"    Top original dims:  {', '.join(orig_strs)}")

    print("-" * 60)
    print("  COVARIANCE DECOMPOSITION")
    print(f"    ||V_A - V_B||_F = {cv['frobenius_A_B']:.4f}")
    print(f"    Effective rank: A={cv['effective_rank_A']:.1f}  B={cv['effective_rank_B']:.1f}")

    top_div = cv["top_divergent_dims"][:5]
    top_rat = cv["top_divergent_ratios"][:5]
    rat_strs = [f"[{d}] {r:.2f}" for d, r in zip(top_div, top_rat)]
    print(f"    Top divergent dims (ratio): {', '.join(rat_strs)}")

    align = cv["subspace_alignment_matrix"]
    n_show = min(5, len(align))
    diag_vals = [align[i][i] for i in range(n_show)]
    diag_strs = [f"{v:.2f}" for v in diag_vals]
    print(f"    Subspace alignment (top-{n_show} diagonal): {' '.join(diag_strs)}")

    if pj.get("top_V_diff_eigenvalues"):
        top_ve = pj["top_V_diff_eigenvalues"][:5]
        ve_strs = [f"{v:.3f}" for v in top_ve]
        print(f"    Top V_diff eigenvalues: {', '.join(ve_strs)}")

    print("=" * 60)


def generate_plots(result, output_dir):
    """Generates diagnostic plots for the interpretability analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ms = result["analysis_1_mean_shift"]
    cv = result["analysis_2_covariance_decomposition"]

    # --- Plot 1: Mean shift per PCA dimension ---
    fig, ax = plt.subplots(figsize=(10, 5))
    delta = np.array(ms["delta_mu_pca"])
    colors = ['tab:blue' if v >= 0 else 'tab:red' for v in delta]
    ax.bar(range(len(delta)), delta, color=colors, width=0.8)
    ax.set_xlabel('PCA Dimension')
    ax.set_ylabel('delta_mu')
    ax.set_title('Mean Shift per PCA Dimension (mu_A - mu_B)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'mean_shift_pca.png'), dpi=150)
    plt.close(fig)

    # --- Plot 2: Mean shift in original space (top dims) ---
    if "delta_mu_original" in ms:
        fig, ax = plt.subplots(figsize=(12, 5))
        delta_orig = np.array(ms["delta_mu_original"])
        top_idx = np.argsort(np.abs(delta_orig))[::-1][:50]
        vals = delta_orig[top_idx]
        colors = ['tab:blue' if v >= 0 else 'tab:red' for v in vals]
        ax.bar(range(len(vals)), vals, color=colors, width=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([str(i) for i in top_idx], rotation=90, fontsize=6)
        ax.set_xlabel('Original Embedding Dimension')
        ax.set_ylabel('delta_mu')
        ax.set_title('Mean Shift in Original Space (Top 50 Dimensions)')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'mean_shift_original.png'), dpi=150)
        plt.close(fig)

    # --- Plot 3: Eigenvalue spectrum comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    evals_A = np.array(cv["eigenvalues_A"])
    evals_B = np.array(cv["eigenvalues_B"])
    ax.plot(range(1, len(evals_A) + 1), np.maximum(evals_A, 1e-12), 'b-o',
            markersize=3, label='V_A eigenvalues')
    ax.plot(range(1, len(evals_B) + 1), np.maximum(evals_B, 1e-12), 'r-o',
            markersize=3, label='V_B eigenvalues')
    if "eigenvalues_pooled" in cv:
        evals_P = np.array(cv["eigenvalues_pooled"])
        ax.plot(range(1, len(evals_P) + 1), np.maximum(evals_P, 1e-12), 'k--o',
                markersize=3, label='V_pooled eigenvalues')
    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Covariance Eigenvalue Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'eigenvalue_spectrum.png'), dpi=150)
    plt.close(fig)

    # --- Plot 4: Eigenvalue ratio profile ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ratios = np.array(cv["eigenvalue_ratio_A_over_B"])
    # Clip extreme ratios for visualization
    ratios_clipped = np.clip(ratios, 0.01, 100)
    colors = ['tab:blue' if r >= 1 else 'tab:red' for r in ratios_clipped]
    ax.bar(range(len(ratios_clipped)), np.log2(ratios_clipped), color=colors, width=0.8)
    ax.axhline(y=0, color='black', linewidth=1.0)
    ax.set_xlabel('PCA Dimension')
    ax.set_ylabel('log2(lambda_A / lambda_B)')
    ax.set_title('Eigenvalue Ratio Profile (A / B)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'eigenvalue_ratio.png'), dpi=150)
    plt.close(fig)

    # --- Plot 5: Subspace alignment heatmap ---
    align = np.array(cv["subspace_alignment_matrix"])
    n_show = min(20, align.shape[0])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(align[:n_show, :n_show], cmap='viridis', vmin=0, vmax=1, aspect='equal')
    ax.set_xlabel('V_B Eigenvector Index')
    ax.set_ylabel('V_A Eigenvector Index')
    ax.set_title(f'Subspace Alignment |q_A^T q_B| (top {n_show})')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'subspace_alignment.png'), dpi=150)
    plt.close(fig)

    # --- Plot 6: Combined summary ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: eigenvalue spectra
    ax = axes[0, 0]
    ax.plot(range(1, len(evals_A) + 1), np.maximum(evals_A, 1e-12), 'b-', linewidth=1.2, label='V_A')
    ax.plot(range(1, len(evals_B) + 1), np.maximum(evals_B, 1e-12), 'r-', linewidth=1.2, label='V_B')
    ax.set_yscale('log')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Spectra')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: eigenvalue ratio
    ax = axes[0, 1]
    ax.bar(range(len(ratios_clipped)), np.log2(ratios_clipped),
           color=['tab:blue' if r >= 1 else 'tab:red' for r in ratios_clipped], width=0.8)
    ax.axhline(y=0, color='black', linewidth=1.0)
    ax.set_xlabel('PCA Dimension')
    ax.set_ylabel('log2(ratio)')
    ax.set_title('Eigenvalue Ratio (A/B)')
    ax.grid(True, alpha=0.3)

    # Bottom-left: mean shift PCA
    ax = axes[1, 0]
    ax.bar(range(len(delta)), delta,
           color=['tab:blue' if v >= 0 else 'tab:red' for v in delta], width=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('PCA Dimension')
    ax.set_ylabel('delta_mu')
    ax.set_title('Mean Shift (PCA space)')
    ax.grid(True, alpha=0.3)

    # Bottom-right: subspace alignment diagonal
    ax = axes[1, 1]
    diag = [align[i][i] for i in range(n_show)]
    ax.bar(range(n_show), diag, color='tab:green', width=0.8)
    ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Eigenvector Index')
    ax.set_ylabel('|q_A^T q_B|')
    ax.set_title('Eigenvector Alignment (diagonal)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'interpretability_summary.png'), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Interpretability Analysis Tool. "
                    "Analyzes mean shifts, covariance decomposition, and "
                    "projects results back to original embedding space."
    )

    parser.add_argument('--split_dir', type=str, required=True,
                        help="Path to a significant split directory")
    parser.add_argument('--embeddings', type=str, required=True,
                        help="Path to the original embeddings .pt file")
    parser.add_argument('--calc_dir', type=str, default=None,
                        help="Override auto-detected calculations directory")
    parser.add_argument('--cov', type=str, default=None,
                        help="Override global covariance matrix CSV path")
    parser.add_argument('--pca_components', type=int, default=100,
                        help="pPCA min components (must match original run, default: 100)")
    parser.add_argument('--pca_variance', type=float, default=0.99,
                        help="pPCA min variance (must match original run, default: 0.99)")
    parser.add_argument('--no_standardize', action='store_true',
                        help="Use covariance-mode pPCA instead of correlation-mode")
    parser.add_argument('--n_top', type=int, default=20,
                        help="Number of top dimensions to report (default: 20)")
    parser.add_argument('--output', type=str, default=None,
                        help="Path for output JSON file")
    parser.add_argument('--plot', action='store_true',
                        help="Generate diagnostic plots")

    args = parser.parse_args()
    standardize = not args.no_standardize

    # --- Step 1: Auto-detect paths ---
    print("Detecting paths...")
    detected = auto_detect_paths(args.split_dir)

    if args.calc_dir:
        detected["calc_dir"] = os.path.abspath(args.calc_dir)
        # Re-derive dependent paths
        sf_name = detected["sf_name"]
        family = detected["family"]
        detected["V_pooled"] = os.path.join(args.calc_dir, sf_name,
                                            f"{sf_name}_global_H0_PCA_cov_mat.csv")
        detected["U_local"] = os.path.join(args.calc_dir, sf_name,
                                           f"{sf_name}_cov_mat.csv")
        detected["U_global"] = os.path.join(args.calc_dir,
                                            f"{family}_cov_mat_tree_ordered.csv")

    if args.cov:
        detected["U_global"] = os.path.abspath(args.cov)

    # --- Step 2: Validate files ---
    required_files = {
        "Split JSON": detected["split_json"],
        "V_A CSV": detected["V_A"],
        "V_B CSV": detected["V_B"],
        "U_local CSV": detected["U_local"],
        "U_global CSV": detected["U_global"],
        "Embeddings": args.embeddings,
    }
    optional_files = {
        "V_pooled CSV": detected["V_pooled"],
        "Subfamilies summary": detected["sf_summary"],
    }

    print("Validating files...")
    missing = []
    for label, path in required_files.items():
        if not os.path.exists(path):
            missing.append(f"  {label}: {path}")
    if missing:
        print("[ERROR] Missing required files:")
        for m in missing:
            print(m)
        sys.exit(1)

    for label, path in optional_files.items():
        if not os.path.exists(path):
            print(f"  [WARNING] Optional file not found: {label}: {path}")

    # --- Step 3: Load split JSON ---
    print("Loading split information...")
    with open(detected["split_json"], 'r') as f:
        split_info = json.load(f)

    group_a_names = split_info.get("group_a", [])
    group_b_names = split_info.get("group_b", [])
    node_name = split_info.get("node_name", "?")
    lambda_obs = split_info.get("lambda_obs", None)
    p_adj = split_info.get("p_adj", None)

    # --- Step 4: Load subfamily membership ---
    sf_names = None
    if os.path.exists(detected["sf_summary"]):
        with open(detected["sf_summary"], 'r') as f:
            sf_summary = json.load(f)
        sf_names = sf_summary.get(detected["sf_name"], None)

    # --- Step 5: Load embeddings and global covariance ---
    print("Loading embeddings and covariance matrix...")
    emb_tensor, emb_names = load_embeddings(args.embeddings)
    U_global = load_matrix_tensor(detected["U_global"]).float()
    cov_df = pd.read_csv(detected["U_global"], index_col=0)
    global_names = list(cov_df.index.astype(str))

    # --- Step 6: Determine subfamily indices ---
    print("Determining subfamily indices...")
    if sf_names is not None:
        # Use subfamily summary to get indices
        sf_names_norm = [str(n).replace('/', '_') for n in sf_names]
        global_norm = [str(n).replace('/', '_') for n in global_names]
        norm_to_global_idx = {n: i for i, n in enumerate(global_norm)}
        sf_indices = [norm_to_global_idx[n] for n in sf_names_norm if n in norm_to_global_idx]

        if len(sf_indices) != len(sf_names_norm):
            matched = len(sf_indices)
            total = len(sf_names_norm)
            print(f"  [WARNING] Only {matched}/{total} subfamily members found in global covariance.")
    else:
        # Fallback: use group A + group B names
        print("  [WARNING] No subfamilies_summary.json found. Using group A + B names.")
        all_names = group_a_names + group_b_names
        all_norm = [str(n).replace('/', '_') for n in all_names]
        global_norm = [str(n).replace('/', '_') for n in global_names]
        norm_to_global_idx = {n: i for i, n in enumerate(global_norm)}
        sf_indices = [norm_to_global_idx[n] for n in all_norm if n in norm_to_global_idx]
        sf_names = all_names

    # Align embeddings with covariance order
    if emb_names is not None:
        emb_norm = [str(n).replace('/', '_') for n in emb_names]
        emb_norm_to_idx = {n: i for i, n in enumerate(emb_norm)}
        reorder = [emb_norm_to_idx[str(n).replace('/', '_')]
                   for n in global_names if str(n).replace('/', '_') in emb_norm_to_idx]
        emb_tensor = emb_tensor[reorder]

    # Determine the subfamily leaf names in covariance order
    sf_leaf_names = [global_names[i] for i in sf_indices]

    # --- Step 7: Load V matrices ---
    print("Loading covariance matrices...")
    V_A = load_matrix_tensor(detected["V_A"]).double()
    V_B = load_matrix_tensor(detected["V_B"]).double()
    U_local = load_matrix_tensor(detected["U_local"]).float()

    V_pooled = None
    if os.path.exists(detected["V_pooled"]):
        V_pooled = load_matrix_tensor(detected["V_pooled"]).double()

    # --- Step 8: Check for saved pPCA loadings ---
    saved_ppca = None
    ppca_dir = detected.get("ppca_dir", "")
    if ppca_dir and os.path.isdir(ppca_dir):
        v_path = os.path.join(ppca_dir, "V.npy")
        a_path = os.path.join(ppca_dir, "a.npy")
        if os.path.exists(v_path) and os.path.exists(a_path):
            print(f"  Using saved pPCA loadings from {ppca_dir}")
            saved_ppca = {
                "V": np.load(v_path),
                "a": np.load(a_path),
                "std_diag": None,
            }
            std_path = os.path.join(ppca_dir, "std_diag.npy")
            if os.path.exists(std_path):
                saved_ppca["std_diag"] = np.load(std_path)
            params_path = os.path.join(ppca_dir, "params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    saved_ppca["params"] = json.load(f)
                print(f"  pPCA params: {saved_ppca['params']}")

    # --- Step 9: Run analysis ---
    print("Running interpretability analysis...")
    result = run_full_interpretability(
        V_A=V_A, V_B=V_B, V_pooled=V_pooled, U_local=U_local,
        embeddings_full=emb_tensor, U_global=U_global,
        sf_indices=sf_indices, sf_names=sf_leaf_names,
        group_a_names=group_a_names, group_b_names=group_b_names,
        pca_min_variance=args.pca_variance,
        pca_min_components=args.pca_components,
        standardize=standardize, n_top=args.n_top,
        saved_ppca=saved_ppca,
    )

    # --- Step 10: Add metadata ---
    metadata = {
        "tool": "interpretability_analysis",
        "split_dir": os.path.abspath(args.split_dir),
        "family": detected["family"],
        "sf_name": detected["sf_name"],
        "rank_name": detected["rank_name"],
        "node_name": node_name,
        "n_A": len(group_a_names),
        "n_B": len(group_b_names),
        "lambda_obs": lambda_obs,
        "p_adj": p_adj,
        "pca_components_setting": args.pca_components,
        "pca_variance_setting": args.pca_variance,
        "standardize": standardize,
    }
    result["metadata"] = metadata

    # --- Step 10: Save JSON ---
    output_path = args.output or os.path.join(
        os.path.abspath(args.split_dir), "interpretability_analysis.json")
    save_results_json(result, output_path)

    # --- Step 11: Print summary ---
    print_summary(result, metadata)
    print(f"  Results saved to: {output_path}")

    # --- Step 12: Optional plots ---
    if args.plot:
        print("Generating diagnostic plots...")
        output_dir = os.path.dirname(output_path)
        generate_plots(result, output_dir)

    print("=" * 60)


if __name__ == "__main__":
    main()
