"""
Interpretability analysis functions for phylogenetic split results.

Provides three analyses:
1. Mean shift direction between groups A and B
2. Covariance eigendecomposition and comparison
3. Projection from PCA space back to original embedding space

All functions operate on tensors and return plain dicts (JSON-serializable).
"""

import torch
import numpy as np

from evaluate_split_options.lrt_statistics import compute_gls_operators, add_jitter
from evaluate_split_options.evaluate_split_options import PhylogeneticPCA


def compute_gls_mean(X, U):
    """
    Computes the GLS mean vector: mu_hat = (1^T U^{-1} 1)^{-1} 1^T U^{-1} X.

    Parameters
    ----------
    X : torch.Tensor
        (n, p) data matrix.
    U : torch.Tensor
        (n, n) phylogenetic covariance matrix.

    Returns
    -------
    mu_hat : torch.Tensor
        (p,) GLS mean vector.
    """
    X = X.double()
    U = U.double()
    _, _, term1, term2 = compute_gls_operators(U)
    mu_hat = (term1 @ term2 @ X).squeeze()
    return mu_hat


def analyze_mean_shift(X_A, X_B, U_A, U_B, pca_loadings=None, std_diag=None, n_top=20):
    """
    Characterizes the mean shift vector between two groups.

    Parameters
    ----------
    X_A : torch.Tensor
        (n_A, p) group A data in PCA space.
    X_B : torch.Tensor
        (n_B, p) group B data in PCA space.
    U_A : torch.Tensor
        (n_A, n_A) phylogenetic covariance for group A.
    U_B : torch.Tensor
        (n_B, n_B) phylogenetic covariance for group B.
    pca_loadings : np.ndarray, optional
        (p_original, p_pca) loadings matrix (pPCA.V) for back-projection.
    std_diag : np.ndarray, optional
        (p_original,) scaling factors for correlation-mode pPCA.
    n_top : int
        Number of top dimensions to report.

    Returns
    -------
    result : dict
    """
    mu_A = compute_gls_mean(X_A, U_A)
    mu_B = compute_gls_mean(X_B, U_B)
    delta_mu = mu_A - mu_B

    magnitude = torch.norm(delta_mu).item()
    direction = (delta_mu / max(magnitude, 1e-12)).cpu().numpy()
    delta_np = delta_mu.cpu().numpy()
    mu_A_np = mu_A.cpu().numpy()
    mu_B_np = mu_B.cpu().numpy()

    # Per-dimension contribution (fraction of squared shift)
    sq = delta_np ** 2
    total_sq = max(sq.sum(), 1e-12)
    contributions = sq / total_sq

    # Top contributing PCA dimensions
    top_idx = np.argsort(contributions)[::-1][:n_top]

    result = {
        "mu_A_pca": mu_A_np.tolist(),
        "mu_B_pca": mu_B_np.tolist(),
        "delta_mu_pca": delta_np.tolist(),
        "delta_mu_magnitude_pca": magnitude,
        "delta_mu_direction_pca": direction.tolist(),
        "per_dim_contribution_pca": contributions.tolist(),
        "top_contributing_dims_pca": top_idx.tolist(),
        "top_contributing_values_pca": contributions[top_idx].tolist(),
        "cosine_with_pc_axes": np.abs(direction).tolist(),
    }

    # Back-projection to original space
    if pca_loadings is not None:
        delta_mu_orig = pca_loadings @ delta_np  # (p_orig,)
        if std_diag is not None:
            delta_mu_orig = delta_mu_orig * std_diag
        mag_orig = float(np.linalg.norm(delta_mu_orig))
        top_orig = np.argsort(np.abs(delta_mu_orig))[::-1][:n_top]
        result["delta_mu_original"] = delta_mu_orig.tolist()
        result["delta_mu_magnitude_original"] = mag_orig
        result["top_original_dims_by_mean_shift"] = top_orig.tolist()
        result["top_original_dims_mean_shift_values"] = delta_mu_orig[top_orig].tolist()

    return result


def analyze_covariance_decomposition(V_A, V_B, V_pooled=None, n_top=10):
    """
    Eigendecomposes and compares covariance matrices V_A and V_B.

    Parameters
    ----------
    V_A : torch.Tensor
        (p, p) group A trait covariance.
    V_B : torch.Tensor
        (p, p) group B trait covariance.
    V_pooled : torch.Tensor, optional
        (p, p) pooled (H0) trait covariance.
    n_top : int
        Number of top dimensions for alignment matrix.

    Returns
    -------
    result : dict
    """
    V_A = V_A.double()
    V_B = V_B.double()

    def sorted_eigh(M):
        evals, evecs = torch.linalg.eigh(M)
        idx = torch.argsort(evals, descending=True)
        return evals[idx], evecs[:, idx]

    evals_A, evecs_A = sorted_eigh(V_A)
    evals_B, evecs_B = sorted_eigh(V_B)

    evals_A_np = evals_A.cpu().numpy()
    evals_B_np = evals_B.cpu().numpy()

    # Eigenvalue ratios (epsilon-guarded)
    eps = 1e-10
    ratios = evals_A_np / np.maximum(evals_B_np, eps)
    diffs = evals_A_np - evals_B_np

    # Top divergent dimensions (by |log(ratio)|)
    log_ratios = np.abs(np.log(np.maximum(ratios, eps)))
    div_idx = np.argsort(log_ratios)[::-1][:n_top]

    # Subspace alignment matrix: |q_A_i^T q_B_j| for top n_top
    k = min(n_top, evecs_A.shape[1])
    alignment = torch.abs(evecs_A[:, :k].T @ evecs_B[:, :k]).cpu().numpy()

    # Frobenius distance
    frob_AB = torch.norm(V_A - V_B, p='fro').item()

    # Log-determinants
    log_det_A = torch.linalg.slogdet(add_jitter(V_A))[1].item()
    log_det_B = torch.linalg.slogdet(add_jitter(V_B))[1].item()

    # Trace
    trace_A = torch.trace(V_A).item()
    trace_B = torch.trace(V_B).item()

    # Effective rank: (sum lambda_i)^2 / sum(lambda_i^2)
    pos_A = np.maximum(evals_A_np, 0)
    pos_B = np.maximum(evals_B_np, 0)
    eff_rank_A = float(pos_A.sum() ** 2 / max(np.sum(pos_A ** 2), eps))
    eff_rank_B = float(pos_B.sum() ** 2 / max(np.sum(pos_B ** 2), eps))

    result = {
        "eigenvalues_A": evals_A_np.tolist(),
        "eigenvalues_B": evals_B_np.tolist(),
        "eigenvalue_ratio_A_over_B": ratios.tolist(),
        "eigenvalue_diff_A_minus_B": diffs.tolist(),
        "top_divergent_dims": div_idx.tolist(),
        "top_divergent_ratios": ratios[div_idx].tolist(),
        "subspace_alignment_matrix": alignment.tolist(),
        "frobenius_A_B": frob_AB,
        "log_det_V_A": log_det_A,
        "log_det_V_B": log_det_B,
        "trace_V_A": trace_A,
        "trace_V_B": trace_B,
        "effective_rank_A": eff_rank_A,
        "effective_rank_B": eff_rank_B,
    }

    # Pooled comparison
    if V_pooled is not None:
        V_pooled = V_pooled.double()
        evals_P, _ = sorted_eigh(V_pooled)
        result["eigenvalues_pooled"] = evals_P.cpu().numpy().tolist()
        result["frobenius_A_pooled"] = torch.norm(V_A - V_pooled, p='fro').item()
        result["frobenius_B_pooled"] = torch.norm(V_B - V_pooled, p='fro').item()
        result["log_det_V_pooled"] = torch.linalg.slogdet(add_jitter(V_pooled))[1].item()

    # Top eigenvectors (in PCA space) for later projection
    result["_evecs_A"] = evecs_A[:, :n_top].cpu().numpy()
    result["_evecs_B"] = evecs_B[:, :n_top].cpu().numpy()

    return result


def refit_ppca(embeddings_full, U_global, sf_indices,
               pca_min_variance, pca_min_components, standardize):
    """
    Re-fits the PhylogeneticPCA to recover the loadings matrix.

    Reproduces the exact pPCA fit done in the pipeline
    (evaluate_split_options.py:500-506).

    Parameters
    ----------
    embeddings_full : torch.Tensor
        (N_global, p_original) full aligned embeddings.
    U_global : torch.Tensor
        (N_global, N_global) global phylogenetic covariance matrix.
    sf_indices : list of int
        Indices into global arrays for this subfamily.
    pca_min_variance : float
        Minimum variance threshold for pPCA.
    pca_min_components : int
        Minimum number of components for pPCA.
    standardize : bool
        If True, use correlation-mode pPCA.

    Returns
    -------
    ppca : PhylogeneticPCA
        Fitted pPCA object with .V (loadings), .a (phylo mean), .std_diag.
    """
    idx = torch.tensor(sf_indices, dtype=torch.long)
    Y_local = embeddings_full[idx].float()
    U_local = U_global[idx][:, idx].float()

    # Shift to positive (matching pipeline: evaluate_split_options.py:485)
    U_shifted = U_local - torch.min(U_local)

    mode = 'corr' if standardize else 'cov'
    ppca = PhylogeneticPCA(
        min_variance=pca_min_variance,
        min_components=pca_min_components,
        mode=mode
    )
    ppca.fit(Y_local.cpu().numpy(), U_shifted.cpu().numpy())
    return ppca


def project_to_original_space(vectors_pca, ppca):
    """
    Projects vectors from PCA space back to original embedding space.

    Parameters
    ----------
    vectors_pca : np.ndarray
        (k, p_pca) or (p_pca,) vectors in PCA space.
    ppca : PhylogeneticPCA
        Fitted pPCA object.

    Returns
    -------
    vectors_orig : np.ndarray
        (k, p_original) or (p_original,) projected vectors.
    """
    squeeze = vectors_pca.ndim == 1
    if squeeze:
        vectors_pca = vectors_pca[np.newaxis, :]

    # Truncate/pad if dimension mismatch
    p_pca = ppca.V.shape[1]
    if vectors_pca.shape[1] > p_pca:
        vectors_pca = vectors_pca[:, :p_pca]
    elif vectors_pca.shape[1] < p_pca:
        pad = np.zeros((vectors_pca.shape[0], p_pca - vectors_pca.shape[1]))
        vectors_pca = np.hstack([vectors_pca, pad])

    vectors_orig = vectors_pca @ ppca.V.T  # (k, p_original)

    if ppca.mode == 'corr' and ppca.std_diag is not None:
        vectors_orig = vectors_orig * ppca.std_diag

    if squeeze:
        vectors_orig = vectors_orig.squeeze(0)

    return vectors_orig


def run_full_interpretability(V_A, V_B, V_pooled, U_local,
                              embeddings_full, U_global,
                              sf_indices, sf_names,
                              group_a_names, group_b_names,
                              pca_min_variance, pca_min_components,
                              standardize, n_top=20, saved_ppca=None):
    """
    Orchestrates all three interpretability analyses for a single split.

    Parameters
    ----------
    V_A : torch.Tensor
        (p_pca, p_pca) group A trait covariance (in PCA space).
    V_B : torch.Tensor
        (p_pca, p_pca) group B trait covariance (in PCA space).
    V_pooled : torch.Tensor or None
        (p_pca, p_pca) pooled trait covariance (in PCA space).
    U_local : torch.Tensor
        (n_sf, n_sf) local phylogenetic covariance for this subfamily.
    embeddings_full : torch.Tensor
        (N_global, p_original) full aligned embeddings.
    U_global : torch.Tensor
        (N_global, N_global) global phylogenetic covariance.
    sf_indices : list of int
        Global indices for this subfamily.
    sf_names : list of str
        Leaf names for this subfamily (in covariance order).
    group_a_names : list of str
        Leaf names in group A.
    group_b_names : list of str
        Leaf names in group B.
    pca_min_variance : float
        pPCA min variance parameter.
    pca_min_components : int
        pPCA min components parameter.
    standardize : bool
        Whether pPCA used correlation mode.
    n_top : int
        Number of top dimensions to report.
    saved_ppca : dict or None
        If provided, dict with keys 'V', 'a', 'std_diag' (numpy arrays)
        from saved pPCA loadings. Skips re-fitting when available.

    Returns
    -------
    result : dict
        Combined results from all three analyses.
    """
    p_original = embeddings_full.shape[1]

    # --- Step 1: Get pPCA loadings (saved or re-fit) ---
    if saved_ppca is not None:
        print("  Using saved pPCA loadings (no re-fit needed).")
        ppca = PhylogeneticPCA(min_variance=pca_min_variance,
                               min_components=pca_min_components,
                               mode='corr' if standardize else 'cov')
        ppca.V = saved_ppca["V"]
        ppca.a = saved_ppca["a"]
        ppca.std_diag = saved_ppca["std_diag"]
        ppca.final_n_components = ppca.V.shape[1]
    else:
        print("  Re-fitting pPCA to recover loadings...")
        ppca = refit_ppca(embeddings_full, U_global, sf_indices,
                          pca_min_variance, pca_min_components, standardize)
    p_pca = ppca.final_n_components

    # Check dimension consistency
    p_from_V = V_A.shape[0]
    if p_pca != p_from_V:
        print(f"  [WARNING] Re-fitted pPCA has {p_pca} components but V matrices have {p_from_V}. "
              f"Using min({p_pca}, {p_from_V}).")

    # --- Step 2: Project embeddings through pPCA ---
    idx = torch.tensor(sf_indices, dtype=torch.long)
    Y_local = embeddings_full[idx].float()
    X_sf = torch.from_numpy(ppca.transform(Y_local.cpu().numpy())).double()

    # --- Step 3: Map group names to local indices ---
    name_to_local = {}
    for i, name in enumerate(sf_names):
        name_to_local[str(name).replace('/', '_')] = i
        name_to_local[str(name)] = i

    local_idx_A = []
    for name in group_a_names:
        norm = str(name).replace('/', '_')
        if norm in name_to_local:
            local_idx_A.append(name_to_local[norm])
    local_idx_B = []
    for name in group_b_names:
        norm = str(name).replace('/', '_')
        if norm in name_to_local:
            local_idx_B.append(name_to_local[norm])

    if not local_idx_A or not local_idx_B:
        raise ValueError(f"Empty group after name matching: A={len(local_idx_A)}, B={len(local_idx_B)}")

    X_A = X_sf[local_idx_A]
    X_B = X_sf[local_idx_B]

    # Use the shifted local covariance for GLS
    U_A = U_local[local_idx_A][:, local_idx_A].double()
    U_B = U_local[local_idx_B][:, local_idx_B].double()

    # --- Step 4: Analysis 1 — Mean Shift ---
    print("  Running mean shift analysis...")
    mean_shift_result = analyze_mean_shift(
        X_A, X_B, U_A, U_B,
        pca_loadings=ppca.V,
        std_diag=ppca.std_diag if ppca.mode == 'corr' else None,
        n_top=n_top
    )

    # --- Step 5: Analysis 2 — Covariance Decomposition ---
    print("  Running covariance decomposition...")
    # Use only the p_pca dimensions that match
    p_use = min(p_pca, p_from_V)
    cov_result = analyze_covariance_decomposition(
        V_A[:p_use, :p_use], V_B[:p_use, :p_use],
        V_pooled[:p_use, :p_use] if V_pooled is not None else None,
        n_top=n_top
    )

    # --- Step 6: Analysis 3 — Projection to Original Space ---
    print("  Projecting to original embedding space...")
    projection_result = {
        "ppca_n_components_recovered": p_pca,
        "ppca_mode": ppca.mode,
        "p_original": p_original,
    }

    # Project top eigenvectors of V difference to original space
    evecs_A = cov_result.pop("_evecs_A")
    evecs_B = cov_result.pop("_evecs_B")

    # Eigenvectors of V_A - V_B (difference matrix)
    V_diff = V_A[:p_use, :p_use].double() - V_B[:p_use, :p_use].double()
    V_diff = (V_diff + V_diff.T) / 2.0
    evals_diff, evecs_diff = torch.linalg.eigh(V_diff)
    idx_sorted = torch.argsort(evals_diff.abs(), descending=True)
    top_k = min(n_top, len(idx_sorted))
    top_diff_evecs = evecs_diff[:, idx_sorted[:top_k]].cpu().numpy()
    top_diff_evals = evals_diff[idx_sorted[:top_k]].cpu().numpy()

    # Project to original space
    diff_evecs_orig = project_to_original_space(top_diff_evecs.T, ppca)  # (top_k, p_orig)

    # Find which original dimensions are most affected by V divergence
    importance_orig = np.mean(np.abs(diff_evecs_orig[:min(5, top_k)]), axis=0)
    top_orig_by_V = np.argsort(importance_orig)[::-1][:n_top]

    projection_result["top_V_diff_eigenvalues"] = top_diff_evals.tolist()
    projection_result["top_original_dims_by_V_divergence"] = top_orig_by_V.tolist()
    projection_result["top_original_dims_V_divergence_values"] = importance_orig[top_orig_by_V].tolist()
    projection_result["reconstruction_loss_note"] = (
        f"Lossy projection: only {p_pca} of {p_original} original dimensions retained by pPCA"
    )

    # --- Combine ---
    return {
        "analysis_1_mean_shift": mean_shift_result,
        "analysis_2_covariance_decomposition": cov_result,
        "analysis_3_projection_to_original": projection_result,
    }
