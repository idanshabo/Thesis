"""
Per-residue attribution for phylogenetic covariance shifts.

Because mean-pooled transformer embeddings are linear functions of per-residue
representations, the discriminating directions identified by the LRT can be
projected back to individual amino acid positions exactly (no approximation).

Given a significant covariance shift V_A != V_B:
1. Compute V_diff = V_A - V_B, eigendecompose, take top eigenvector
2. Back-project through pPCA loadings to original embedding space -> w
3. For each residue position i: score(i) = e_i^T w

Positions with high |score(i)| are the amino acids driving the divergence.
"""

import os
import torch
import numpy as np


def compute_discriminating_direction(V_A, V_B, ppca_loadings, std_diag=None,
                                     n_directions=1):
    """
    Compute the top discriminating direction(s) in original embedding space.

    1. V_diff = V_A - V_B (in PCA space)
    2. Eigendecompose V_diff, take eigenvector(s) with largest |eigenvalue|
    3. Back-project to original space via pPCA loadings

    Parameters
    ----------
    V_A : torch.Tensor
        (p_pca, p_pca) group A covariance in PCA space.
    V_B : torch.Tensor
        (p_pca, p_pca) group B covariance in PCA space.
    ppca_loadings : numpy.ndarray
        (p_original, p_pca) pPCA loadings matrix (W from PhylogeneticPCA).
    std_diag : numpy.ndarray or None
        (p_original,) scaling factors for correlation-mode pPCA.
    n_directions : int
        Number of top directions to return.

    Returns
    -------
    results : list of dict
        Each dict has keys: 'w_original', 'eigenvalue', 'w_pca',
                            'variance_explained_frac'.
    """
    V_A = V_A.double()
    V_B = V_B.double()
    V_diff = V_A - V_B
    V_diff = (V_diff + V_diff.T) / 2.0

    evals, evecs = torch.linalg.eigh(V_diff)

    # Sort by absolute eigenvalue (descending)
    order = torch.argsort(evals.abs(), descending=True)
    total_var = evals.abs().sum().item()

    results = []
    for rank in range(min(n_directions, len(order))):
        idx = order[rank]
        eigenvalue = evals[idx].item()
        w_pca = evecs[:, idx].cpu().numpy()

        # Back-project: w_original = W @ w_pca
        p_pca = ppca_loadings.shape[1]
        if len(w_pca) > p_pca:
            w_pca = w_pca[:p_pca]
        elif len(w_pca) < p_pca:
            w_pca = np.pad(w_pca, (0, p_pca - len(w_pca)))

        w_original = ppca_loadings @ w_pca

        if std_diag is not None:
            w_original = w_original * std_diag

        # Normalize
        norm = np.linalg.norm(w_original)
        if norm > 1e-12:
            w_original = w_original / norm

        frac = abs(eigenvalue) / max(total_var, 1e-12)

        results.append({
            'w_original': w_original,
            'eigenvalue': eigenvalue,
            'w_pca': w_pca,
            'variance_explained_frac': frac,
        })

    return results


def compute_residue_scores(per_residue_embeddings, w_direction):
    """
    Compute per-residue attribution scores for a single sequence.

    score(i) = e_i^T w

    This is exact because mean-pooling is linear:
    ebar^T w = (1/L) sum_i e_i^T w

    Parameters
    ----------
    per_residue_embeddings : torch.Tensor or numpy.ndarray
        (L, p_original) per-residue embeddings for one sequence.
    w_direction : numpy.ndarray
        (p_original,) discriminating direction.

    Returns
    -------
    scores : numpy.ndarray
        (L,) per-position attribution scores.
    """
    if isinstance(per_residue_embeddings, torch.Tensor):
        emb = per_residue_embeddings.cpu().numpy()
    else:
        emb = np.asarray(per_residue_embeddings)

    w = np.asarray(w_direction, dtype=np.float64)

    # Handle dimension mismatch
    p_emb = emb.shape[1]
    p_w = len(w)
    if p_emb != p_w:
        p_min = min(p_emb, p_w)
        emb = emb[:, :p_min]
        w = w[:p_min]

    scores = emb @ w  # (L,)
    return scores.astype(np.float64)


def map_scores_to_msa(scores, ungapped_sequence, aligned_sequence):
    """
    Map ungapped residue scores to MSA column positions.

    Parameters
    ----------
    scores : numpy.ndarray
        (L_ungapped,) scores for each residue in the ungapped sequence.
    ungapped_sequence : str
        The protein sequence without gaps.
    aligned_sequence : str
        The aligned sequence (with gap characters '-' or '.').

    Returns
    -------
    msa_scores : numpy.ndarray
        (L_aligned,) scores mapped to MSA columns. Gap positions get 0.0.
    """
    msa_scores = np.zeros(len(aligned_sequence), dtype=np.float64)
    residue_idx = 0
    for col_idx, char in enumerate(aligned_sequence):
        if char not in ('-', '.'):
            if residue_idx < len(scores):
                msa_scores[col_idx] = scores[residue_idx]
            residue_idx += 1
    return msa_scores


def compute_group_attribution(per_residue_dir, group_names, w_direction,
                               msa_sequences=None, n_top=20):
    """
    Compute average per-residue attribution across all sequences in a group,
    optionally mapped to MSA columns.

    Parameters
    ----------
    per_residue_dir : str
        Directory containing per-residue .pt files (named {seq_name}.pt).
    group_names : list of str
        Sequence names in this group.
    w_direction : numpy.ndarray
        (p_original,) discriminating direction.
    msa_sequences : dict or None
        If provided, {name: aligned_sequence_str} for MSA column mapping.
    n_top : int
        Number of top positions to report.

    Returns
    -------
    result : dict
        'per_sequence_scores': dict of {name: (L,) array}
        'mean_scores_msa': (n_msa_cols,) array if msa_sequences provided
        'top_positions': list of (position, score) sorted by |score|
        'n_sequences_loaded': int
    """
    per_seq_scores = {}
    n_loaded = 0

    for name in group_names:
        # Try common naming patterns
        candidates = [
            os.path.join(per_residue_dir, f"{name}.pt"),
            os.path.join(per_residue_dir, f"{name.replace('/', '_')}.pt"),
        ]
        loaded = False
        for path in candidates:
            if os.path.exists(path):
                emb = torch.load(path, map_location='cpu')
                if isinstance(emb, dict):
                    emb = emb.get('embeddings', emb.get('representations', emb))
                if isinstance(emb, torch.Tensor):
                    emb = emb.squeeze(0)  # Remove batch dim if present
                    scores = compute_residue_scores(emb, w_direction)
                    per_seq_scores[name] = scores
                    n_loaded += 1
                    loaded = True
                    break
        if not loaded:
            pass  # Skip sequences without per-residue data

    result = {
        'per_sequence_scores': {k: v.tolist() for k, v in per_seq_scores.items()},
        'n_sequences_loaded': n_loaded,
        'n_sequences_requested': len(group_names),
    }

    # Map to MSA columns if sequences provided
    if msa_sequences is not None and n_loaded > 0:
        n_cols = len(next(iter(msa_sequences.values())))
        all_msa_scores = []

        for name, scores in per_seq_scores.items():
            if name in msa_sequences:
                aligned = msa_sequences[name]
                ungapped = aligned.replace('-', '').replace('.', '')
                msa_scores = map_scores_to_msa(scores, ungapped, aligned)
                all_msa_scores.append(msa_scores)

        if all_msa_scores:
            mean_msa = np.mean(all_msa_scores, axis=0)
            result['mean_scores_msa'] = mean_msa.tolist()

            # Top positions by absolute score
            top_idx = np.argsort(np.abs(mean_msa))[::-1][:n_top]
            result['top_positions'] = [
                {'msa_column': int(idx), 'score': float(mean_msa[idx])}
                for idx in top_idx
            ]

    return result


def run_residue_attribution(V_A, V_B, ppca_loadings, std_diag,
                            per_residue_dir, group_a_names, group_b_names,
                            msa_sequences=None, n_top=20):
    """
    Full per-residue attribution pipeline for a single split.

    1. Compute discriminating direction w (top eigenvector of V_A - V_B)
    2. Back-project to original embedding space
    3. Compute per-residue scores for each sequence in both groups
    4. Aggregate and identify top contributing MSA positions

    Parameters
    ----------
    V_A : torch.Tensor
        (p_pca, p_pca) group A covariance.
    V_B : torch.Tensor
        (p_pca, p_pca) group B covariance.
    ppca_loadings : numpy.ndarray
        (p_original, p_pca) pPCA loadings matrix.
    std_diag : numpy.ndarray or None
        Scaling factors for correlation-mode pPCA.
    per_residue_dir : str
        Directory with per-residue .pt files.
    group_a_names : list of str
        Leaf names in group A.
    group_b_names : list of str
        Leaf names in group B.
    msa_sequences : dict or None
        {name: aligned_sequence} for MSA mapping.
    n_top : int
        Number of top positions to report.

    Returns
    -------
    result : dict
        Combined attribution results.
    """
    # Step 1: Discriminating direction
    directions = compute_discriminating_direction(
        V_A, V_B, ppca_loadings, std_diag, n_directions=3
    )

    top_dir = directions[0]
    w = top_dir['w_original']

    print(f"  Top discriminating direction: eigenvalue={top_dir['eigenvalue']:.4f}, "
          f"var_explained={top_dir['variance_explained_frac']:.1%}")

    # Step 2: Attribution for group A
    print(f"  Computing attribution for group A ({len(group_a_names)} sequences)...")
    attr_a = compute_group_attribution(
        per_residue_dir, group_a_names, w, msa_sequences, n_top
    )

    # Step 3: Attribution for group B
    print(f"  Computing attribution for group B ({len(group_b_names)} sequences)...")
    attr_b = compute_group_attribution(
        per_residue_dir, group_b_names, w, msa_sequences, n_top
    )

    # Step 4: Differential attribution (A vs B)
    diff_result = {}
    if ('mean_scores_msa' in attr_a and 'mean_scores_msa' in attr_b):
        mean_a = np.array(attr_a['mean_scores_msa'])
        mean_b = np.array(attr_b['mean_scores_msa'])
        diff = mean_a - mean_b
        top_diff_idx = np.argsort(np.abs(diff))[::-1][:n_top]
        diff_result = {
            'differential_scores_msa': diff.tolist(),
            'top_differential_positions': [
                {'msa_column': int(idx),
                 'score_a': float(mean_a[idx]),
                 'score_b': float(mean_b[idx]),
                 'difference': float(diff[idx])}
                for idx in top_diff_idx
            ],
        }

    return {
        'discriminating_directions': [
            {k: v if not isinstance(v, np.ndarray) else v.tolist()
             for k, v in d.items()}
            for d in directions
        ],
        'group_a_attribution': attr_a,
        'group_b_attribution': attr_b,
        'differential_attribution': diff_result,
    }


if __name__ == "__main__":
    # Simple test with synthetic data
    p_pca = 10
    p_orig = 50

    V_A = torch.eye(p_pca, dtype=torch.float64) * 2.0
    V_B = torch.eye(p_pca, dtype=torch.float64)
    V_A[0, 0] = 5.0  # Make dimension 0 differ strongly

    loadings = np.random.randn(p_orig, p_pca)

    dirs = compute_discriminating_direction(V_A, V_B, loadings)
    print(f"Top direction eigenvalue: {dirs[0]['eigenvalue']:.4f}")
    print(f"Direction shape: {dirs[0]['w_original'].shape}")

    # Test per-residue scoring
    fake_emb = np.random.randn(100, p_orig)  # 100 residues
    scores = compute_residue_scores(fake_emb, dirs[0]['w_original'])
    print(f"Scores shape: {scores.shape}, mean: {scores.mean():.4f}")

    print("All tests passed.")
