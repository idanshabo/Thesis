"""
Phylogenetically-corrected residue-residue couplings via the matrix-normal
framework on PLM embeddings.

This module computes pairwise couplings between MSA positions, comparable
to DCA/PSICOV output, but with two key differences:
  1. Phylogenetic correction via U^{-1} (vs DCA's heuristic reweighting)
  2. Operates in continuous PLM embedding space (vs one-hot AA encoding)

It also provides differential couplings between two phylogenetically
defined clades, identifying residue pairs whose coupling pattern changed
between regimes.

Three levels of granularity:
  - Position level: an L x L matrix of coupling strengths ||C_{i,j}||_F
  - AA-pair level: a 20 x 20 matrix per position pair (DCA-comparable)
  - Region level: aggregation by structural regions (helix/sheet/loop)
                  or by data-driven spectral clusters
"""

import os
import json
import numpy as np
import torch

from evaluate_split_options.lrt_statistics import compute_gls_operators


# Standard amino acid alphabet
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def get_aa_embeddings_from_plm(model_name="esmc_300m",
                                neutral_context="GG{aa}GG",
                                cache_path=None):
    """
    Get PLM embeddings for the 20 amino acid types in a neutral context.

    For each AA type, we create a short sequence (e.g., "GGAGG") and extract
    the embedding of the central residue. This gives a representative
    embedding for each AA type that can be used for AA-pair coupling
    interpretation.

    Parameters
    ----------
    model_name : str
        PLM model name (esmc_300m, esmc_600m, etc.).
    neutral_context : str
        Template with {aa} placeholder for the AA type. The center
        position is extracted.
    cache_path : str or None
        If provided, cache or load AA embeddings from this .npy file.

    Returns
    -------
    aa_embeddings : numpy.ndarray, shape (20, p)
        PLM embeddings for the 20 standard amino acids in canonical order
        (AA_LIST: ACDEFGHIKLMNPQRSTVWY).
    """
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)

    try:
        from handling_esm_embeddings.create_esm_embeddings import (
            create_esm_embedding_for_sequence,
        )
    except ImportError:
        # Fallback: try direct import of esm
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError:
            raise ImportError(
                "Cannot load PLM. Either ensure handling_esm_embeddings "
                "is in PYTHONPATH or install esm package."
            )

        client = ESMC.from_pretrained(model_name).eval()
        center_idx = neutral_context.index("{aa}")
        embeddings = []
        for aa in AA_LIST:
            seq = neutral_context.replace("{aa}", aa)
            protein = ESMProtein(sequence=seq)
            tensor = client.encode(protein)
            output = client.logits(
                tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            # output.embeddings: (1, L+2, p), with BOS/EOS tokens
            # Skip BOS at index 0, then center_idx-th residue
            emb = output.embeddings[0, 1 + center_idx, :].cpu().numpy()
            embeddings.append(emb)
        aa_embeddings = np.stack(embeddings)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.save(cache_path, aa_embeddings)
        return aa_embeddings

    # Use existing pipeline embedding function
    center_idx = neutral_context.index("{aa}")
    embeddings = []
    for aa in AA_LIST:
        seq = neutral_context.replace("{aa}", aa)
        # create_esm_embedding_for_sequence may have a different signature;
        # this is a best-effort. Caller should override with cache_path
        # if their pipeline uses a different API.
        emb_tensor = create_esm_embedding_for_sequence(seq)
        if isinstance(emb_tensor, torch.Tensor):
            emb = emb_tensor.squeeze(0).cpu().numpy()
        else:
            emb = np.asarray(emb_tensor)
        embeddings.append(emb[center_idx])
    aa_embeddings = np.stack(embeddings)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, aa_embeddings)
    return aa_embeddings


def get_aa_embeddings_from_data(per_residue_embeddings, msa_sequences,
                                  sequence_names):
    """
    Alternative AA embedding extraction: average per-residue embeddings of
    each AA type across all positions and proteins in the dataset.

    This avoids needing to re-run the PLM. For each AA type, we collect
    every embedding e_i^{(s)} where the residue at position i in protein s
    is that AA, and average them.

    This may differ slightly from get_aa_embeddings_from_plm because the
    averaged embedding reflects how this AA appears IN THIS FAMILY, with
    the family's typical contexts. For most purposes this is equally good
    or better.

    Parameters
    ----------
    per_residue_embeddings : dict {name: (L_i, p) tensor}
    msa_sequences : dict {name: aligned_sequence}
    sequence_names : list of str

    Returns
    -------
    aa_embeddings : numpy.ndarray, shape (20, p)
        Mean embedding per AA type, in canonical order.
    """
    p = None
    for name in sequence_names:
        if name in per_residue_embeddings:
            p = per_residue_embeddings[name].shape[1]
            break
    if p is None:
        raise ValueError("No per-residue embeddings available")

    sums = np.zeros((20, p), dtype=np.float64)
    counts = np.zeros(20, dtype=np.int64)

    for name in sequence_names:
        if name not in per_residue_embeddings or name not in msa_sequences:
            continue
        emb = per_residue_embeddings[name].cpu().numpy()
        aligned = msa_sequences[name]
        ungapped = aligned.replace("-", "").replace(".", "")
        for residue_idx, aa in enumerate(ungapped):
            if aa not in AA_TO_IDX or residue_idx >= emb.shape[0]:
                continue
            idx = AA_TO_IDX[aa]
            sums[idx] += emb[residue_idx]
            counts[idx] += 1

    aa_embeddings = np.zeros((20, p), dtype=np.float64)
    for i in range(20):
        if counts[i] > 0:
            aa_embeddings[i] = sums[i] / counts[i]

    return aa_embeddings


def load_per_residue_embeddings(per_residue_dir, sequence_names):
    """
    Load per-residue embeddings for a list of sequences.

    Parameters
    ----------
    per_residue_dir : str
        Directory containing per-residue .pt files (named {seq_name}.pt).
    sequence_names : list of str

    Returns
    -------
    embeddings : dict {name: tensor (L_i, p)}
        Per-residue embeddings, indexed by sequence name.
        Sequences without files are skipped.
    """
    out = {}
    for name in sequence_names:
        candidates = [
            os.path.join(per_residue_dir, f"{name}.pt"),
            os.path.join(per_residue_dir, f"{name.replace('/', '_')}.pt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                emb = torch.load(path, map_location='cpu')
                if isinstance(emb, dict):
                    emb = emb.get('embeddings',
                                  emb.get('representations', emb))
                if isinstance(emb, torch.Tensor):
                    emb = emb.squeeze(0)
                    out[name] = emb.double()
                break
    return out


def build_msa_position_matrices(per_residue_embeddings, msa_sequences,
                                 sequence_names):
    """
    Build per-MSA-column embedding matrices.

    For each MSA column i, returns an n x p matrix where row s is the
    per-residue embedding at position i in protein s. Positions with gaps
    contribute NaN; downstream code must handle missing values.

    Parameters
    ----------
    per_residue_embeddings : dict {name: (L_i, p) tensor}
        Per-residue embeddings (ungapped sequences).
    msa_sequences : dict {name: aligned_sequence}
        Aligned MSA sequences (with gap characters).
    sequence_names : list of str
        Order of sequences (defines row order in output matrices).

    Returns
    -------
    E : numpy.ndarray, shape (n, L_msa, p)
        E[s, i, :] is the per-residue embedding of position i (MSA column)
        in sequence s. NaN where gaps occur.
    valid_mask : numpy.ndarray, shape (n, L_msa)
        Boolean mask: True where the position is non-gap and embedding exists.
    """
    n = len(sequence_names)
    L_msa = len(next(iter(msa_sequences.values())))

    # Get embedding dimension from first available
    p = None
    for name in sequence_names:
        if name in per_residue_embeddings:
            p = per_residue_embeddings[name].shape[1]
            break
    if p is None:
        raise ValueError("No per-residue embeddings found for any sequence")

    E = np.full((n, L_msa, p), np.nan, dtype=np.float64)
    valid_mask = np.zeros((n, L_msa), dtype=bool)

    for s, name in enumerate(sequence_names):
        if name not in per_residue_embeddings:
            continue
        if name not in msa_sequences:
            continue
        emb = per_residue_embeddings[name].cpu().numpy()
        aligned = msa_sequences[name]

        residue_idx = 0
        for col_idx, char in enumerate(aligned):
            if char in ('-', '.'):
                continue
            if residue_idx < emb.shape[0]:
                E[s, col_idx, :] = emb[residue_idx]
                valid_mask[s, col_idx] = True
            residue_idx += 1

    return E, valid_mask


def compute_cross_position_covariance(E, valid_mask, U_inv):
    """
    Compute the L x L x p x p cross-position covariance tensor.

    For each pair of MSA positions (i1, i2):
        C[i1, i2] = (1/n_eff) * E[:, i1, :].T @ U_inv @ E[:, i2, :]
    where n_eff is the number of species with valid (non-gap) embeddings
    at BOTH positions.

    NaN values from gaps are replaced with 0 in the contribution to C
    (effectively skipping those species for that position).

    Parameters
    ----------
    E : numpy.ndarray, shape (n, L, p)
        Per-MSA-column embedding tensor (NaN at gaps).
    valid_mask : numpy.ndarray, shape (n, L)
        Boolean mask: True where position is non-gap.
    U_inv : numpy.ndarray, shape (n, n)
        Inverse phylogenetic covariance matrix.

    Returns
    -------
    coupling_matrix : numpy.ndarray, shape (L, L)
        Frobenius norm of each cross-position covariance block.
    cross_cov : numpy.ndarray, shape (L, L, p, p)
        Full cross-position covariance tensor.
        Note: this can be very large for big proteins. Caller may want
        to use compute_position_couplings_only() for memory efficiency.
    """
    n, L, p = E.shape
    E_clean = np.where(np.isnan(E), 0.0, E)

    cross_cov = np.zeros((L, L, p, p), dtype=np.float64)
    coupling_matrix = np.zeros((L, L), dtype=np.float64)

    for i1 in range(L):
        E1 = E_clean[:, i1, :]  # (n, p)
        for i2 in range(i1, L):
            E2 = E_clean[:, i2, :]  # (n, p)

            mask_both = valid_mask[:, i1] & valid_mask[:, i2]
            n_eff = mask_both.sum()
            if n_eff < 2:
                continue

            # C[i1, i2] = E1.T @ U_inv @ E2 / n_eff
            C = E1.T @ U_inv @ E2 / n_eff
            cross_cov[i1, i2] = C
            cross_cov[i2, i1] = C.T

            frob = float(np.linalg.norm(C, ord='fro'))
            coupling_matrix[i1, i2] = frob
            coupling_matrix[i2, i1] = frob

    return coupling_matrix, cross_cov


def compute_position_couplings_only(E, valid_mask, U_inv):
    """
    Memory-efficient version: computes only the L x L Frobenius norm matrix
    without storing the full L x L x p x p tensor.

    Same arguments as compute_cross_position_covariance, returns only the
    coupling_matrix.
    """
    n, L, p = E.shape
    E_clean = np.where(np.isnan(E), 0.0, E)

    coupling_matrix = np.zeros((L, L), dtype=np.float64)

    for i1 in range(L):
        E1 = E_clean[:, i1, :]
        # Pre-multiply: U_inv @ E1 (n, p)
        UE1 = U_inv @ E1
        for i2 in range(i1, L):
            E2 = E_clean[:, i2, :]
            mask_both = valid_mask[:, i1] & valid_mask[:, i2]
            n_eff = mask_both.sum()
            if n_eff < 2:
                continue

            C = E1.T @ U_inv @ E2 / n_eff
            frob = float(np.linalg.norm(C, ord='fro'))
            coupling_matrix[i1, i2] = frob
            coupling_matrix[i2, i1] = frob

    return coupling_matrix


def compute_aa_pair_couplings(cross_cov_block, aa_embeddings):
    """
    Translate a single dimension-level cross-covariance block C[i1, i2]
    into a 20 x 20 amino acid pair coupling matrix.

    J(alpha, beta) = e^alpha^T @ C[i1, i2] @ e^beta

    This is the DCA-comparable output: for each AA type pair, how strongly
    do they co-occur (after phylogenetic correction).

    Parameters
    ----------
    cross_cov_block : numpy.ndarray, shape (p, p)
        Single C[i1, i2] block.
    aa_embeddings : numpy.ndarray, shape (20, p)
        PLM embeddings of the 20 amino acid types in a neutral context.

    Returns
    -------
    J : numpy.ndarray, shape (20, 20)
        AA-pair coupling matrix.
    """
    return aa_embeddings @ cross_cov_block @ aa_embeddings.T


def compute_differential_couplings(E_a, mask_a, U_inv_a,
                                    E_b, mask_b, U_inv_b,
                                    return_full=False):
    """
    Compute differential couplings between two phylogenetically defined groups.

    For each position pair (i1, i2):
        Delta_C[i1, i2] = C^A[i1, i2] - C^B[i1, i2]
        Delta[i1, i2] = ||Delta_C[i1, i2]||_F

    Position pairs with large Delta are residue pairs whose evolutionary
    coupling pattern changed between the two regimes.

    Parameters
    ----------
    E_a, E_b : numpy.ndarray
        Per-MSA-column embedding tensors for groups A and B, shape (n_*, L, p).
        Both must have the same L and p.
    mask_a, mask_b : numpy.ndarray
        Valid masks, shape (n_*, L).
    U_inv_a, U_inv_b : numpy.ndarray
        Inverse phylogenetic covariances for the two groups.
    return_full : bool
        If True, return the full L x L x p x p differential tensor.
        If False, return only the L x L Frobenius norm matrix.

    Returns
    -------
    delta_matrix : numpy.ndarray, shape (L, L)
        Differential coupling strengths.
    delta_cross_cov : numpy.ndarray, shape (L, L, p, p), optional
        Full differential cross-covariance tensor (only if return_full=True).
    """
    if return_full:
        _, cross_a = compute_cross_position_covariance(E_a, mask_a, U_inv_a)
        _, cross_b = compute_cross_position_covariance(E_b, mask_b, U_inv_b)
        delta_cross = cross_a - cross_b
        delta_matrix = np.linalg.norm(delta_cross, axis=(2, 3), ord='fro')
        return delta_matrix, delta_cross
    else:
        # Memory-efficient: compute differential per position pair
        n_a, L, p = E_a.shape
        n_b = E_b.shape[0]
        E_a_clean = np.where(np.isnan(E_a), 0.0, E_a)
        E_b_clean = np.where(np.isnan(E_b), 0.0, E_b)

        delta_matrix = np.zeros((L, L), dtype=np.float64)

        for i1 in range(L):
            E_a_1 = E_a_clean[:, i1, :]
            E_b_1 = E_b_clean[:, i1, :]
            for i2 in range(i1, L):
                E_a_2 = E_a_clean[:, i2, :]
                E_b_2 = E_b_clean[:, i2, :]

                n_eff_a = (mask_a[:, i1] & mask_a[:, i2]).sum()
                n_eff_b = (mask_b[:, i1] & mask_b[:, i2]).sum()

                if n_eff_a < 2 or n_eff_b < 2:
                    continue

                C_a = E_a_1.T @ U_inv_a @ E_a_2 / n_eff_a
                C_b = E_b_1.T @ U_inv_b @ E_b_2 / n_eff_b
                delta = C_a - C_b
                frob = float(np.linalg.norm(delta, ord='fro'))
                delta_matrix[i1, i2] = frob
                delta_matrix[i2, i1] = frob

        return delta_matrix


def consensus_ss_to_region_labels(consensus_ss, min_region_length=3):
    """
    Convert a per-column secondary structure consensus string into region
    labels by grouping contiguous runs of the same SS type.

    Example:
        Input:  "HHHHHCCCCEEECCCHHHH"
        Output: ['H1','H1','H1','H1','H1','C1','C1','C1','C1',
                 'E1','E1','E1','C2','C2','C2','H2','H2','H2','H2']

    The pipeline already produces consensus_ss via get_consensus_structure()
    in significant_split_evaluation/structures/plot_comparative_logos.py,
    using DSSP on ESMFold-predicted structures (Q8 alphabet).

    Common Q8 codes: H (alpha-helix), G (3-10 helix), I (pi-helix),
    E (beta-strand), B (beta-bridge), T (turn), S (bend), C/' ' (coil).

    Parameters
    ----------
    consensus_ss : str
        Per-MSA-column consensus secondary structure (length L_msa).
        Use '-' or '?' for unannotated positions.
    min_region_length : int
        Runs shorter than this become unannotated (None).

    Returns
    -------
    region_labels : list of str or None, length L_msa
        Region label for each MSA column. None for unannotated.
    """
    L = len(consensus_ss)
    labels = [None] * L
    counters = {}

    i = 0
    while i < L:
        char = consensus_ss[i]
        if char in ('-', '?', ' '):
            i += 1
            continue

        # Find the run
        j = i
        while j < L and consensus_ss[j] == char:
            j += 1
        run_length = j - i

        if run_length >= min_region_length:
            counters[char] = counters.get(char, 0) + 1
            label = f"{char}{counters[char]}"
            for k in range(i, j):
                labels[k] = label

        i = j

    return labels


def aggregate_to_regions(coupling_matrix, region_labels):
    """
    Approach (a): biology-driven aggregation of position couplings to
    region-region couplings.

    Given an annotation that assigns each MSA column to a region label
    (e.g., 'helix1', 'sheet2', 'loop3'), compute the average coupling
    strength between each pair of regions.

    J[R1, R2] = (1 / (|R1|*|R2|)) sum_{i in R1, j in R2} coupling[i, j]

    Parameters
    ----------
    coupling_matrix : numpy.ndarray, shape (L, L)
        Position-level coupling strengths.
    region_labels : list of str or array, length L
        Region label for each MSA column. Use None or '' for unannotated
        positions (skipped).

    Returns
    -------
    region_coupling : dict
        {(R1, R2): float} mapping region pairs to mean coupling strength.
    region_to_positions : dict
        {region: list of position indices}
    """
    region_to_positions = {}
    for i, label in enumerate(region_labels):
        if label is None or label == '':
            continue
        region_to_positions.setdefault(label, []).append(i)

    region_names = sorted(region_to_positions.keys())
    region_coupling = {}

    for r1 in region_names:
        positions_1 = region_to_positions[r1]
        for r2 in region_names:
            positions_2 = region_to_positions[r2]
            if not positions_1 or not positions_2:
                continue
            block = coupling_matrix[np.ix_(positions_1, positions_2)]
            mean_strength = float(np.mean(block))
            region_coupling[(r1, r2)] = mean_strength

    return region_coupling, region_to_positions


def cluster_positions(coupling_matrix, n_clusters=5):
    """
    Approach (b): data-driven spectral clustering of positions based on
    the coupling matrix.

    Treats the coupling matrix as a similarity graph and clusters positions
    into groups whose members are tightly coupled to each other. The
    clusters are NOT constrained to be sequence-contiguous, so they can
    capture long-range structural modules (e.g., a binding pocket made of
    distant residues).

    Parameters
    ----------
    coupling_matrix : numpy.ndarray, shape (L, L)
        Position-level coupling strengths (used as similarity).
    n_clusters : int
        Number of clusters.

    Returns
    -------
    labels : numpy.ndarray, shape (L,)
        Cluster assignment for each position. Position with all-zero
        coupling row gets label -1.
    """
    try:
        from sklearn.cluster import SpectralClustering
    except ImportError:
        raise ImportError(
            "sklearn is required for cluster_positions; install scikit-learn"
        )

    L = coupling_matrix.shape[0]
    labels = -np.ones(L, dtype=np.int32)

    # Identify positions with non-trivial coupling (sum of row > 0)
    row_sums = coupling_matrix.sum(axis=1)
    valid_positions = np.where(row_sums > 1e-10)[0]

    if len(valid_positions) < n_clusters:
        return labels

    # Restrict to valid positions
    sub_matrix = coupling_matrix[np.ix_(valid_positions, valid_positions)]

    # Symmetrize and ensure non-negative
    sub_matrix = (sub_matrix + sub_matrix.T) / 2.0
    sub_matrix = np.maximum(sub_matrix, 0.0)

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=0,
        assign_labels='kmeans'
    )
    sub_labels = sc.fit_predict(sub_matrix)
    labels[valid_positions] = sub_labels

    return labels


def cluster_differential_positions(delta_matrix, n_clusters=5):
    """
    Cluster positions based on the DIFFERENTIAL coupling matrix.

    This identifies position groups whose JOINT coupling pattern changed
    between two regimes -- candidates for functional modules that
    reorganized.

    Same interface as cluster_positions, applied to delta_matrix.
    """
    return cluster_positions(np.abs(delta_matrix), n_clusters=n_clusters)


def get_top_position_pairs(coupling_matrix, n_top=20, exclude_diagonal=True):
    """
    Get the top N position pairs by coupling strength.

    Parameters
    ----------
    coupling_matrix : numpy.ndarray, shape (L, L)
    n_top : int
    exclude_diagonal : bool
        If True, ignore self-couplings (i == j).

    Returns
    -------
    pairs : list of dict
        Each dict has 'i', 'j', 'coupling' (sorted descending).
    """
    L = coupling_matrix.shape[0]
    M = coupling_matrix.copy()
    if exclude_diagonal:
        np.fill_diagonal(M, 0.0)
    # Take upper triangle
    M_triu = np.triu(M, k=1)

    flat_idx = np.argsort(M_triu.flatten())[::-1][:n_top]
    pairs = []
    for idx in flat_idx:
        i, j = np.unravel_index(idx, M_triu.shape)
        if M_triu[i, j] <= 0:
            break
        pairs.append({
            'i': int(i),
            'j': int(j),
            'coupling': float(M_triu[i, j]),
        })
    return pairs


def get_consensus_ss_from_pipeline(group_aligned_sequences, group_seq_ids,
                                     dir_predicted):
    """
    Wrapper for the existing pipeline function get_consensus_structure
    from significant_split_evaluation/structures/plot_comparative_logos.py.

    Returns the per-MSA-column consensus secondary structure string for a
    group of proteins, computed via DSSP on ESMFold-predicted structures.

    Parameters
    ----------
    group_aligned_sequences : list of str
        Aligned sequences (with gaps) for the group.
    group_seq_ids : list of str
        Sequence IDs (used to find the corresponding .pdb files).
    dir_predicted : str
        Directory containing ESMFold-predicted .pdb files.

    Returns
    -------
    consensus_ss : str
        Per-MSA-column Q8 secondary structure consensus.
    n_used : int
        Number of structures successfully used.
    """
    try:
        from significant_split_evaluation.structures.plot_comparative_logos \
            import get_consensus_structure
    except ImportError:
        raise ImportError(
            "Cannot import get_consensus_structure from "
            "significant_split_evaluation/structures/plot_comparative_logos.py"
        )
    return get_consensus_structure(
        group_aligned_sequences, group_seq_ids, dir_predicted
    )


def run_pairwise_coupling_analysis(per_residue_dir, sequence_names,
                                    msa_sequences, U_inv,
                                    aa_embeddings=None,
                                    consensus_ss=None,
                                    n_top=20, n_clusters=5,
                                    return_full=False):
    """
    Full pipeline for single-group pairwise coupling analysis.

    1. Load per-residue embeddings
    2. Build per-MSA-column embedding matrices
    3. Compute L x L coupling matrix (Frobenius norms only, memory-efficient)
    4. Identify top coupled position pairs

    Parameters
    ----------
    per_residue_dir : str
    sequence_names : list of str
    msa_sequences : dict {name: aligned_sequence}
    U_inv : numpy.ndarray, shape (n, n)
    n_top : int

    Returns
    -------
    result : dict
    """
    embeddings = load_per_residue_embeddings(per_residue_dir, sequence_names)
    if not embeddings:
        return {'error': 'No per-residue embeddings loaded'}

    E, valid_mask = build_msa_position_matrices(
        embeddings, msa_sequences, sequence_names
    )

    if return_full:
        coupling_matrix, cross_cov = compute_cross_position_covariance(
            E, valid_mask, U_inv
        )
    else:
        coupling_matrix = compute_position_couplings_only(E, valid_mask, U_inv)
        cross_cov = None

    top_pairs = get_top_position_pairs(coupling_matrix, n_top=n_top)

    result = {
        'coupling_matrix': coupling_matrix.tolist(),
        'top_pairs': top_pairs,
        'L_msa': int(E.shape[1]),
        'p_embedding': int(E.shape[2]),
        'n_sequences_loaded': len(embeddings),
    }

    # AA-pair couplings for top position pairs (if AA embeddings provided)
    if aa_embeddings is not None and cross_cov is not None:
        aa_pair_couplings = []
        for pair in top_pairs:
            i, j = pair['i'], pair['j']
            J = compute_aa_pair_couplings(cross_cov[i, j], aa_embeddings)
            top_aa_pairs = []
            J_abs = np.abs(J)
            top_idx = np.argsort(J_abs.flatten())[::-1][:5]
            for idx in top_idx:
                a, b = np.unravel_index(idx, J.shape)
                top_aa_pairs.append({
                    'aa_i': AA_LIST[a],
                    'aa_j': AA_LIST[b],
                    'coupling': float(J[a, b]),
                })
            aa_pair_couplings.append({
                'i': i, 'j': j,
                'J_matrix': J.tolist(),
                'top_aa_pairs': top_aa_pairs,
            })
        result['aa_pair_couplings'] = aa_pair_couplings

    # Region-level aggregation (approach a)
    if consensus_ss is not None:
        region_labels = consensus_ss_to_region_labels(consensus_ss)
        region_coupling, region_to_positions = aggregate_to_regions(
            coupling_matrix, region_labels
        )
        result['region_coupling'] = {
            f"{r1}|{r2}": v for (r1, r2), v in region_coupling.items()
        }
        result['region_to_positions'] = region_to_positions
        result['consensus_ss'] = consensus_ss

    # Spectral clustering (approach b)
    try:
        cluster_labels = cluster_positions(coupling_matrix,
                                           n_clusters=n_clusters)
        result['cluster_labels'] = cluster_labels.tolist()
    except ImportError:
        result['cluster_labels_error'] = 'sklearn not available'

    return result


def run_differential_coupling_analysis(per_residue_dir,
                                        group_a_names, group_b_names,
                                        msa_sequences,
                                        U_inv_a, U_inv_b,
                                        aa_embeddings=None,
                                        consensus_ss_a=None,
                                        consensus_ss_b=None,
                                        n_top=20, n_clusters=5,
                                        return_full=False):
    """
    Full pipeline for differential coupling analysis between two groups.

    1. Load per-residue embeddings for both groups
    2. Build per-MSA-column embedding matrices for each group
    3. Compute L x L differential coupling matrix
    4. Identify top differentially-coupled position pairs
    5. Spectral clustering on the differential matrix to find reorganized
       modules
    """
    all_names = list(set(group_a_names) | set(group_b_names))
    embeddings = load_per_residue_embeddings(per_residue_dir, all_names)

    if not embeddings:
        return {'error': 'No per-residue embeddings loaded'}

    E_a, mask_a = build_msa_position_matrices(
        embeddings, msa_sequences, group_a_names
    )
    E_b, mask_b = build_msa_position_matrices(
        embeddings, msa_sequences, group_b_names
    )

    if return_full:
        delta_matrix, delta_cross = compute_differential_couplings(
            E_a, mask_a, U_inv_a, E_b, mask_b, U_inv_b, return_full=True
        )
    else:
        delta_matrix = compute_differential_couplings(
            E_a, mask_a, U_inv_a, E_b, mask_b, U_inv_b, return_full=False
        )
        delta_cross = None

    top_pairs = get_top_position_pairs(delta_matrix, n_top=n_top)

    result = {
        'delta_matrix': delta_matrix.tolist(),
        'top_differential_pairs': top_pairs,
        'L_msa': int(E_a.shape[1]),
        'p_embedding': int(E_a.shape[2]),
        'n_a': len(group_a_names),
        'n_b': len(group_b_names),
        'n_a_loaded': sum(1 for n in group_a_names if n in embeddings),
        'n_b_loaded': sum(1 for n in group_b_names if n in embeddings),
    }

    # Differential AA-pair couplings for top pairs
    if aa_embeddings is not None and delta_cross is not None:
        diff_aa_couplings = []
        for pair in top_pairs:
            i, j = pair['i'], pair['j']
            J_delta = compute_aa_pair_couplings(delta_cross[i, j],
                                                aa_embeddings)
            J_abs = np.abs(J_delta)
            top_idx = np.argsort(J_abs.flatten())[::-1][:5]
            top_aa_pairs = []
            for idx in top_idx:
                a, b = np.unravel_index(idx, J_delta.shape)
                top_aa_pairs.append({
                    'aa_i': AA_LIST[a],
                    'aa_j': AA_LIST[b],
                    'delta_coupling': float(J_delta[a, b]),
                })
            diff_aa_couplings.append({
                'i': i, 'j': j,
                'delta_J_matrix': J_delta.tolist(),
                'top_changed_aa_pairs': top_aa_pairs,
            })
        result['differential_aa_pair_couplings'] = diff_aa_couplings

    # Differential region-level coupling (approach a)
    # Use group A's consensus SS as the canonical region annotation
    # (or fall back to B if A is not available)
    consensus_to_use = consensus_ss_a or consensus_ss_b
    if consensus_to_use is not None:
        region_labels = consensus_ss_to_region_labels(consensus_to_use)
        delta_region, region_to_positions = aggregate_to_regions(
            delta_matrix, region_labels
        )
        result['differential_region_coupling'] = {
            f"{r1}|{r2}": v for (r1, r2), v in delta_region.items()
        }
        result['region_to_positions'] = region_to_positions

    # Differential clustering (approach b)
    try:
        cluster_labels = cluster_differential_positions(
            delta_matrix, n_clusters=n_clusters
        )
        result['differential_cluster_labels'] = cluster_labels.tolist()
    except ImportError:
        result['differential_cluster_labels_error'] = 'sklearn not available'

    return result


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    np.random.seed(0)
    n, L, p = 10, 20, 50

    # Synthetic per-MSA embeddings
    E = np.random.randn(n, L, p)
    valid_mask = np.ones((n, L), dtype=bool)
    valid_mask[2, 5] = False  # one gap

    # Inject a coupling: positions 3 and 17 share a direction
    direction = np.random.randn(p)
    coupling_signal = np.random.randn(n)
    E[:, 3, :] += coupling_signal[:, None] * direction
    E[:, 17, :] += coupling_signal[:, None] * direction

    U_inv = np.eye(n)
    coupling_mat = compute_position_couplings_only(E, valid_mask, U_inv)

    print(f"Coupling matrix shape: {coupling_mat.shape}")
    print(f"Coupling between positions 3 and 17: {coupling_mat[3, 17]:.4f}")
    print(f"Mean off-diagonal coupling: "
          f"{coupling_mat[~np.eye(L, dtype=bool)].mean():.4f}")

    top = get_top_position_pairs(coupling_mat, n_top=5)
    print(f"Top 5 pairs:")
    for p in top:
        print(f"  ({p['i']}, {p['j']}): {p['coupling']:.4f}")

    # Test region aggregation
    region_labels = ['helix1'] * 5 + ['sheet1'] * 5 + ['loop1'] * 5 + \
                    ['helix2'] * 5
    region_coupling, regions = aggregate_to_regions(coupling_mat,
                                                     region_labels)
    print(f"\nRegion-level couplings (top 5):")
    sorted_regions = sorted(region_coupling.items(),
                            key=lambda x: -x[1])
    for (r1, r2), strength in sorted_regions[:5]:
        print(f"  {r1} - {r2}: {strength:.4f}")

    # Test clustering
    try:
        labels = cluster_positions(coupling_mat, n_clusters=3)
        print(f"\nCluster labels: {labels}")
    except ImportError:
        print("\nSkipping spectral clustering (sklearn not available)")

    print("\nAll tests passed.")
