#!/usr/bin/env python3
"""
KAVERET Simulation Validation Framework.

Generic validation for any phylogenetic split test that produces
(statistic, p_value).  Tests are pluggable via a simple callable
interface, so the same simulation machinery validates the covariance
LRT, phylogenetic ANOVA, Hotelling T^2, or any future test.

Three simulation studies:
  1. Type I error (size): data under H0 — test should reject at ~alpha
  2. Power: data under H1 (mean shift and/or covariance shift)
  3. Robustness: OU-generated data tested with BM-based method

Example usage:
    # Quick Type I check with covariance LRT
    python simulate_test_validation.py --test cov_lrt --n 100 --p 10 --reps 200

    # Mean ANOVA validation
    python simulate_test_validation.py --test mean_anova --n 100 --p 10 --reps 200

    # Power curve for mean shift detected by ANOVA
    python simulate_test_validation.py --test mean_anova --study power \\
        --shift_type mean --n 200 --p 20 --reps 200 --plot

    # Power curve for covariance shift detected by LRT
    python simulate_test_validation.py --test cov_lrt --study power \\
        --shift_type covariance --n 200 --p 20 --reps 200 --plot

    # All studies
    python simulate_test_validation.py --test cov_lrt --study all \\
        --n 100 --p 10 --reps 200 --plot

    # Grid of (n, p) values
    python simulate_test_validation.py --test cov_lrt --study grid --reps 200 --plot
"""

import os
import sys
import inspect
import argparse
import json
import time
import math
import numpy as np
import torch
from typing import Callable, Tuple, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_split_options.lrt_statistics import (
    simulate_null_data,
    compute_mle_and_lrt,
    compute_mean_lrt,
    compute_gls_operators,
    robust_cholesky,
    add_jitter,
)

# Optional: phylogenetic ANOVA (requires ete3)
try:
    from evaluate_split_options.phylogenetic_anova import phylogenetic_anova_rrpp
    HAS_ANOVA = True
except ImportError:
    HAS_ANOVA = False


# ===================================================================
# Type alias
# ===================================================================
# A split-test callable:  (X, U, idx_A, idx_B, **kw) -> (stat, pval)
SplitTestFn = Callable[..., Tuple[float, float]]


# ===================================================================
# Test function wrappers
# ===================================================================

def cov_lrt_bootstrap_test(X, U, idx_A, idx_B, n_bootstrap=500):
    """
    Covariance equality LRT with parametric bootstrap p-value.
    Tests H0: V_A = V_B under the matrix-normal model.
    """
    n, p = X.shape
    X_A, X_B = X[idx_A], X[idx_B]
    U_A = U[idx_A][:, idx_A]
    U_B = U[idx_B][:, idx_B]
    n_A, n_B = len(idx_A), len(idx_B)

    _, P_A, _, _ = compute_gls_operators(U_A)
    _, P_B, _, _ = compute_gls_operators(U_B)

    lambda_obs = compute_mle_and_lrt(X_A, X_B, P_A, P_B, n_A, n_B).item()

    # Bootstrap under H0: fit shared model, simulate, recompute
    _, P_full, t1, t2 = compute_gls_operators(U)
    mu_hat = (t1 @ t2 @ X).squeeze()
    V_hat = (X.T @ P_full @ X) / n
    L_U = robust_cholesky((U + U.T) / 2.0)
    L_V_hat = robust_cholesky((V_hat + V_hat.T) / 2.0)

    null_lambdas = []
    for _ in range(n_bootstrap):
        X_sim = simulate_null_data(n, p, mu_hat, L_U, L_V_hat)
        lam = compute_mle_and_lrt(
            X_sim[idx_A], X_sim[idx_B], P_A, P_B, n_A, n_B
        ).item()
        null_lambdas.append(lam)

    p_value = (np.sum(np.array(null_lambdas) >= lambda_obs) + 1) / (n_bootstrap + 1)
    return lambda_obs, p_value


def mean_lrt_bootstrap_test(X, U, idx_A, idx_B, n_bootstrap=500):
    """
    Mean equality LRT with parametric bootstrap p-value.
    Tests H0: mu_A = mu_B under the matrix-normal model (shared V).
    """
    n, p = X.shape
    U_inv = torch.linalg.pinv(U)

    lambda_obs = compute_mean_lrt(X, U_inv, idx_A, idx_B).item()

    # Bootstrap under H0: shared mean, shared V
    _, P_full, t1, t2 = compute_gls_operators(U)
    mu_hat = (t1 @ t2 @ X).squeeze()
    V_hat = (X.T @ P_full @ X) / n
    L_U = robust_cholesky((U + U.T) / 2.0)
    L_V_hat = robust_cholesky((V_hat + V_hat.T) / 2.0)

    null_lambdas = []
    for _ in range(n_bootstrap):
        X_sim = simulate_null_data(n, p, mu_hat, L_U, L_V_hat)
        U_inv_sim = U_inv  # Same tree structure
        lam = compute_mean_lrt(X_sim, U_inv_sim, idx_A, idx_B).item()
        null_lambdas.append(lam)

    p_value = (np.sum(np.array(null_lambdas) >= lambda_obs) + 1) / (n_bootstrap + 1)
    return lambda_obs, p_value


def mean_anova_rrpp_test(X, U, idx_A, idx_B, n_permutations=999):
    """
    Phylogenetic ANOVA with RRPP permutation p-value.
    Tests H0: mu_A = mu_B under phylogenetic non-independence.
    """
    if not HAS_ANOVA:
        raise ImportError(
            "phylogenetic_anova_rrpp not available. "
            "Install ete3 and check evaluate_split_options/phylogenetic_anova.py."
        )
    F_obs, p_val = phylogenetic_anova_rrpp(X, U, idx_A, idx_B, n_permutations)
    return F_obs, p_val


# ===================================================================
# Test registry
# ===================================================================

TEST_REGISTRY: Dict[str, SplitTestFn] = {
    "cov_lrt": cov_lrt_bootstrap_test,
    "mean_lrt": mean_lrt_bootstrap_test,
    "mean_anova": mean_anova_rrpp_test,
}

# Default kwargs for each test (used when CLI doesn't override)
TEST_DEFAULT_KWARGS: Dict[str, Dict[str, Any]] = {
    "cov_lrt": {"n_bootstrap": 500},
    "mean_lrt": {"n_bootstrap": 500},
    "mean_anova": {"n_permutations": 999},
}

# Natural shift type for power study per test
TEST_DEFAULT_SHIFT: Dict[str, str] = {
    "cov_lrt": "covariance",
    "mean_lrt": "mean",
    "mean_anova": "mean",
}


def _filter_kwargs(fn, kwargs):
    """Pass only kwargs that *fn* actually accepts (avoids TypeError)."""
    sig = inspect.signature(fn)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return valid


# ===================================================================
# Tree generation
# ===================================================================

def generate_balanced_tree_covariance(n, total_height=1.0):
    """
    Phylogenetic covariance U from a balanced binary tree.
    U_ij = shared branch length from root to MRCA(i, j).
    """
    n_levels = max(1, math.ceil(math.log2(n)))
    edge_len = total_height / n_levels

    U = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                U[i, j] = total_height
            else:
                xor = i ^ j
                mrca_level = n_levels - int(math.floor(math.log2(xor))) - 1
                mrca_level = max(0, mrca_level)
                U[i, j] = mrca_level * edge_len
    return U


def generate_random_tree_covariance(n, total_height=1.0, seed=None):
    """U from a random coalescent-like ultrametric tree."""
    rng = np.random.RandomState(seed)
    merge_times = np.sort(rng.uniform(0, total_height, size=n - 1))
    merge_times = np.concatenate([[0.0], merge_times, [total_height]])

    groups = [[i] for i in range(n)]
    mrca_time = np.zeros((n, n))

    for step in range(n - 1):
        t = merge_times[step + 1]
        if len(groups) < 2:
            break
        idx = rng.choice(len(groups), size=2, replace=False)
        g1, g2 = groups[idx[0]], groups[idx[1]]
        for i in g1:
            for j in g2:
                mrca_time[i, j] = t
                mrca_time[j, i] = t
        merged = g1 + g2
        groups = [g for k, g in enumerate(groups) if k not in idx]
        groups.append(merged)

    U = torch.tensor(mrca_time, dtype=torch.float64)
    for i in range(n):
        U[i, i] = total_height
    return U


def generate_split_indices(n, frac_a=0.5):
    """Split n taxa into two groups.  Returns (idx_A, idx_B)."""
    n_a = max(2, int(n * frac_a))
    n_b = n - n_a
    if n_b < 2:
        n_a = n - 2
        n_b = 2
    return list(range(n_a)), list(range(n_a, n))


def load_real_tree_covariance(tree_path, cov_path=None):
    """
    Load a real phylogenetic tree and return (U, n, idx_A, idx_B).

    If cov_path is given, loads the covariance matrix from CSV.
    Otherwise, computes it from the tree file using the project's utility.

    The split is determined by the deepest internal node that gives
    the most balanced bipartition (closest to 50/50).
    """
    import pandas as pd

    try:
        from ete3 import Tree as EteTree
    except ImportError:
        raise ImportError("ete3 is required for --tree real")

    # Load tree
    try:
        tree = EteTree(tree_path, format=1)
    except Exception:
        tree = EteTree(tree_path, format=0)

    all_leaves = tree.get_leaf_names()
    n = len(all_leaves)
    leaf_to_idx = {name: i for i, name in enumerate(all_leaves)}

    # Load or compute covariance matrix
    if cov_path and os.path.exists(cov_path):
        df = pd.read_csv(cov_path, index_col=0)
        # Reorder to match tree leaf order
        df = df.loc[all_leaves, all_leaves]
        U = torch.tensor(df.values, dtype=torch.float64)
    else:
        # Compute from tree: U[i,j] = shared path from root to MRCA(i,j)
        U = torch.zeros(n, n, dtype=torch.float64)
        # Cache ancestor distances for efficiency
        root = tree.get_tree_root()
        leaf_nodes = {leaf.name: leaf for leaf in tree.get_leaves()}

        for i, name_i in enumerate(all_leaves):
            node_i = leaf_nodes[name_i]
            U[i, i] = node_i.get_distance(root)
            for j in range(i + 1, n):
                name_j = all_leaves[j]
                node_j = leaf_nodes[name_j]
                mrca = tree.get_common_ancestor(node_i, node_j)
                shared = mrca.get_distance(root)
                U[i, j] = shared
                U[j, i] = shared

    # Find the most balanced split from the tree topology
    best_node = None
    best_balance = 0.0

    for node in tree.traverse("postorder"):
        if node.is_leaf() or node.is_root():
            continue
        clade_size = len(node.get_leaf_names())
        balance = min(clade_size, n - clade_size) / n
        if balance > best_balance:
            best_balance = balance
            best_node = node

    if best_node is not None:
        clade_leaves = set(best_node.get_leaf_names())
        idx_A = [leaf_to_idx[name] for name in all_leaves if name in clade_leaves]
        idx_B = [leaf_to_idx[name] for name in all_leaves if name not in clade_leaves]
    else:
        idx_A, idx_B = generate_split_indices(n)

    print(f"  Real tree: {n} taxa, split {len(idx_A)}/{len(idx_B)} "
          f"(balance={best_balance:.2f})")

    return U, n, idx_A, idx_B


# ===================================================================
# Covariance / mean generation & perturbation
# ===================================================================

def generate_random_V(p, condition_number=10.0, seed=None):
    """Random PD matrix V with controlled condition number."""
    rng = np.random.RandomState(seed)
    A = rng.randn(p, p)
    Q, _ = np.linalg.qr(A)
    evals = np.linspace(1.0, condition_number, p)
    V = Q @ np.diag(evals) @ Q.T
    V = (V + V.T) / 2.0
    return torch.tensor(V, dtype=torch.float64)


def perturb_V(V, delta_scale, mode="eigenvalue", seed=None):
    """
    Create V_B from V_A with controlled perturbation.

    Modes:
      eigenvalue : scale top eigenvalues by (1 + delta_scale)
      additive   : add delta_scale * random PSD matrix
      rotation   : rotate eigenvectors by delta_scale radians
    """
    rng = np.random.RandomState(seed)
    V_np = V.numpy().copy()
    evals, evecs = np.linalg.eigh(V_np)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    if mode == "eigenvalue":
        n_perturb = max(1, len(evals) // 2)
        evals_new = evals.copy()
        evals_new[:n_perturb] *= (1.0 + delta_scale)
        V_B = evecs @ np.diag(evals_new) @ evecs.T

    elif mode == "additive":
        p = V_np.shape[0]
        A = rng.randn(p, p) * delta_scale
        Delta = A @ A.T / p
        V_B = V_np + Delta

    elif mode == "rotation":
        p = V_np.shape[0]
        if p >= 2:
            G = np.eye(p)
            G[0, 0] = np.cos(delta_scale)
            G[0, 1] = -np.sin(delta_scale)
            G[1, 0] = np.sin(delta_scale)
            G[1, 1] = np.cos(delta_scale)
            evecs_rot = evecs @ G
            V_B = evecs_rot @ np.diag(evals) @ evecs_rot.T
        else:
            V_B = V_np
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    V_B = (V_B + V_B.T) / 2.0
    return torch.tensor(V_B, dtype=torch.float64)


def generate_mean_perturbation(p, delta_scale, seed=None):
    """
    Random mean-shift vector with ||delta_mu||_2 = delta_scale.
    """
    rng = np.random.RandomState(seed)
    direction = rng.randn(p)
    direction = direction / np.linalg.norm(direction)
    return torch.tensor(direction * delta_scale, dtype=torch.float64)


# ===================================================================
# Data simulation helpers
# ===================================================================

def simulate_h0_data(n, p, mu, V, U):
    """Simulate X ~ MN(1 mu^T, U, V) under H0 (shared parameters)."""
    L_U = robust_cholesky((U + U.T) / 2.0)
    L_V = robust_cholesky((V + V.T) / 2.0)
    return simulate_null_data(n, p, mu, L_U, L_V)


def simulate_h1_data(n, p, mu_A, mu_B, V_A, V_B, U, idx_A, idx_B):
    """
    Simulate data under H1 with per-clade parameters.

    For covariance shift only : mu_A == mu_B,  V_A != V_B
    For mean shift only       : mu_A != mu_B,  V_A == V_B
    For both                  : different mu AND V
    """
    U_A = U[idx_A][:, idx_A]
    U_B = U[idx_B][:, idx_B]
    n_A, n_B = len(idx_A), len(idx_B)

    L_U_A = robust_cholesky((U_A + U_A.T) / 2.0)
    L_U_B = robust_cholesky((U_B + U_B.T) / 2.0)
    L_V_A = robust_cholesky((V_A + V_A.T) / 2.0)
    L_V_B = robust_cholesky((V_B + V_B.T) / 2.0)

    X_A = simulate_null_data(n_A, p, mu_A, L_U_A, L_V_A)
    X_B = simulate_null_data(n_B, p, mu_B, L_U_B, L_V_B)

    X = torch.zeros(n, p, dtype=torch.float64)
    X[idx_A] = X_A
    X[idx_B] = X_B
    return X


# ===================================================================
# OU covariance helper
# ===================================================================

def bm_to_ou_covariance(U_bm, alpha_ou):
    """Convert BM covariance to OU: U_OU_ij = exp(-alpha * d_ij)."""
    n = U_bm.shape[0]
    height = U_bm.diagonal()
    d = torch.zeros_like(U_bm)
    for i in range(n):
        for j in range(n):
            d[i, j] = height[i] + height[j] - 2 * U_bm[i, j]
    return torch.exp(-alpha_ou * d)


# ===================================================================
# Generic replicate runners
# ===================================================================

def run_single_h0_replicate(test_fn, n, p, mu, V, U,
                            idx_A, idx_B, **test_kwargs):
    """
    One replicate under H0 (shared parameters).
    Returns (statistic, p_value).
    """
    X = simulate_h0_data(n, p, mu, V, U)
    kw = _filter_kwargs(test_fn, test_kwargs)
    return test_fn(X, U, idx_A, idx_B, **kw)


def run_single_h1_replicate(test_fn, n, p, mu_A, mu_B, V_A, V_B, U,
                            idx_A, idx_B, **test_kwargs):
    """
    One replicate under H1 (per-clade parameters).
    Returns (statistic, p_value).
    """
    X = simulate_h1_data(n, p, mu_A, mu_B, V_A, V_B, U, idx_A, idx_B)
    kw = _filter_kwargs(test_fn, test_kwargs)
    return test_fn(X, U, idx_A, idx_B, **kw)


# ===================================================================
# Study 1: Type I error
# ===================================================================

def _setup_tree(tree_type, n, seed, tree_path=None, cov_path=None):
    """Create or load the phylogenetic covariance and split indices."""
    if tree_type == "real":
        if tree_path is None:
            raise ValueError("--tree_path is required when --tree real")
        U, n_actual, idx_A, idx_B = load_real_tree_covariance(tree_path, cov_path)
        return U, n_actual, idx_A, idx_B
    elif tree_type == "balanced":
        U = generate_balanced_tree_covariance(n)
    else:
        U = generate_random_tree_covariance(n, seed=seed)
    idx_A, idx_B = generate_split_indices(n, frac_a=0.5)
    return U, n, idx_A, idx_B


def study_type_i_error(test_fn, test_name, n, p, n_reps, alpha=0.05,
                       tree_type="balanced", seed=42,
                       tree_path=None, cov_path=None, **test_kwargs):
    """
    Simulate data under H0 and check empirical rejection rate.
    Should be close to the nominal alpha.
    """
    print(f"\n{'='*60}")
    print(f"  STUDY 1: Type I Error (Size)  [{test_name}]")
    print(f"{'='*60}")
    print(f"  n={n}, p={p}, reps={n_reps}, alpha={alpha}, tree={tree_type}")
    print(f"  test kwargs: {test_kwargs}")

    U, n, idx_A, idx_B = _setup_tree(tree_type, n, seed, tree_path, cov_path)
    V = generate_random_V(p, condition_number=5.0, seed=seed)
    mu = torch.zeros(p, dtype=torch.float64)

    rejections = 0
    p_values = []
    statistics = []

    for rep in range(n_reps):
        if (rep + 1) % 50 == 0 or rep == 0:
            print(f"    Replicate {rep+1}/{n_reps}...")
        stat, pval = run_single_h0_replicate(
            test_fn, n, p, mu, V, U, idx_A, idx_B, **test_kwargs
        )
        p_values.append(pval)
        statistics.append(stat)
        if pval <= alpha:
            rejections += 1

    empirical_alpha = rejections / n_reps
    se = np.sqrt(alpha * (1 - alpha) / n_reps)
    ci_lo, ci_hi = alpha - 1.96 * se, alpha + 1.96 * se
    in_ci = ci_lo <= empirical_alpha <= ci_hi

    print(f"\n  Results:")
    print(f"    Empirical rejection rate: {empirical_alpha:.4f}")
    print(f"    Nominal alpha:            {alpha:.4f}")
    print(f"    95% CI for nominal:       [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Within CI:                {'YES' if in_ci else 'NO'}")
    print(f"    Mean statistic:           {np.mean(statistics):.2f}")
    print(f"    Median p-value:           {np.median(p_values):.4f}")

    return {
        "study": "type_i_error", "test": test_name,
        "n": n, "p": p, "n_reps": n_reps,
        "alpha": alpha, "tree_type": tree_type,
        "empirical_alpha": empirical_alpha,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "within_ci": in_ci,
        "mean_statistic": float(np.mean(statistics)),
        "median_pvalue": float(np.median(p_values)),
        "p_values": [float(x) for x in p_values],
        "statistics": [float(x) for x in statistics],
    }


# ===================================================================
# Study 2: Power
# ===================================================================

def study_power(test_fn, test_name, n, p, n_reps, alpha=0.05,
                shift_type="covariance",
                delta_scales=None, perturbation_mode="eigenvalue",
                tree_type="balanced", seed=42,
                tree_path=None, cov_path=None, **test_kwargs):
    """
    Simulate data under H1 with increasing effect size.
    Supports both mean shifts and covariance shifts.
    """
    if delta_scales is None:
        delta_scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    print(f"\n{'='*60}")
    print(f"  STUDY 2: Power Analysis  [{test_name}]")
    print(f"{'='*60}")
    print(f"  n={n}, p={p}, reps={n_reps}, shift={shift_type}")
    print(f"  perturbation: {perturbation_mode}")
    print(f"  delta_scales: {delta_scales}")

    U, n, idx_A, idx_B = _setup_tree(tree_type, n, seed, tree_path, cov_path)
    V = generate_random_V(p, condition_number=5.0, seed=seed)
    mu = torch.zeros(p, dtype=torch.float64)

    power_curve = []

    for delta in delta_scales:
        # Build per-clade parameters depending on shift type
        if shift_type == "covariance":
            mu_A, mu_B = mu, mu
            V_A = V
            V_B = perturb_V(V, delta, mode=perturbation_mode, seed=seed + 1)
            effect_label = f"||V_A - V_B||_F = {torch.norm(V_A - V_B, p='fro').item():.3f}"
        elif shift_type == "mean":
            delta_mu = generate_mean_perturbation(p, delta, seed=seed + 1)
            mu_A = mu
            mu_B = mu + delta_mu
            V_A, V_B = V, V
            effect_label = f"||mu_A - mu_B||_2 = {torch.norm(delta_mu).item():.3f}"
        elif shift_type == "both":
            delta_mu = generate_mean_perturbation(p, delta, seed=seed + 1)
            mu_A = mu
            mu_B = mu + delta_mu
            V_A = V
            V_B = perturb_V(V, delta, mode=perturbation_mode, seed=seed + 2)
            effect_label = (
                f"||dmu||={torch.norm(delta_mu).item():.3f}, "
                f"||dV||_F={torch.norm(V_A - V_B, p='fro').item():.3f}"
            )
        else:
            raise ValueError(f"Unknown shift_type: {shift_type}")

        print(f"\n  delta_scale={delta}, {effect_label}")

        rejections = 0
        p_values = []

        for rep in range(n_reps):
            if (rep + 1) % 50 == 0 or rep == 0:
                print(f"    Replicate {rep+1}/{n_reps}...")
            _, pval = run_single_h1_replicate(
                test_fn, n, p, mu_A, mu_B, V_A, V_B, U,
                idx_A, idx_B, **test_kwargs
            )
            p_values.append(pval)
            if pval <= alpha:
                rejections += 1

        power = rejections / n_reps
        print(f"    Power: {power:.4f}")

        power_curve.append({
            "delta_scale": delta,
            "effect_label": effect_label,
            "power": power,
            "p_values": [float(x) for x in p_values],
        })

    return {
        "study": "power", "test": test_name,
        "n": n, "p": p, "n_reps": n_reps,
        "alpha": alpha, "shift_type": shift_type,
        "perturbation_mode": perturbation_mode,
        "tree_type": tree_type,
        "power_curve": power_curve,
    }


# ===================================================================
# Study 3: Robustness (OU data, BM test)
# ===================================================================

def study_robustness(test_fn, test_name, n, p, n_reps, alpha=0.05,
                     alpha_ou_values=None, tree_type="balanced", seed=42,
                     tree_path=None, cov_path=None, **test_kwargs):
    """
    Generate data under OU but test with BM-based method.
    Checks whether Type I error is still controlled.
    """
    if alpha_ou_values is None:
        alpha_ou_values = [0.5, 1.0, 2.0, 5.0]

    print(f"\n{'='*60}")
    print(f"  STUDY 3: Robustness (OU data, BM test)  [{test_name}]")
    print(f"{'='*60}")
    print(f"  n={n}, p={p}, reps={n_reps}")
    print(f"  OU alpha values: {alpha_ou_values}")

    U_bm, n, idx_A, idx_B = _setup_tree(tree_type, n, seed, tree_path, cov_path)
    V = generate_random_V(p, condition_number=5.0, seed=seed)
    mu = torch.zeros(p, dtype=torch.float64)

    robustness_results = []

    for alpha_ou in alpha_ou_values:
        U_ou = bm_to_ou_covariance(U_bm, alpha_ou)
        U_ou = add_jitter(U_ou, jitter=1e-6)

        print(f"\n  OU alpha={alpha_ou}")

        rejections = 0
        p_values = []

        for rep in range(n_reps):
            if (rep + 1) % 50 == 0 or rep == 0:
                print(f"    Replicate {rep+1}/{n_reps}...")

            # Generate with OU covariance, but TEST with BM covariance
            X = simulate_h0_data(n, p, mu, V, U_ou)
            kw = _filter_kwargs(test_fn, test_kwargs)
            stat, pval = test_fn(X, U_bm, idx_A, idx_B, **kw)

            p_values.append(pval)
            if pval <= alpha:
                rejections += 1

        empirical_alpha = rejections / n_reps
        print(f"    Empirical rejection rate: {empirical_alpha:.4f} "
              f"(nominal: {alpha:.4f})")

        robustness_results.append({
            "alpha_ou": alpha_ou,
            "empirical_alpha": empirical_alpha,
            "p_values": [float(x) for x in p_values],
        })

    return {
        "study": "robustness", "test": test_name,
        "n": n, "p": p, "n_reps": n_reps,
        "alpha": alpha, "tree_type": tree_type,
        "results": robustness_results,
    }


# ===================================================================
# Study 1b: Type I error across (n, p) grid
# ===================================================================

def study_type_i_grid(test_fn, test_name, configs, n_reps, n_bootstrap_or_perm,
                      alpha=0.05, tree_type="balanced", seed=42,
                      tree_path=None, cov_path=None, **test_kwargs):
    """Run Type I error study across multiple (n, p) configurations."""
    print(f"\n{'='*60}")
    print(f"  STUDY 1b: Type I Error Grid  [{test_name}]")
    print(f"{'='*60}")
    print(f"  Configs: {configs}")

    grid_results = []
    for n, p in configs:
        print(f"\n  --- n={n}, p={p} ---")
        result = study_type_i_error(
            test_fn, test_name, n, p, n_reps, alpha,
            tree_type, seed, tree_path=tree_path, cov_path=cov_path,
            **test_kwargs
        )
        grid_results.append({
            "n": n, "p": p,
            "empirical_alpha": result["empirical_alpha"],
            "within_ci": result["within_ci"],
            "median_pvalue": result["median_pvalue"],
        })

    print(f"\n  Grid Summary:")
    print(f"  {'n':>6} {'p':>6} {'emp_alpha':>12} {'within_CI':>12} {'med_pval':>12}")
    for r in grid_results:
        print(f"  {r['n']:>6} {r['p']:>6} {r['empirical_alpha']:>12.4f} "
              f"{'YES' if r['within_ci'] else 'NO':>12} {r['median_pvalue']:>12.4f}")

    return grid_results


# ===================================================================
# Plotting
# ===================================================================

def plot_results(results, output_dir):
    """Generate diagnostic plots for simulation results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for result in results:
        study = result["study"]
        test_name = result.get("test", "unknown")

        if study == "type_i_error":
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # P-value histogram
            ax = axes[0]
            pvals = result["p_values"]
            ax.hist(pvals, bins=20, density=True, alpha=0.7, edgecolor="black")
            ax.axhline(y=1.0, color="red", linestyle="--", label="Uniform(0,1)")
            ax.set_xlabel("p-value")
            ax.set_ylabel("Density")
            ax.set_title(
                f"P-value Distribution under H0  [{test_name}]\n"
                f"n={result['n']}, p={result['p']}"
            )
            ax.legend()

            # QQ plot
            ax = axes[1]
            sorted_pvals = np.sort(pvals)
            expected = np.linspace(0, 1, len(sorted_pvals) + 2)[1:-1]
            ax.scatter(expected, sorted_pvals, s=10, alpha=0.5)
            ax.plot([0, 1], [0, 1], "r--", label="y=x")
            ax.set_xlabel("Expected (Uniform)")
            ax.set_ylabel("Observed p-value")
            ax.set_title(
                f"QQ Plot  [{test_name}]\n"
                f"Empirical alpha={result['empirical_alpha']:.3f}"
            )
            ax.legend()

            plt.tight_layout()
            path = os.path.join(output_dir, f"type_i_error_{test_name}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved: {path}")

        elif study == "power":
            fig, ax = plt.subplots(figsize=(8, 5))
            deltas = [pc["delta_scale"] for pc in result["power_curve"]]
            powers = [pc["power"] for pc in result["power_curve"]]
            labels = [pc["effect_label"] for pc in result["power_curve"]]

            ax.plot(deltas, powers, "bo-", linewidth=2, markersize=8)
            ax.axhline(
                y=result["alpha"], color="red", linestyle="--",
                label=f"alpha={result['alpha']}"
            )
            ax.set_xlabel("Delta Scale")
            ax.set_ylabel("Power (rejection rate)")
            ax.set_title(
                f"Power Curve  [{test_name}]\n"
                f"n={result['n']}, p={result['p']}, "
                f"shift={result['shift_type']}"
            )
            ax.legend()
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

            for d, pw, lab in zip(deltas, powers, labels):
                ax.annotate(
                    lab.split("=")[-1].strip(),
                    (d, pw), textcoords="offset points",
                    xytext=(5, 10), fontsize=7, alpha=0.7,
                )

            plt.tight_layout()
            path = os.path.join(output_dir, f"power_{test_name}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved: {path}")

        elif study == "robustness":
            fig, ax = plt.subplots(figsize=(8, 5))
            alphas_ou = [r["alpha_ou"] for r in result["results"]]
            emp_alphas = [r["empirical_alpha"] for r in result["results"]]

            ax.bar(
                range(len(alphas_ou)), emp_alphas,
                tick_label=[str(a) for a in alphas_ou],
                alpha=0.7, edgecolor="black",
            )
            ax.axhline(
                y=result["alpha"], color="red", linestyle="--",
                label=f"Nominal alpha={result['alpha']}",
            )
            se = np.sqrt(result["alpha"] * (1 - result["alpha"]) / result["n_reps"])
            ax.axhspan(
                result["alpha"] - 1.96 * se,
                result["alpha"] + 1.96 * se,
                alpha=0.15, color="red", label="95% CI",
            )
            ax.set_xlabel("OU alpha (selection strength)")
            ax.set_ylabel("Empirical rejection rate")
            ax.set_title(
                f"Robustness: OU Data with BM Test  [{test_name}]\n"
                f"n={result['n']}, p={result['p']}"
            )
            ax.legend()
            ax.set_ylim(0, max(0.2, max(emp_alphas) * 1.3))

            plt.tight_layout()
            path = os.path.join(output_dir, f"robustness_{test_name}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved: {path}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KAVERET Simulation Validation Framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick Type I check with covariance LRT
  python simulate_test_validation.py --test cov_lrt --n 60 --p 10 --reps 100

  # Full Type I error study for mean ANOVA
  python simulate_test_validation.py --test mean_anova --study size \\
      --n 200 --p 20 --reps 500

  # Power curve (covariance shift detected by LRT)
  python simulate_test_validation.py --test cov_lrt --study power \\
      --n 200 --p 20 --reps 200

  # Power curve (mean shift detected by ANOVA)
  python simulate_test_validation.py --test mean_anova --study power \\
      --shift_type mean --n 200 --p 20 --reps 200

  # Robustness (OU misspecification)
  python simulate_test_validation.py --test cov_lrt --study robustness \\
      --n 100 --p 10 --reps 200

  # All studies
  python simulate_test_validation.py --test cov_lrt --study all \\
      --n 100 --p 10 --reps 200 --plot

  # Grid of (n, p) values
  python simulate_test_validation.py --test cov_lrt --study grid --reps 200 --plot
        """,
    )

    parser.add_argument(
        "--test", type=str, default="cov_lrt",
        choices=list(TEST_REGISTRY.keys()),
        help="Which test to validate (default: cov_lrt)",
    )
    parser.add_argument(
        "--study", type=str, default="size",
        choices=["size", "power", "robustness", "all", "grid"],
        help="Which study to run (default: size)",
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Number of taxa (default: 100)")
    parser.add_argument("--p", type=int, default=10,
                        help="Embedding dimensions after PCA (default: 10)")
    parser.add_argument("--reps", type=int, default=200,
                        help="Monte Carlo replicates (default: 200)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--tree", type=str, default="balanced",
                        choices=["balanced", "random", "real"],
                        help="Tree type (default: balanced)")
    parser.add_argument("--family", type=str, default=None,
                        help="Pfam family (e.g. PF00076). Auto-resolves tree/cov paths from config. Use with --tree real.")
    parser.add_argument("--tree_path", type=str, default=None,
                        help="Path to .tree file (required when --tree real, unless --family is given)")
    parser.add_argument("--cov_path", type=str, default=None,
                        help="Path to covariance CSV (optional with --tree real, computed from tree if omitted)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")

    # Test-specific parameters
    parser.add_argument("--n_bootstrap", type=int, default=None,
                        help="Bootstrap replicates for cov_lrt (default: 500)")
    parser.add_argument("--n_permutations", type=int, default=None,
                        help="Permutation replicates for mean_anova (default: 999)")

    # Power study parameters
    parser.add_argument(
        "--shift_type", type=str, default=None,
        choices=["mean", "covariance", "both"],
        help="Type of shift for power study (default: auto from test type)",
    )
    parser.add_argument("--delta_scales", type=float, nargs="+", default=None,
                        help="Effect sizes for power study")
    parser.add_argument(
        "--perturbation", type=str, default="eigenvalue",
        choices=["eigenvalue", "additive", "rotation"],
        help="Perturbation mode for covariance shift (default: eigenvalue)",
    )

    args = parser.parse_args()

    # ---- Resolve family -> tree/cov paths from config ----
    if args.family and args.tree == "real":
        from config_utils import get_family_tree_path, get_family_cov_path
        if args.tree_path is None:
            args.tree_path = get_family_tree_path(args.family)
        if args.cov_path is None:
            resolved = get_family_cov_path(args.family)
            if os.path.exists(resolved):
                args.cov_path = resolved

    # ---- Resolve test function and kwargs ----
    test_fn = TEST_REGISTRY[args.test]
    test_kwargs = dict(TEST_DEFAULT_KWARGS.get(args.test, {}))
    # Override defaults with CLI values
    if args.n_bootstrap is not None:
        test_kwargs["n_bootstrap"] = args.n_bootstrap
    if args.n_permutations is not None:
        test_kwargs["n_permutations"] = args.n_permutations

    shift_type = args.shift_type or TEST_DEFAULT_SHIFT.get(args.test, "covariance")

    print(f"KAVERET Simulation Validation")
    print(f"  Test:       {args.test}")
    print(f"  Study:      {args.study}")
    print(f"  Test kwargs: {test_kwargs}")
    print(f"  Seed:       {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()
    all_results = []

    if args.study in ["size", "all"]:
        result = study_type_i_error(
            test_fn, args.test, args.n, args.p, args.reps,
            args.alpha, args.tree, args.seed,
            tree_path=args.tree_path, cov_path=args.cov_path,
            **test_kwargs,
        )
        all_results.append(result)

    if args.study in ["power", "all"]:
        result = study_power(
            test_fn, args.test, args.n, args.p, args.reps,
            args.alpha, shift_type, args.delta_scales,
            args.perturbation, args.tree, args.seed,
            tree_path=args.tree_path, cov_path=args.cov_path,
            **test_kwargs,
        )
        all_results.append(result)

    if args.study in ["robustness", "all"]:
        result = study_robustness(
            test_fn, args.test, args.n, args.p, args.reps,
            args.alpha, tree_type=args.tree, seed=args.seed,
            tree_path=args.tree_path, cov_path=args.cov_path,
            **test_kwargs,
        )
        all_results.append(result)

    if args.study == "grid":
        configs = [
            (50, 5), (50, 10), (50, 20),
            (100, 5), (100, 10), (100, 20),
            (200, 10), (200, 20), (200, 50),
        ]
        grid = study_type_i_grid(
            test_fn, args.test, configs, args.reps, 0,
            args.alpha, args.tree, args.seed,
            tree_path=args.tree_path, cov_path=args.cov_path,
            **test_kwargs,
        )
        all_results.append({"study": "grid", "test": args.test,
                            "grid_results": grid})

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to: {out_path}")

    if args.plot:
        plot_dir = (os.path.dirname(args.output) if args.output else ".")
        plot_results(all_results, os.path.join(plot_dir, "simulation_plots"))

    return all_results


if __name__ == "__main__":
    main()
