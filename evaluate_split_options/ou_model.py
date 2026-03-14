import torch
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from evaluate_split_options.lrt_statistics import compute_gls_operators, add_jitter


def bm_to_ou_covariance(U_BM, alpha):
    """
    Converts a BM phylogenetic covariance matrix to an OU covariance matrix
    without re-traversing the tree.

    Uses the algebraic relationship:
        d_ij = U_BM[i,i] + U_BM[j,j] - 2*U_BM[i,j]     (patristic distance)
        U_OU[i,j] = exp(-alpha * d_ij) * (1 - exp(-2*alpha * U_BM[i,j]))   for i != j
        U_OU[i,i] = 1 - exp(-2*alpha * U_BM[i,i])

    As alpha -> 0: U_OU -> 2*alpha * U_BM (BM nested in OU, scalar absorbed into V).

    Parameters
    ----------
    U_BM : torch.Tensor
        (n x n) BM phylogenetic covariance matrix.
        U_BM[i,j] = dist(root, MRCA(i,j)), U_BM[i,i] = dist(root, tip_i).
    alpha : float
        OU selection strength parameter. Must be > 0.

    Returns
    -------
    U_OU : torch.Tensor
        (n x n) OU covariance matrix in float64.
    """
    U = U_BM.double()
    n = U.shape[0]

    # Diagonal: root-to-tip distances
    diag = U.diag().unsqueeze(1)  # (n, 1)

    # Patristic distance matrix: d_ij = t_i + t_j - 2 * t_mrca
    D = diag + diag.T - 2 * U
    D = torch.clamp(D, min=0.0)  # Remove floating-point noise

    # OU covariance: exp(-alpha * d_ij) * (1 - exp(-2*alpha * t_mrca))
    U_OU = torch.exp(-alpha * D) * (1.0 - torch.exp(-2.0 * alpha * U))

    # Overwrite diagonal: Var(X_i) = 1 - exp(-2*alpha*t_i)
    U_OU.diagonal().copy_(1.0 - torch.exp(-2.0 * alpha * U.diag()))

    # Force symmetry
    U_OU = (U_OU + U_OU.T) / 2.0

    return U_OU


def profile_log_likelihood(X, U):
    """
    Computes the profile (concentrated) log-likelihood of the matrix normal model
    X ~ MN(M, U, V) with M and V profiled out (maximized analytically).

        LL(U) = -(np/2)*log(2*pi) - (p/2)*log|U| - (n/2)*log|V_hat(U)| - np/2

    where V_hat(U) = (1/n) * X^T * P(U) * X and P is the GLS projection operator.

    Parameters
    ----------
    X : torch.Tensor
        (n x p) data matrix (PCA-reduced embeddings).
    U : torch.Tensor
        (n x n) phylogenetic covariance matrix (BM or OU).

    Returns
    -------
    ll : float
        The profile log-likelihood value.
    """
    X = X.double()
    U = U.double()
    n, p = X.shape

    # Reuse existing GLS machinery
    U_inv, P, t1, t2 = compute_gls_operators(U)

    # MLE of V given U
    S = X.T @ P @ X
    V_hat = S / n
    V_hat = (V_hat + V_hat.T) / 2.0

    # Log-determinants (with jitter for numerical stability)
    log_det_U = torch.linalg.slogdet(add_jitter(U))[1]
    log_det_V = torch.linalg.slogdet(add_jitter(V_hat))[1]

    # Profile log-likelihood
    ll = -(n * p / 2.0) * math.log(2.0 * math.pi) \
         - (p / 2.0) * log_det_U \
         - (n / 2.0) * log_det_V \
         - (n * p / 2.0)

    return ll.item()


def estimate_alpha_mle(X, U_BM, alpha_bounds=(1e-4, 50.0)):
    """
    Estimates the optimal OU selection strength alpha by maximizing
    the profile log-likelihood over alpha.

    Optimization is on log(alpha) for numerical stability.

    Parameters
    ----------
    X : torch.Tensor
        (n x p) PCA-reduced embedding data matrix.
    U_BM : torch.Tensor
        (n x n) BM phylogenetic covariance matrix.
    alpha_bounds : tuple of (float, float)
        Lower and upper bounds for alpha. Default (1e-4, 50.0).

    Returns
    -------
    result : dict
        {"alpha_hat", "ll_ou", "converged", "n_feval"}
    """
    X = X.double()
    U_BM = U_BM.double()

    log_lb = math.log(alpha_bounds[0])
    log_ub = math.log(alpha_bounds[1])

    def neg_ll(log_alpha):
        alpha = math.exp(log_alpha)
        try:
            U_OU = bm_to_ou_covariance(U_BM, alpha)
            return -profile_log_likelihood(X, U_OU)
        except Exception:
            return float('inf')

    opt = minimize_scalar(neg_ll, bounds=(log_lb, log_ub), method='bounded',
                          options={'xatol': 1e-6, 'maxiter': 200})

    alpha_hat = math.exp(opt.x)
    ll_ou = -opt.fun

    # Check if alpha landed at a boundary
    converged = True
    if abs(opt.x - log_lb) < 0.01:
        print(f"  [WARNING] alpha_hat={alpha_hat:.6f} is at the lower bound. "
              f"Consider lowering --alpha_min.")
        converged = False
    if abs(opt.x - log_ub) < 0.01:
        print(f"  [WARNING] alpha_hat={alpha_hat:.6f} is at the upper bound. "
              f"Consider raising --alpha_max.")
        converged = False

    return {
        "alpha_hat": alpha_hat,
        "ll_ou": ll_ou,
        "converged": converged,
        "n_feval": opt.nfev
    }


def compare_bm_ou(X, U_BM, alpha_bounds=(1e-4, 50.0)):
    """
    Full BM vs OU model comparison for phylogenetic trait evolution.

    Computes profile log-likelihoods, LRT (chi2, df=1), AIC, and BIC.

    Parameters
    ----------
    X : torch.Tensor
        (n x p) PCA-reduced embedding data matrix.
    U_BM : torch.Tensor
        (n x n) BM phylogenetic covariance matrix.
    alpha_bounds : tuple of (float, float)
        Bounds for alpha search. Default (1e-4, 50.0).

    Returns
    -------
    result : dict
        Full comparison results including likelihoods, LRT, AIC, BIC.
    """
    X = X.double()
    U_BM = U_BM.double()
    n, p = X.shape

    # BM profile log-likelihood
    ll_bm = profile_log_likelihood(X, U_BM)

    # OU: estimate alpha and compute profile log-likelihood
    ou_result = estimate_alpha_mle(X, U_BM, alpha_bounds)
    alpha_hat = ou_result["alpha_hat"]
    ll_ou = ou_result["ll_ou"]

    # LRT: BM is nested in OU (alpha -> 0), df = 1
    lrt_stat = max(0.0, 2.0 * (ll_ou - ll_bm))
    lrt_pvalue = float(chi2.sf(lrt_stat, df=1))

    # Parameter counts (U is fixed from tree in both models)
    # Common: mean M (p params) + V (p*(p+1)/2 params)
    k_common = p + p * (p + 1) // 2
    k_bm = k_common       # BM adds 0 free params for U
    k_ou = k_common + 1   # OU adds alpha

    # AIC and BIC
    aic_bm = -2.0 * ll_bm + 2.0 * k_bm
    aic_ou = -2.0 * ll_ou + 2.0 * k_ou
    bic_bm = -2.0 * ll_bm + k_bm * math.log(n)
    bic_ou = -2.0 * ll_ou + k_ou * math.log(n)

    return {
        "n": n,
        "p": p,
        "alpha_hat": alpha_hat,
        "alpha_converged": ou_result["converged"],
        "alpha_n_feval": ou_result["n_feval"],
        "ll_bm": ll_bm,
        "ll_ou": ll_ou,
        "lrt_statistic": lrt_stat,
        "lrt_df": 1,
        "lrt_pvalue": lrt_pvalue,
        "aic_bm": aic_bm,
        "aic_ou": aic_ou,
        "bic_bm": bic_bm,
        "bic_ou": bic_ou,
        "preferred_model_aic": "OU" if aic_ou < aic_bm else "BM",
        "preferred_model_bic": "OU" if bic_ou < bic_bm else "BM",
    }
