import numpy as np
from scipy.stats import matrix_normal, wishart
import matplotlib.pyplot as plt
import sys
import os
from estimate_matrix_normal.estimate_matrix_normal import *


def create_matrix_sizes_with_one_scale(proportions, scale):
    """
    Create matrix dimensions based on proportions and a fixed scale.

    Args:
        proportions (list): List of proportions n/p to test
        scale (int): Fixed scale value

    Returns:
        dict: Dictionary mapping each proportion to a tuple of matrix dimensions (n, p)
    """
    sizes_by_proportion = {}
    for proportion in proportions:
        if proportion < 1:
            sizes_by_proportion[proportion] = (int(scale), int(scale / proportion))
        else:
            sizes_by_proportion[proportion] = (int(scale * proportion), int(scale))
    return sizes_by_proportion

def create_matrix_sizes_with_multiple_scales(proportions, scales):
    """
    Create matrix dimensions based on proportions and multiple scales.

    Args:
        proportions (list): List of proportions n/p to test
        scales (list): List of scale values to test

    Returns:
        dict: Dictionary mapping each proportion to a list of tuples of matrix dimensions (n, p)
    """
    sizes_by_proportion = {}
    for proportion in proportions:
        if proportion < 1:
            sizes_by_proportion[proportion] = [(int(scale), int(scale / proportion)) for scale in scales]
        else:
            sizes_by_proportion[proportion] = [(int(scale * proportion), int(scale)) for scale in scales]
    return sizes_by_proportion

def compute_frobenius_norm_errors(true_matrix, estimated_matrix):
    """
    Compute the log of the normalized Frobenius norm error between two matrices.

    Args:
        true_matrix: The true matrix
        estimated_matrix: The estimated matrix

    Returns:
        float or None: Log of the normalized Frobenius norm error, or None if estimated_matrix is None
    """
    if estimated_matrix is None:
        return None
    return np.log(np.linalg.norm(true_matrix - estimated_matrix) / np.linalg.norm(true_matrix))

def calculate_stats(errors_list):
    """
    Calculate mean and standard deviation of a list of errors.

    Args:
        errors_list: List of error values

    Returns:
        tuple: (mean, std) of errors, or (None, None) if list is empty
    """
    errors_filtered = [e for e in errors_list if e is not None]
    if not errors_filtered:
        return None, None
    return np.mean(errors_filtered), np.std(errors_filtered)

def generate_covariance_matrices(Sigma_r, Sigma_c, n=None, p=None, cov_distribution=None):
    """
    Generate covariance matrices based on the specified distribution.

    Args:
        Sigma_r: Row covariance matrix
        Sigma_c: Column covariance matrix
        n: Number of rows
        p: Number of columns
        cov_distribution: Distribution type for covariance matrices

    Returns:
        tuple: (Sigma_r, Sigma_c) potentially modified based on distribution
    """
    if cov_distribution == 'wishart':
        Sigma_r = wishart.rvs(df=n, scale=Sigma_r, size=1)
        Sigma_c = wishart.rvs(df=p, scale=Sigma_c, size=1)
    return Sigma_r, Sigma_c

def sample_matrix_normal(M, Sigma_r, Sigma_c, sample_size):
    """
    Generate samples from a matrix normal distribution.

    Args:
        M: Mean matrix
        Sigma_r: Row covariance matrix
        Sigma_c: Column covariance matrix
        sample_size: Number of samples to generate

    Returns:
        Sample matrices
    """
    matrix_normal_sample = matrix_normal.rvs(M, Sigma_r, Sigma_c, size=sample_size)
    if matrix_normal_sample.shape[0] == 1:
        matrix_normal_sample = matrix_normal_sample[0]
    return matrix_normal_sample

# --------------------------------------------------------------------------------
# Simulation Core Functions
# --------------------------------------------------------------------------------

def run_simulation(n, p, sample_size, num_runs, cov_distribution=None):
    """
    Run simulation for a specific matrix size and sample size.

    Args:
        n (int): Number of rows
        p (int): Number of columns
        sample_size (int): Number of sample matrices to generate
        num_runs (int): Number of simulation runs
        cov_distribution: Distribution type for covariance matrices

    Returns:
        dict: Dictionary containing error statistics for row covariance, column covariance,
              and column covariance with fixed U
    """
    # True matrices for simulation
    M_true = np.zeros((n, p))
    Sigma_r_true = np.eye(n)
    Sigma_c_true = np.eye(p)

    # Store errors from multiple runs
    row_error_runs = []
    col_error_runs = []
    col_error_fixed_u_runs = []

    for _ in range(num_runs):
        # Generate covariance matrices for sampling:
        Sigma_r_use, Sigma_c_use = generate_covariance_matrices(Sigma_r_true, Sigma_c_true, n, p, cov_distribution)

        # Generate sample matrices:
        matrix_normal_sample = sample_matrix_normal(M_true, Sigma_r_use, Sigma_c_use, sample_size)

        # Estimate parameters
        M_est, Sigma_r_est, Sigma_c_est = matrix_normal_mle(matrix_normal_sample)
        M_est_fixed_u, Sigma_c_est_fixed_u = matrix_normal_mle_fixed_u(matrix_normal_sample, Sigma_r_true)

        # Compute Frobenius norm errors for row and column covariance:
        true_and_estimated_matrices_with_error_list = [
            (Sigma_r_true, Sigma_r_est, row_error_runs),
            (Sigma_c_true, Sigma_c_est, col_error_runs),
            (Sigma_c_true, Sigma_c_est_fixed_u, col_error_fixed_u_runs)
        ]

        for true_matrix, estimated_matrix, errors_list in true_and_estimated_matrices_with_error_list:
            errors_list.append(compute_frobenius_norm_errors(true_matrix, estimated_matrix))

    # Calculate mean and standard deviation for each error type
    row_error_mean, row_error_std = calculate_stats(row_error_runs)
    col_error_mean, col_error_std = calculate_stats(col_error_runs)
    col_error_fixed_u_mean, col_error_fixed_u_std = calculate_stats(col_error_fixed_u_runs)

    # Return results in dictionary format
    return {
        'row_error': {
            'mean': row_error_mean,
            'std': row_error_std
        },
        'col_error': {
            'mean': col_error_mean,
            'std': col_error_std
        },
        'col_error_fixed_u': {
            'mean': col_error_fixed_u_mean,
            'std': col_error_fixed_u_std
        }
    }

def collect_simulation_data_single_scale(dimensions_to_test, sample_sizes, run_counts, cov_distribution=None):
    """
    Function to collect simulation data for various configurations with a single scale.

    Args:
        dimensions_to_test (dict): Dictionary mapping proportions to (n,p) tuples
        sample_sizes (list): List of sample sizes to test
        run_counts (list): List of run counts to test
        cov_distribution: Distribution type for covariance matrices

    Returns:
        dict: Nested dictionary with simulation results
    """
    results = {}

    # Handle as a dict mapping proportions to (n,p) tuples
    for proportion, (n, p) in dimensions_to_test.items():
        proportion_key = float(proportion)  # Ensure it's a float for consistent keys
        results[proportion_key] = {'n': n, 'p': p, 'results': {}}

        for sample_size in sample_sizes:
            results[proportion_key]['results'][sample_size] = {}
            for run_count in run_counts:
                results[proportion_key]['results'][sample_size][run_count] = run_simulation(n, p, sample_size, run_count, cov_distribution)

    return results

def collect_simulation_data_multiple_scales(dimensions_by_proportion, sample_sizes, num_runs, cov_distribution=None):
    """
    Function to collect simulation data for various configurations with multiple scales.

    Args:
        dimensions_by_proportion (dict): Dictionary mapping proportions to list of (n,p) tuples
        sample_sizes (list): List of sample sizes to test
        num_runs (int): Number of simulation runs
        cov_distribution: Distribution type for covariance matrices

    Returns:
        dict: Nested dictionary with simulation results
    """
    results = {}

    for proportion, sizes_list in dimensions_by_proportion.items():
        proportion_key = float(proportion)  # Ensure it's a float for consistent keys
        results[proportion_key] = {'sizes': {}, 'results': {}}

        for n, p in sizes_list:
            results[proportion_key]['sizes'][(n, p)] = {'n': n, 'p': p}
            results[proportion_key]['results'][(n, p)] = {}

            for sample_size in sample_sizes:
                results[proportion_key]['results'][(n, p)][sample_size] = run_simulation(n, p, sample_size, num_runs, cov_distribution)

    return results

def collect_simulation_data_multi_scale(proportions, scales, sample_sizes, num_runs, cov_distribution=None):
    """
    Collects simulation data for multiple scales and proportions.
    Returns: dict[proportion][scale][sample_size] = results dict
    """
    all_results = {}
    for proportion in proportions:
        all_results[proportion] = {}
        for scale in scales:
            n, p = create_matrix_sizes_with_one_scale([proportion], scale)[proportion]
            scale_results = {}
            for k in sample_sizes:
                scale_results[k] = run_simulation(n, p, k, num_runs, cov_distribution)
            all_results[proportion][scale] = {
                'n': n,
                'p': p,
                'results': scale_results
            }
    return all_results
