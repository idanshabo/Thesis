import numpy as np
from scipy.stats import matrix_normal, wishart
import matplotlib.pyplot as plt
#from google.colab import drive
#drive.mount('/content/drive')
from estimate_matrix_normal import matrix_normal_mle, matrix_normal_mle_fixed_u


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


def compute_frobenius_norm_errors(true_matrix, estimated_matrix):
    if estimated_matrix is None:
        return None
    return np.log(np.linalg.norm(true_matrix - estimated_matrix) / np.linalg.norm(true_matrix))


def calculate_stats(errors_list):
    if not errors_list:
        return None, None
    return np.mean(errors_list), np.std(errors_list)


def generate_covariance_matrices(Sigma_r, Sigma_c, n=None, p=None, cov_distribution=None,):
    if cov_distribution == 'wishart':
        Sigma_r = wishart.rvs(df=n, scale=Sigma_r, size=1)
        Sigma_c = wishart.rvs(df=p, scale=Sigma_c, size=1)
    return Sigma_r, Sigma_c


def adjust_sample_to_match_sample_size_equal_to_one(matrix_normal_sample):
    if matrix_normal_sample.shape[0] == 1:
        matrix_normal_sample = matrix_normal_sample[0]
    return matrix_normal_sample


def sample_matrix_normal(M, Sigma_r, Sigma_c, sample_size):
    matrix_normal_sample = matrix_normal.rvs(M, Sigma_r, Sigma_c, size=sample_size)
    matrix_normal_sample = adjust_sample_to_match_sample_size_equal_to_one(matrix_normal_sample)
    return matrix_normal_sample


def run_simulation(n, p, sample_size, num_runs, cov_distribution):
    """
    Run simulation for a specific matrix size and sample size.
    
    Args:
        n (int): Number of rows
        p (int): Number of columns
        sample_size (int): Number of sample matrices to generate
        num_runs (int): Number of simulation runs
        
    Returns:
        tuple: Containing lists of errors for row covariance, column covariance,
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
        Sigma_r_true, Sigma_c_true = generate_covariance_matrices(Sigma_r_true, Sigma_c_true, cov_distribution)
        # Generate sample matrices:
        matrix_normal_sample = sample_matrix_normal(M_true, Sigma_r_true, Sigma_c_true, sample_size)
            
        # Estimate parameters
        M_est, Sigma_r_est, Sigma_c_est = matrix_normal_mle(matrix_normal_sample)
        M_est_fixed_u, Sigma_c_est_fixed_u = matrix_normal_mle_fixed_u(matrix_normal_sample, Sigma_r_true)
        
        # Compute Frobenius norm errors for row and column covariance:
        true_and_estimated_matrics_with_error_list = [(Sigma_r_true, Sigma_r_est, row_error_runs),
                                                      (Sigma_c_true, Sigma_c_est, col_error_runs),
                                                      (Sigma_c_true, Sigma_c_est_fixed_u, col_error_fixed_u_runs)]
        for true_matrix, estimated_matrix, errors_list in true_and_estimated_matrics_with_error_list:
            errors_list.append(compute_frobenius_norm_errors(true_matrix, estimated_matrix))

    
    # Filter out None values
    row_error_runs_filtered = list(filter(None, row_error_runs))
    col_error_runs_filtered = list(filter(None, col_error_runs))
    col_error_fixed_u_runs_filtered = list(filter(None, col_error_fixed_u_runs))
    
    # Calculate mean and standard deviation for each error type
    row_error_stats = calculate_stats(row_error_runs_filtered)
    col_error_stats = calculate_stats(col_error_runs_filtered)
    col_error_fixed_u_stats = calculate_stats(col_error_fixed_u_runs_filtered)
    
    return row_error_stats, col_error_stats, col_error_fixed_u_stats


def collect_errors(sizes, sample_sizes, num_runs, cov_distribution=None):
    """
    Collect errors for different matrix sizes and sample sizes.
    
    Args:
        sizes (list): List of tuples (n, p) representing matrix dimensions
        sample_sizes (list): List of sample sizes to test
        num_runs (int): Number of simulation runs for each configuration
        
    Returns:
        tuple: Dictionaries containing errors for row covariance, column covariance,
               and column covariance with fixed U
    """
    row_errors = {size: [] for size in sizes}
    col_errors = {size: [] for size in sizes}
    col_errors_fixed_u = {size: [] for size in sizes}
    
    for n, p in sizes:
        for sample_size in sample_sizes:
            row_error_stats, col_error_stats, col_error_fixed_u_stats = run_simulation(n, p, sample_size, num_runs, cov_distribution)
            
            row_errors[(n, p)].append(row_error_stats)
            col_errors[(n, p)].append(col_error_stats)
            col_errors_fixed_u[(n, p)].append(col_error_fixed_u_stats)
    
    return row_errors, col_errors, col_errors_fixed_u


def plot_results(sizes, sample_sizes, row_errors, col_errors, col_errors_fixed_u, save_path=None):
    """
    Plot the results of the simulation.
    
    Args:
        sizes (list): List of tuples (n, p) representing matrix dimensions
        sample_sizes (list): List of sample sizes used
        row_errors (dict): Dictionary containing row covariance errors
        col_errors (dict): Dictionary containing column covariance errors
        col_errors_fixed_u (dict): Dictionary containing column covariance errors with fixed U
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(sizes), figsize=(18, 6))
    
    # Ensure axes is iterable (even for a single plot)
    if len(sizes) == 1:
        axes = [axes]  # If there's only one axis, wrap it in a list
    
    for idx, (n, p) in enumerate(sizes):
        ax = axes[idx]
        
        # Plot row covariance error with error bars
        row_mean, row_std = zip(*row_errors[(n, p)])
        row_mean, row_std = np.array(row_mean), np.array(row_std)
        row_mean = np.array([np.nan if x is None else x for x in row_mean])
        row_std = np.array([np.nan if x is None else x for x in row_std])
        valid_indices = ~np.isnan(row_mean)
        
        if np.count_nonzero(~np.isnan(row_mean)) < 3:
            valid_indices = np.array([False] * len(sample_sizes))
        
        ax.errorbar(np.array(sample_sizes)[valid_indices], row_mean[valid_indices], yerr=row_std[valid_indices],
                    label='Row Covariance Error', marker='o', linestyle='--', color='b', capsize=5)
        
        # Plot column covariance error with error bars
        col_mean, col_std = zip(*col_errors[(n, p)])
        col_mean, col_std = np.array(col_mean), np.array(col_std)
        col_mean = np.array([np.nan if x is None else x for x in col_mean])
        col_std = np.array([np.nan if x is None else x for x in col_std])
        valid_indices = ~np.isnan(col_mean)
        
        if np.count_nonzero(~np.isnan(col_mean)) < 3:
            valid_indices = np.array([False] * len(sample_sizes))
        
        ax.errorbar(np.array(sample_sizes)[valid_indices], col_mean[valid_indices], yerr=col_std[valid_indices],
                    label='Column Covariance Error', marker='o', linestyle='-', color='r', capsize=5)
        
        # Plot column covariance error with fixed U and error bars
        col_fixed_u_mean, col_fixed_u_std = zip(*col_errors_fixed_u[(n, p)])
        col_fixed_u_mean, col_fixed_u_std = np.array(col_fixed_u_mean), np.array(col_fixed_u_std)
        valid_indices = ~np.isnan(col_fixed_u_mean)
        
        if np.count_nonzero(~np.isnan(col_fixed_u_mean)) < 3:
            valid_indices = np.array([False] * len(sample_sizes))
        
        ax.errorbar(np.array(sample_sizes)[valid_indices], col_fixed_u_mean[valid_indices], yerr=col_fixed_u_std[valid_indices],
                    label='Column Covariance Error with Fixed U', marker='o', linestyle=':', color='orange', capsize=5)
        
        ax.set_title(f'n={n}, p={p}, n/p={n/p:.2f}')
        ax.set_xlabel('Sample Size (k)')
        ax.set_ylabel('Frobenius Norm of Estimation Error')
        ax.grid(True)
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = ['Row Covariance Error', 'Column Covariance Error', 'Column Covariance Error with Fixed U']
    fig.legend(handles[:3], unique_labels, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    fig.suptitle(f'Errors for Matrix Normal Estimation', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def test_matrix_normal_mle_functions_after_convergence(convergence_scale=200, 
                                                       proportions=[1/4, 1/3, 1/2, 1, 2, 3, 4],
                                                       sample_sizes_to_check=list(range(1, 6)),
                                                       num_runs=5,
                                                       cov_distribution='wishart',
                                                       save_folder=None):
    """
    Test matrix normal MLE functions after convergence.
    
    Args:
        convergence_scale (int, optional): Fixed scale value. Defaults to 200.
        proportions (list, optional): List of proportions n/p to test. Defaults to [1/4, 1/3, 1/2, 1, 2, 3, 4].
        sample_sizes_to_check (list, optional): List of sample sizes to test. Defaults to list(range(1, 6)).
        num_runs (int, optional): Number of simulation runs for each configuration. Defaults to 5.
        save_folder (str, optional): Folder to save plots. Defaults to None.
    """
    sizes_by_proportion = create_matrix_sizes_with_one_scale(proportions, convergence_scale)
    
    # Run tests for each proportion
    for proportion, matrix_size in sizes_by_proportion.items():
        sizes = [matrix_size]  # List with a single tuple (n, p)
        
        # Collect errors for different sample sizes
        row_errors, col_errors, col_errors_fixed_u = collect_errors(sizes, sample_sizes_to_check, num_runs, cov_distribution)
        
        # Plot the results
        save_path = None
        if save_folder:
            save_path = f"{save_folder}/different_sample_sizes_with_proportion_n_p={round(matrix_size[0]/matrix_size[1], 2)}.png"
        
        plot_results(sizes, sample_sizes_to_check, row_errors, col_errors, col_errors_fixed_u, save_path)
