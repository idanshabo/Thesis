from utils import *


def plot_error_by_sample_size(data, sample_sizes, proportion, run_count=5, save_folder=None, show_plots=False):
    """
    Plot Type 2: Error by sample size (k) on the x-axis.
    Creates one plot for a given proportion, with multiple matrix sizes as subplots.

    Args:
        data (dict): Simulation data organized by proportion
        sample_sizes (list): List of sample sizes used
        proportion (float): The proportion to plot
        run_count (int, optional): The run count to use. Defaults to 5.
        save_folder (str, optional): Folder to save plots. Defaults to None.
        show_plots (bool, optional): Whether to display plots. Defaults to True.
    """
    # For type 2 with multiple scales structure
    if 'sizes' in data[proportion] and 'results' in data[proportion]:
        proportion_data = data[proportion]
        sizes_data = proportion_data['sizes']
        results_data = proportion_data['results']

        # Calculate number of rows and columns for subplots
        num_sizes = len(sizes_data)
        num_cols = min(3, num_sizes)
        num_rows = (num_sizes + num_cols - 1) // num_cols  # Ceiling division

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 10))
        fig.suptitle(f'U,V ~ I (Proportion n/p = {proportion:.2f})', fontsize=16)
        fig.text(0.5, 0.956, 'Values that dont appear are because the calculation was numerically unstable or MLE doesnt exist',
                 ha='center', va='center', fontsize=10)

        # Flatten axes array for easier indexing
        if num_rows > 1 or num_cols > 1:
            axes = axes.ravel()
        else:
            axes = np.array([axes]).flatten()

        # Hide unused subplots if any
        for i in range(num_sizes, num_rows * num_cols):
            axes[i].set_visible(False)

        # Iterate over different (n, p) pairs
        for idx, ((n, p), size_data) in enumerate(sizes_data.items()):
            ax = axes[idx]

            # Extract data for this matrix size
            row_errors = []
            row_stds = []
            col_errors = []
            col_stds = []
            col_fixed_u_errors = []
            col_fixed_u_stds = []

            for sample_size in sample_sizes:
                sim_result = results_data[(n, p)][sample_size]

                # Get means and stds for each error type
                row_mean = sim_result['row_error']['mean'] if sim_result['row_error']['mean'] is not None else np.nan
                row_std = sim_result['row_error']['std'] if sim_result['row_error']['std'] is not None else np.nan
                col_mean = sim_result['col_error']['mean'] if sim_result['col_error']['mean'] is not None else np.nan
                col_std = sim_result['col_error']['std'] if sim_result['col_error']['std'] is not None else np.nan
                col_fixed_u_mean = sim_result['col_error_fixed_u']['mean'] if sim_result['col_error_fixed_u']['mean'] is not None else np.nan
                col_fixed_u_std = sim_result['col_error_fixed_u']['std'] if sim_result['col_error_fixed_u']['std'] is not None else np.nan

                row_errors.append(row_mean)
                row_stds.append(row_std)
                col_errors.append(col_mean)
                col_stds.append(col_std)
                col_fixed_u_errors.append(col_fixed_u_mean)
                col_fixed_u_stds.append(col_fixed_u_std)

            # Convert to numpy arrays for easier handling
            row_errors = np.array(row_errors)
            row_stds = np.array(row_stds)
            col_errors = np.array(col_errors)
            col_stds = np.array(col_stds)
            col_fixed_u_errors = np.array(col_fixed_u_errors)
            col_fixed_u_stds = np.array(col_fixed_u_stds)

            # Plot row covariance error with error bars
            valid_indices = ~np.isnan(row_errors)
            if np.count_nonzero(valid_indices) >= 3:
                ax.errorbar(np.array(sample_sizes)[valid_indices], row_errors[valid_indices],
                           yerr=row_stds[valid_indices], marker='o', linestyle='--',
                           color='b', label='Row Covariance Error', capsize=5)

            # Plot column covariance error with error bars
            valid_indices = ~np.isnan(col_errors)
            if np.count_nonzero(valid_indices) >= 3:
                ax.errorbar(np.array(sample_sizes)[valid_indices], col_errors[valid_indices],
                           yerr=col_stds[valid_indices], marker='o', linestyle='-',
                           color='g', label='Column Covariance Error', capsize=5)

            # Plot column covariance error with fixed U and error bars
            valid_indices = ~np.isnan(col_fixed_u_errors)
            if np.count_nonzero(valid_indices) >= 3:
                ax.errorbar(np.array(sample_sizes)[valid_indices], col_fixed_u_errors[valid_indices],
                           yerr=col_fixed_u_stds[valid_indices], marker='o', linestyle=':',
                           color='orange', label='Column Covariance Error with Fixed U', capsize=5)

            ax.set_title(f'n={n}, p={p}, n/p={n/p:.2f}')
            ax.set_xlabel('Sample Size (k)')
            ax.set_ylabel('Frobenius Norm of Estimation Error')
            ax.grid(True)

            # Only add legend to the first subplot
            if idx == 0:
                ax.legend()

        plt.tight_layout()

        if save_folder:
            save_path = f"{save_folder}/different_sample_sizes_proportion_{proportion:.2f}.png"
            plt.savefig(save_path)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # For type 2 with single scale structure
    else:
        n = data[proportion]['n']
        p = data[proportion]['p']

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Errors by Sample Size (n={n}, p={p}, n/p={n/p:.2f})', fontsize=16)

        # Extract data for this matrix size
        row_errors = []
        row_stds = []
        col_errors = []
        col_stds = []
        col_fixed_u_errors = []
        col_fixed_u_stds = []

        for sample_size in sample_sizes:
            sim_result = data[proportion]['results'][sample_size][run_count]

            # Get means and stds for each error type
            row_mean = sim_result['row_error']['mean'] if sim_result['row_error']['mean'] is not None else np.nan
            row_std = sim_result['row_error']['std'] if sim_result['row_error']['std'] is not None else np.nan
            col_mean = sim_result['col_error']['mean'] if sim_result['col_error']['mean'] is not None else np.nan
            col_std = sim_result['col_error']['std'] if sim_result['col_error']['std'] is not None else np.nan
            col_fixed_u_mean = sim_result['col_error_fixed_u']['mean'] if sim_result['col_error_fixed_u']['mean'] is not None else np.nan
            col_fixed_u_std = sim_result['col_error_fixed_u']['std'] if sim_result['col_error_fixed_u']['std'] is not None else np.nan

            row_errors.append(row_mean)
            row_stds.append(row_std)
            col_errors.append(col_mean)
            col_stds.append(col_std)
            col_fixed_u_errors.append(col_fixed_u_mean)
            col_fixed_u_stds.append(col_fixed_u_std)

        # Plot row covariance error with error bars
        ax.errorbar(sample_sizes, row_errors, yerr=row_stds, marker='o', linestyle='--',
                    color='b', label='Row Covariance Error', capsize=5)

        # Plot column covariance error with error bars
        ax.errorbar(sample_sizes, col_errors, yerr=col_stds, marker='s', linestyle='-',
                    color='g', label='Column Covariance Error', capsize=5)

        # Plot column covariance error with fixed U and error bars
        ax.errorbar(sample_sizes, col_fixed_u_errors, yerr=col_fixed_u_stds, marker='^', linestyle=':',
                    color='orange', label='Column Covariance Error with Fixed U', capsize=5)

        ax.set_xlabel('Sample Size (k)')
        ax.set_ylabel('Frobenius Norm of Estimation Error')
        ax.grid(True)
        ax.legend(loc='best')

        plt.tight_layout()

        if save_folder:
            save_path = f"{save_folder}/error_by_sample_size_proportion_{proportion:.2f}.png"
            plt.savefig(save_path)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)


        ax.legend(loc='best')

        plt.tight_layout()

        if save_folder:
            save_path = f"{save_folder}/convergence_by_n_sample_size_{sample_size}.png"
            plt.savefig(save_path)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

def plot_error_by_proportion(data, sample_sizes, run_count, save_folder=None, show_plots=False):
    """
    Plot type 3: After reached convergence - comparing error size for different proportions.
    One plot for every sample size (k), the x-axis is the proportion.

    Args:
        data (dict): Simulation data organized by proportion
        sample_sizes (list): List of sample sizes used
        run_count (int): The run count to use for plots
        save_folder (str, optional): Folder to save plots. Defaults to None.
        show_plots (bool, optional): Whether to display plots. Defaults to True.
    """
    # Create a separate plot for each sample size
    for sample_size in sample_sizes:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data for this sample size across all proportions
        proportions = []
        row_errors = []
        row_errors_std = []
        col_errors = []
        col_errors_std = []
        col_fixed_u_errors = []
        col_fixed_u_errors_std = []

        for proportion, prop_data in sorted(data.items()):
            proportions.append(proportion)

            sim_result = prop_data['results'][sample_size][run_count]

            # Replace None with np.nan to avoid errors in errorbar
            row_mean = sim_result['row_error']['mean'] if sim_result['row_error']['mean'] is not None else np.nan
            row_std = sim_result['row_error']['std'] if sim_result['row_error']['std'] is not None else np.nan
            col_mean = sim_result['col_error']['mean'] if sim_result['col_error']['mean'] is not None else np.nan
            col_std = sim_result['col_error']['std'] if sim_result['col_error']['std'] is not None else np.nan
            col_fixed_u_mean = sim_result['col_error_fixed_u']['mean'] if sim_result['col_error_fixed_u']['mean'] is not None else np.nan
            col_fixed_u_std = sim_result['col_error_fixed_u']['std'] if sim_result['col_error_fixed_u']['std'] is not None else np.nan

            row_errors.append(row_mean)
            row_errors_std.append(row_std)
            col_errors.append(col_mean)
            col_errors_std.append(col_std)
            col_fixed_u_errors.append(col_fixed_u_mean)
            col_fixed_u_errors_std.append(col_fixed_u_std)

        # Plot each error type
        ax.errorbar(proportions, row_errors, yerr=row_errors_std, marker='o', linestyle='--', color='b',
                label='Row Covariance Error', capsize=5)
        ax.errorbar(proportions, col_errors, yerr=col_errors_std, marker='s', linestyle='-', color='g',
                label='Column Covariance Error', capsize=5)
        ax.errorbar(proportions, col_fixed_u_errors, yerr=col_fixed_u_errors_std, marker='^', linestyle=':', color='orange',
                label='Column Covariance Error with Fixed U', capsize=5)

        ax.set_title(f'Errors by Proportion for Sample Size k={sample_size}', fontsize=16)
        ax.set_xlabel('n/p (Proportion)')
        ax.set_ylabel('Frobenius Norm of Estimation Error')
        ax.grid(True)
        ax.legend(loc='best')

        # Set x-ticks to match the proportions
        ax.set_xticks(proportions)
        ax.set_xticklabels([f"{p:.2f}" for p in proportions])

        plt.tight_layout()

        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = f"{save_folder}/error_by_proportion_sample_size_{sample_size}.png"
            plt.savefig(save_path)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)


def plot_error_metric_multi_scale(results, metric_key, title_prefix, file_prefix, color, drive_folder, sample_sizes):
    for proportion in results:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
        fig.suptitle(f'{title_prefix} - Proportion n/p = {round(proportion, 2)}', fontsize=16)
        fig.text(0.5, 0.956, 'Matrix size (n) on x-axis, each curve = different scale', ha='center', va='center', fontsize=10)
        axes = axes.ravel()

        for k_idx, k in enumerate(sample_sizes):
            ax = axes[k_idx]
            ax.set_title(f'Sample Size (k) = {k}')
            ax.set_xlabel('Matrix Size (n)')
            ax.set_ylabel('Log Frobenius Norm Error')
            ax.grid(True)

            # Store all scale results for this subplot (k) in lists
            x_vals = []
            y_vals = []
            y_errs = []
            p_vals = []

            for scale in sorted(results[proportion]):
                scale_data = results[proportion][scale]
                n = scale_data['n']
                p = scale_data['p']
                result = scale_data['results'][k]

                error_stats = result.get(metric_key, {})
                mean = error_stats.get('mean', None)
                std = error_stats.get('std', None)

                if mean is not None:
                    x_vals.append(n)
                    y_vals.append(mean)
                    y_errs.append(std)
                    p_vals.append(p)

            if x_vals:
                # Sort by x_vals for a proper line plot
                sorted_data = sorted(zip(x_vals, y_vals, y_errs, p_vals))
                x_vals, y_vals, y_errs, p_vals = zip(*sorted_data)

                ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='o-', capsize=3, color=color)

                for xi, yi, pi in zip(x_vals, y_vals, p_vals):
                    ax.annotate(f'p={pi}', (xi, yi), xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        save_path = f'{drive_folder}{file_prefix}_proportion_{round(proportion, 2)}.png'
        plt.savefig(save_path)
        plt.close(fig)
