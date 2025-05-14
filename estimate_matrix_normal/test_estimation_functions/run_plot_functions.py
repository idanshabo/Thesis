from plot_functions import *


def run_plot_error_by_proportion(save_path,
                                 proportions=[1/4, 1/3, 1/2, 1, 2, 3, 4],
                                 convergence_scale=200,
                                 sample_sizes=list(range(1, 6)),
                                 num_runs=10,
                                 cov_distribution=None):
    # Create matrix sizes for each proportion
    sizes_by_proportion = create_matrix_sizes_with_one_scale(proportions, convergence_scale)

    # Collect simulation data
    data = collect_simulation_data_single_scale(
        dimensions_to_test=sizes_by_proportion,
        sample_sizes=sample_sizes,
        run_counts=[num_runs],  # Only need one run count
        cov_distribution=None
    )

    # Generate plots
    plot_error_by_proportion(
        data=data,
        sample_sizes=sample_sizes,
        run_count=num_runs,
        save_folder=save_path,
        show_plots=False
    )


def run_plot_convergence_by_matrix_size(save_path,
                                        proportions=[1/4, 1/3, 1/2, 1, 2, 3, 4],
                                        scales=[2, 10, 50, 100, 200],
                                        sample_sizes=list(range(1, 6)),
                                        num_runs=20,
                                        cov_distribution=None):
    # Create matrix sizes with multiple scales for each proportion
    dimensions_by_proportion = create_matrix_sizes_with_multiple_scales(proportions, scales)

    # Collect simulation data
    data = collect_simulation_data_multiple_scales(
        dimensions_by_proportion=dimensions_by_proportion,
        sample_sizes=sample_sizes,
        num_runs=num_runs,
        cov_distribution=None
    )

    # For each proportion, generate a plot showing errors for different sample sizes
    for proportion in proportions:
        plot_error_by_sample_size(
            data=data,
            sample_sizes=sample_sizes,
            proportion=proportion,
            save_folder=save_path,
            show_plots=True
        )


def run_error_metric_multi_scale(save_path,
                                 proportions=[1/4, 1/3, 1/2, 1, 2, 3, 4],
                                 scales = [2, 10, 50, 100, 200, 250],
                                 sample_sizes=list(range(1, 7)), 
                                 num_runs=5, 
                                 cov_distribution=None):
    # Collect results
    simulation_results_multi_scale = collect_simulation_data_multi_scale(proportions, scales, sample_sizes, num_runs)

    if cov_distribution == "wishart":
        distribution = '(U,V ~ Wishart)'
    else:
        distribution = '(U,V ~ I)'

    # Plots
    plot_error_metric_multi_scale(simulation_results_multi_scale, 'row_error',
                                  f'Row Covariance Error {distribution}', 'row_error', 'blue', save_path, sample_sizes)

    plot_error_metric_multi_scale(simulation_results_multi_scale, 'col_error',
                                  f'Column Covariance Error {distribution}', 'col_error', 'green', save_path, sample_sizes)

    plot_error_metric_multi_scale(simulation_results_multi_scale, 'col_error_fixed_u',
                                  f'Column Covariance Error with Fixed U {distribution}', 'col_fixed_u_error', 'orange', save_path, sample_sizes)

    print("✅ Plots saved successfully.")
