from run_plot_functions import *


# Example usage
if __name__ == "__main__":
    test_matrix_normal_mle_functions_after_convergence(
        convergence_scale=200,
        proportions=[1/4, 1/3, 1/2, 1, 2, 3, 4],
        sample_sizes_to_check=list(range(1, 6)),
        num_runs=5,
        save_folder="/content/drive/My Drive/Thesis/matrix normal estimation/"
