from run_plot_functions import *


# Example usage
if __name__ == "__main__":
    save_path = '/content/drive/My Drive/Thesis/matrix normal estimation/'
    proportions = [1/4, 1/3, 1/2, 1]
    scales = [2, 10, 50]
    convergence_scale = 20

    run_plot_convergence_by_matrix_size(save_path, proportions, scales)
    run_error_metric_multi_scale(save_path, proportions, scales)
    run_plot_error_by_proportion(save_path, proportions, convergence_scale)

# There are more parameters adjustable, defined in the run_plot_functions file
