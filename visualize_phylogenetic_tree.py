# Import the necessary libraries. We'll use biopython for tree parsing and matplotlib for plotting.
from Bio import Phylo
import matplotlib.pyplot as plt

def visualize_phylogenetic_tree(file_path: str, output_image_path: str = "tree_visualization.png"):
    """
    Reads a phylogenetic tree from a Newick file, visualizes it using biopython and matplotlib,
    and saves the image. The visualization is customized to improve label readability.

    Args:
        file_path (str): The path to the input Newick file.
        output_image_path (str): The desired path for the output image file (e.g., 'my_tree.png').
    """
    try:
        # Load the tree from the Newick file using biopython's read function.
        # The 'newick' format is specified.
        tree = Phylo.read(file_path, "newick")

        # Create a matplotlib figure and axes with an increased size for better label spacing.
        # The figsize is now 15x15 inches, which provides much more room.
        fig, ax = plt.subplots(figsize=(15, 15))

        # We'll use a custom layout for the tree. By default, labels are horizontal.
        # Here we add a function to rotate the labels for better readability if they are overlapping.
        def get_label(clade):
            # We check if the clade has a name before trying to return it.
            if clade.name:
                # We return a Text object with a rotation of 45 degrees.
                return plt.text(clade.branch_length, 0, clade.name,
                                        rotation=45, va="bottom", ha="left", fontsize=8)
            return '' # Return an empty string if there's no name

        # Draw the tree on the axes. We've added more customization options here:
        # - The `label_func` is a callable that returns the label for each clade.
        # - The `label_fontsize` is set to a smaller value to help prevent overlap.
        # - We use `branch_labels` to show branch lengths, which can be useful.
        Phylo.draw(tree, axes=ax, do_show=False,
                   label_func=lambda c: get_label(c),
                   branch_labels=lambda c: f"{c.branch_length:.2f}" if c.branch_length is not None else "",
                   label_fontsize=8,
                   branch_label_fontsize=6)

        # You can add a title to the plot if you like.
        ax.set_title("Phylogenetic Tree", fontsize=16)

        # We need to adjust the plot to make sure the labels fit.
        plt.tight_layout()

        # Save the plot to the specified output file.
        # The 'bbox_inches="tight"' argument ensures no part of the plot is cut off.
        plt.savefig(output_image_path, bbox_inches="tight")

        print(f"Tree visualization saved successfully to '{output_image_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
