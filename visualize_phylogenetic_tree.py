# Import the necessary libraries.
from Bio import Phylo
import matplotlib.pyplot as plt

def visualize_phylogenetic_tree(file_path: str, output_path: str = None):
    """
    Reads a phylogenetic tree from a Newick file, visualizes it using biopython and matplotlib,
    and saves the image. The visualization is customized to improve label readability.

    Args:
        file_path (str): The path to the input Newick file.
        output_image_path (str): The desired path for the output image file (e.g., 'my_tree.png').
    """
    if not output_path:
        base_path = os.path.splitext(file_path)[0].replace('.tree', '')
        output_path = base_path + 'phylogenetic_tree_visualization.png'
    if os.path.exists(output_path):
        print(f"phylogenetic tree visualization already exists in path {output_path}")
        return(output_path)
    try:
        tree = Phylo.read(file_path, "newick")

        # Create a matplotlib figure and axes with an increased size for better label spacing.
        fig, ax = plt.subplots(figsize=(15, 15))

        # A function to rotate the labels for better readability if they are overlapping.
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

        ax.set_title("Phylogenetic Tree", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_image_path, bbox_inches="tight")

        print(f"Tree visualization saved successfully to '{output_image_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
