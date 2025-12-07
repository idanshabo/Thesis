from Bio import Phylo
import matplotlib.pyplot as plt
import os

def visualize_newick_tree(file_path: str, save_image: bool = True, output_image_path: str = None):
    """
    Reads a phylogenetic tree from a Newick file, visualizes it using biopython and matplotlib,
    and saves the image. The visualization is customized to improve label readability, with dynamic
    resizing of the figure based on the number of entries in the tree.

    Args:
        file_path (str): The path to the input Newick file.
        output_image_path (str): The desired path for the output image file (e.g., 'my_tree.png').
    """
    try:
        # Load the tree from the Newick file using biopython's read function.
        tree = Phylo.read(file_path, "newick")

        # Count the number of clades (entries) in the tree
        def count_clades(clade):
            # Count the clades in the tree
            count = 1  # Count the current clade
            for subclade in clade:
                count += count_clades(subclade)
            return count

        num_clades = count_clades(tree.root)

        # Dynamically adjust figsize based on the number of clades
        figsize = (15, num_clades / 5)  # You can adjust the scaling factor (here it's 5)

        # Create a matplotlib figure and axes with a dynamically adjusted size
        fig, ax = plt.subplots(figsize=figsize)

        # Function to get the label and rotate it for better readability
        def get_label(clade):
            # Only display labels for terminal clades (leaves)
            if clade.is_terminal() and clade.name:
                # Set a more suitable position for the label (adjust for vertical and horizontal position)
                label_position = (clade.branch_length, 0)
                
                # If the branch is long, rotate the label for better readability
                rotation_angle = 0
                if len(clade.name) > 10:  # Adjust rotation condition based on label length
                    rotation_angle = 45  # Rotate if the name is long
                
                return plt.text(label_position[0], label_position[1], clade.name,
                                rotation=rotation_angle, va="bottom", ha="left", fontsize=8)
            return ''

        # Draw the tree on the axes with custom labels and branch lengths
        Phylo.draw(tree, axes=ax, do_show=False,
                   branch_labels=lambda c: f"{c.branch_length:.2f}" if c.branch_length is not None else "")

        # Add a title to the plot
        ax.set_title("Phylogenetic Tree", fontsize=16)

        # Adjust the plot layout
        plt.tight_layout()
        if not output_image_path:
            output_image_path = os.path.dirname(file_path) + "/tree_visualization.png"
        if save_image:
            # Save the plot to the specified output file
            plt.savefig(output_image_path, bbox_inches="tight")
            print(f"Tree visualization saved successfully to '{output_image_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
