from ete3 import Tree
import numpy as np
import pandas as pd
import os

def tree_to_covariance_matrix(tree_path):
    # Load and root the tree
    tree = Tree(tree_path, format=1)
    tree.set_outgroup(tree.get_midpoint_outgroup())  # Or choose a known outgroup

    # Get the actual root node
    root = tree.get_tree_root()

    # Get species (leaf names)
    species = [leaf.name for leaf in tree.get_leaves()]
    n = len(species)

    # Initialize the covariance matrix
    cov_matrix = np.zeros((n, n))

    # Compute shared distance from root to MRCA
    for i, sp1 in enumerate(species):
        for j, sp2 in enumerate(species):
            if i == j:
                cov_matrix[i, j] = root.get_distance(sp1)
            else:
                mrca = tree.get_common_ancestor(sp1, sp2)
                shared_distance = root.get_distance(mrca)  # ✅ CORRECT
                cov_matrix[i, j] = shared_distance

    # Convert to DataFrame
    cov_df = pd.DataFrame(cov_matrix, index=species, columns=species)
    return(cov_df)
