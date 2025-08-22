from ete3 import Tree
import numpy as np
import pandas as pd
import os
import re

def tree_to_covariance_matrix(tree_path, output_path = None):
    if not output_path:
        output_path = os.
    # Load and root the tree
    tree = Tree(tree_path, format=1)
    tree.set_outgroup(tree.get_midpoint_outgroup())  # Or choose a known outgroup

    # Get the actual root node
    root = tree.get_tree_root()

    # Get species (leaf names)
    species = [leaf.name for leaf in tree.get_leaves()]
    
    # Clean species names (column and row names)
    cleaned_species = [re.sub(r'[\\/*?:"<>|]', '_', sp) for sp in species]
    cleaned_species = [re.sub(r'\s+', '_', sp) for sp in cleaned_species]  # Optional: replace spaces with underscores
    
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
                shared_distance = root.get_distance(mrca)
                cov_matrix[i, j] = shared_distance

    # Convert to DataFrame with cleaned species names
    cov_df = pd.DataFrame(cov_matrix, index=cleaned_species, columns=cleaned_species)
    cov_df.to_csv('/content/drive/My Drive/filename.csv', index=False)

    print("Calculated covariance matrix successfully")
    return cov_df
