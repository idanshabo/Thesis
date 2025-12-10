from ete3 import Tree
import pandas as pd
import os
import re

def order_covariance_matrix_by_tree(cov_mat_path, tree_path):
    """
    Sorts the covariance matrix based strictly on the leaf order 
    of the phylogenetic tree. This ensures that any clade/group 
    derived from the tree will appear as a contiguous block in the matrix.
    """
    # 1. Load the Tree
    try:
        t = Tree(tree_path, format=1)
        # Ensure the tree is rooted the same way as when you created the matrix
        t.set_outgroup(t.get_midpoint_outgroup()) 
        
        # If you used ladderizing (sorting branches) in your tree viewer, 
        # apply it here to make the matrix look "cleaner"
        t.ladderize() 
        
    except Exception as e:
        print(f"Error loading tree: {e}")
        return

    # 2. Get Leaf Order
    # get_leaves() returns leaves in the visual order they appear in the tree
    leaves = [leaf.name for leaf in t.get_leaves()]
    
    # 3. Clean Names to match your Matrix keys
    # (Copying the logic from your creation function)
    cleaned_leaves = [re.sub(r'[\\/*?:"<>|]', '_', sp) for sp in leaves]
    cleaned_leaves = [re.sub(r'\s+', '_', sp) for sp in cleaned_leaves] 

    # 4. Load Matrix
    df = pd.read_csv(cov_mat_path, index_col=0)

    # 5. Reorder
    # Filter to ensure we only grab leaves that actually exist in the matrix 
    # (in case of previous filtering steps)
    final_order = [x for x in cleaned_leaves if x in df.index]
    
    # Check for missing data
    missing = set(df.index) - set(final_order)
    if missing:
        print(f"Warning: {len(missing)} species in matrix but not in tree leaf list.")
        # Append them at the end or handle as errors
        final_order.extend(list(missing))

    df_ordered = df.loc[final_order, final_order]

    # 6. Save
    dir_path = os.path.dirname(cov_mat_path)
    base, ext = os.path.splitext(os.path.basename(cov_mat_path))
    new_name = base + "_tree_ordered" + ext
    output_path = os.path.join(dir_path, new_name)  
    
    df_ordered.to_csv(output_path)
    print(f"Tree-ordered matrix saved to: {output_path}")
    
    return output_path
