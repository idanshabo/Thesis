import pandas as pd
import numpy as np
import scipy.spatial.distance as spd
import scipy.cluster.hierarchy as sph
import os

def order_covariance_matrix(cov_mat_path):
    """
    Loads a covariance matrix, clusters it using UPGMA (Average linkage),
    and applies Optimal Leaf Ordering (OLO) to smooth the diagonal.
    """
    dir_path = os.path.dirname(cov_mat_path)
    base, ext = os.path.splitext(os.path.basename(cov_mat_path))
    
    new_name = base + "_ordered" + ext
    output_path = os.path.join(dir_path, new_name)    
    
    try:
        # 1. Load Data
        df = pd.read_csv(cov_mat_path, index_col=0)
        
        # 2. Prepare Distance Matrix
        # Calculate a crude "similarity to distance" conversion
        max_val = df.max().max()
        dist_df = max_val - df
        
        # squareform checks the diagonal; if it's not 0, it throws the error.
        np.fill_diagonal(dist_df.values, 0.0)
        
        # Convert square matrix to condensed distance array (upper triangle)
        # checks=False allows slight numerical noise, but explicit 0.0 is safer
        dist_array = spd.squareform(dist_df.values, checks=False) 
        
        # 3. Compute Linkage (Clustering)
        # Use 'average' (UPGMA)
        Z = sph.linkage(dist_array, method='average')
        
        # 4. Apply Optimal Leaf Ordering (OLO)
        print("Calculating Optimal Leaf Ordering (this may take a moment)...")
        Z_ordered = sph.optimal_leaf_ordering(Z, dist_array)
        
        # 5. Get the new sorted order
        sort_order = sph.leaves_list(Z_ordered)
        
        # 6. Reorder and Save
        df_ordered = df.iloc[sort_order, sort_order]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_ordered.to_csv(output_path)
        print(f"Optimized ordered matrix saved to: {output_path}")
        
    except Exception as e:
        print(f"Failed to order matrix: {e}")
        # Raising the error helps you debug the full traceback if needed
        # raise e 

    return output_path
