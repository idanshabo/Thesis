import pandas as pd
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
        # If df is Covariance/Correlation, we need to convert it to "distance"
        # High Covariance = Low Distance. 
        # A simple inversion is often sufficient for sorting: max(df) - df
        # Or simpler: if it's correlation (-1 to 1), use 1 - corr
        dist_array = spd.squareform(df.max().max() - df) 
        
        # 3. Compute Linkage (Clustering)
        # Use 'average' (UPGMA) to respect phylogenetic distances better than 'ward'
        Z = sph.linkage(dist_array, method='average')
        
        # 4. Apply Optimal Leaf Ordering (The Magic Step)
        # This rotates branches to maximize similarity between adjacent leaves
        print("Calculating Optimal Leaf Ordering (this may take a moment)...")
        Z_ordered = sph.optimal_leaf_ordering(Z, dist_array)
        
        # 5. Get the new sorted order
        # leaves_list returns the optimal linear order from left to right
        sort_order = sph.leaves_list(Z_ordered)
        
        # 6. Reorder and Save
        df_ordered = df.iloc[sort_order, sort_order]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_ordered.to_csv(output_path)
        print(f"Optimized ordered matrix saved to: {output_path}")
        
    except Exception as e:
        print(f"Failed to order matrix: {e}")
    return output_path
