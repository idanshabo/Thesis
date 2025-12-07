import pandas as pd
import scipy.spatial.distance as spd
import scipy.cluster.hierarchy as sph
import os

def order_covariance_matrix(cov_mat_path, method='ward'):
    """
    Loads a covariance/correlation matrix, performs hierarchical clustering 
    to determine a global sort order, and saves the reordered matrix.
    """
    dir_path = os.path.dirname(cov_mat_path)
    base, ext = os.path.splitext(os.path.basename(cov_mat_path))
    
    new_name = base + "_ordered" + ext
    output_path = os.path.join(dir_path, new_name)    
    try:
        # 1. Load Data
        df = pd.read_csv(cov_mat_path, index_col=0)
        
        if df.shape[0] < 2:
            print("Matrix too small to cluster. Saving as is.")
            df.to_csv(output_path)
            return

        # 2. Compute Clustering (Ward's Method)
        linkage = sph.linkage(df, method=method)
        
        # 3. Get the sorted order (leaves)
        dendro = sph.dendrogram(linkage, no_plot=True)
        sort_order = dendro['leaves']
        
        # 4. Reorder the DataFrame
        df_ordered = df.iloc[sort_order, sort_order]
        
        # 5. Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_ordered.to_csv(output_path)
        print(f"Global ordered matrix saved to: {output_path}")
        
    except Exception as e:
        print(f"Failed to order matrix: {e}")
