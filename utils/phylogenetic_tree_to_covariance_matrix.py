import os
import shutil
import re
import numpy as np
import pandas as pd
from ete3 import Tree

def tree_to_covariance_matrix(tree_path, output_path=None, chunk_size=50):
    if not output_path:
        base_path = os.path.splitext(tree_path)[0].replace('.tree', '')
        output_path = base_path + '_cov_mat.csv'
    else:
        base_path = os.path.splitext(output_path)[0]

    if os.path.exists(output_path):
        print(f"Covariance matrix already exists in path {output_path}")
        return output_path

    print("Starting to calculate covariance matrix with checkpointing...\n")
    
    # --- 1. Load Tree and Prep Names ---
    tree = Tree(tree_path, format=1)
    tree.set_outgroup(tree.get_midpoint_outgroup()) 
    root = tree.get_tree_root()
    
    species = [leaf.name for leaf in tree.get_leaves()]
    
    # Clean species names
    cleaned_species = [re.sub(r'[\\/*?:"<>|]', '_', sp) for sp in species]
    cleaned_species = [re.sub(r'\s+', '_', sp) for sp in cleaned_species]
    
    n = len(species)
    
    # --- 2. Setup Chunk Directory ---
    chunks_dir = base_path + "_temp_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    print(f"Storing intermediate chunks in: {chunks_dir}")

    # --- 3. Process in Batches ---
    # We iterate through the rows (species 1) in blocks of `chunk_size`
    for start_idx in range(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        chunk_filename = f"chunk_{start_idx}_{end_idx}.csv"
        chunk_path = os.path.join(chunks_dir, chunk_filename)
        
        # Check if this chunk is already done
        if os.path.exists(chunk_path):
            print(f"Batch {start_idx}-{end_idx}/{n} already exists. Skipping.")
            continue
            
        print(f"Calculating batch {start_idx}-{end_idx}/{n}...")
        
        # Initialize sub-matrix for this batch: (Batch Size x N)
        batch_rows = species[start_idx:end_idx]
        batch_matrix = np.zeros((len(batch_rows), n))
        
        # Calculation logic
        for i, sp1 in enumerate(batch_rows):
            global_row_idx = start_idx + i
            for j, sp2 in enumerate(species):
                if global_row_idx == j:
                    batch_matrix[i, j] = root.get_distance(sp1)
                else:
                    # Note: get_common_ancestor is the bottleneck. 
                    # Consider caching paths if this is still too slow.
                    mrca = tree.get_common_ancestor(sp1, sp2)
                    batch_matrix[i, j] = root.get_distance(mrca)
        
        # Save this chunk immediately
        # We save without headers/index to keep merging simple later, 
        # or verify data integrity easily.
        pd.DataFrame(batch_matrix).to_csv(chunk_path, index=False, header=False)

    # --- 4. Merge Chunks ---
    print("All chunks calculated. Merging...")
    
    # Read chunks in order
    chunk_frames = []
    for start_idx in range(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        chunk_filename = f"chunk_{start_idx}_{end_idx}.csv"
        chunk_path = os.path.join(chunks_dir, chunk_filename)
        
        # Load chunk (header=None because we saved without headers)
        chunk_frames.append(pd.read_csv(chunk_path, header=None))
        
    # Concatenate all rows
    cov_df = pd.concat(chunk_frames, axis=0, ignore_index=True)
    
    # Clean up temp folder upon successful merge (Optional: comment out if you want to keep debug files)
    shutil.rmtree(chunks_dir)

    # --- 5. Finalize and Save ---
    # Assign names
    cov_df.index = cleaned_species
    cov_df.columns = cleaned_species
    
    # Regularize
    cov_df = assure_cov_mat_positive_definite(cov_df)
    
    # Save final output
    cov_df.to_csv(output_path, index=True)
    print(f"Calculated and saved covariance matrix successfully to {output_path}")
    
    return output_path
