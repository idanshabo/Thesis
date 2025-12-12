import os
import random
from Bio import SeqIO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the local modules
from significant_split_evaluation.structures.structure_predictor import run_prediction_batch
from significant_split_evaluation.structures.structure_analysis import calculate_tm_matrix
from significant_split_evaluation.structures.visualization import plot_tm_heatmap
from significant_split_evaluation.structures.structures_from_experiments import get_pdb_from_uniprot, select_best_pdb, prepare_experimental_folder


def normalize_id(identifier):
    """
    Standardizes IDs by replacing typical problem characters.
    Seq/1 -> Seq_1
    """
    return identifier.replace("/", "_")


def get_aligned_matrices(cov_path, tm_df, sample_list, sort_by="groups"):
    """
    Returns aligned Covariance and TM dataframes based on the requested sorting mode.
    
    Args:
        sort_by (str): "groups" (default) or "covariance"
    """
    # 1. Load Full Covariance
    try:
        df_cov_full = pd.read_csv(cov_path, index_col=0)
    except Exception as e:
        print(f"Error loading covariance: {e}")
        return None, None

    # 2. Create Mapping for ID matching
    # Map normalized_id -> original_id_in_cov_file
    df_index_map = {normalize_id(x): x for x in df_cov_full.index}
    df_index_set = set(df_cov_full.index)

    # 3. Determine the Order of IDs
    final_order = []
    
    if sort_by == "groups":
        # ORDER 1: Respect the input list (Group A then Group B)
        # We iterate through sample_list and find their match in the covariance matrix
        for sample_id in sample_list:
            if sample_id in df_index_set:
                final_order.append(sample_id)
            elif normalize_id(sample_id) in df_index_map:
                final_order.append(df_index_map[normalize_id(sample_id)])
                
    elif sort_by == "covariance":
        # ORDER 2: Respect the Covariance CSV order
        # We iterate through the CSV index and keep rows that exist in our sample_list
        # This preserves the spectral/clustering order of the CSV
        
        # Create a quick lookup for our samples to filter the big matrix
        sample_set_norm = {normalize_id(s) for s in sample_list}
        
        for cov_id in df_cov_full.index:
            if normalize_id(cov_id) in sample_set_norm:
                final_order.append(cov_id)

    if not final_order:
        print("Warning: No overlap found between samples and covariance matrix.")
        return None, None

    # 4. Reindex Both Matrices to this Order
    # Filter Covariance
    df_cov_aligned = df_cov_full.loc[final_order, final_order]
    
    # Filter/Reorder TM (need to handle normalization mapping back to TM keys)
    # The TM matrix columns are likely the normalized IDs from the prediction step.
    # We need to be careful to map the covariance IDs back to the keys used in tm_df.
    
    # Create map: Covariance_ID -> TM_DataFrame_ID
    # (Assuming TM DF uses the normalized or original IDs from the FASTA)
    tm_keys = []
    valid_final_order = []
    
    for cov_id in final_order:
        # Check if this ID (or its norm) exists in TM columns
        if cov_id in tm_df.columns:
            tm_keys.append(cov_id)
            valid_final_order.append(cov_id)
        elif normalize_id(cov_id) in tm_df.columns:
            tm_keys.append(normalize_id(cov_id))
            valid_final_order.append(cov_id)
            
    # Final consistency check
    df_cov_aligned = df_cov_full.loc[valid_final_order, valid_final_order]
    df_tm_aligned = tm_df.loc[tm_keys, tm_keys]
    
    return df_cov_aligned, df_tm_aligned


def plot_side_by_side(df_cov, df_tm, split_point, output_path, title_suffix=""):
    """
    Generic plotter for two matrices side-by-side.
    If split_point is provided, draws crosshairs and labels.
    If split_point is None, draws heatmaps only (for covariance sorting).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # --- LEFT: COVARIANCE ---
    sns.heatmap(df_cov, cmap='viridis', cbar=True, 
                xticklabels=False, yticklabels=False, square=True, ax=ax1)
    ax1.set_title(f"Covariance Signal\n{title_suffix}", fontsize=14, pad=10)

    # --- RIGHT: TM SCORE ---
    if df_tm is not None:
        sns.heatmap(df_tm, cmap='RdYlBu_r', vmin=0, vmax=1.0, cbar=True,
                    xticklabels=False, yticklabels=False, square=True, ax=ax2)
        ax2.set_title(f"TM-Score Structure\n{title_suffix}", fontsize=14, pad=10)
    
    # --- OVERLAYS (Only if split_point is active) ---
    if split_point is not None and df_tm is not None:
        # We only calculate stats if we have a clean split block structure
        
        # Calc Stats
        avg_a = df_tm.iloc[:split_point, :split_point].values.mean()
        avg_b = df_tm.iloc[split_point:, split_point:].values.mean()
        avg_inter = df_tm.iloc[:split_point, split_point:].values.mean()
        center_b = split_point + ((len(df_tm) - split_point) / 2)

        # Draw Lines on Both
        for ax in [ax1, ax2]:
            color = 'white' if ax == ax1 else 'black'
            ax.axvline(x=split_point, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axhline(y=split_point, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

        # Add Text Labels (Group A/B)
        for ax, color in [(ax1, 'white'), (ax2, 'black')]:
            ax.text(split_point/2, -0.5, "Group A", ha='center', va='bottom', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(-0.5, split_point/2, "Group A", ha='right', va='center', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(center_b, -0.5, "Group B", ha='center', va='bottom', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(-0.5, center_b, "Group B", ha='right', va='center', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')

        # Add Stats to TM Plot (Right)
        ax2.text(split_point/2, split_point/2, f"Avg: {avg_a:.2f}", 
                 ha='center', va='center', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax2.text(center_b, center_b, f"Avg: {avg_b:.2f}", 
                 ha='center', va='center', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax2.text(center_b, split_point/2, f"Inter: {avg_inter:.2f}", 
                 ha='center', va='center', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_side_by_side(df_cov, df_tm, split_point, output_path, title_suffix=""):
    """
    Generic plotter for two matrices side-by-side.
    If split_point is provided, draws crosshairs and labels.
    If split_point is None, draws heatmaps only (for covariance sorting).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # --- LEFT: COVARIANCE ---
    sns.heatmap(df_cov, cmap='viridis', cbar=True, 
                xticklabels=False, yticklabels=False, square=True, ax=ax1)
    ax1.set_title(f"Covariance Signal\n{title_suffix}", fontsize=14, pad=10)

    # --- RIGHT: TM SCORE ---
    if df_tm is not None:
        sns.heatmap(df_tm, cmap='RdYlBu_r', vmin=0, vmax=1.0, cbar=True,
                    xticklabels=False, yticklabels=False, square=True, ax=ax2)
        ax2.set_title(f"TM-Score Structure\n{title_suffix}", fontsize=14, pad=10)
    
    # --- OVERLAYS (Only if split_point is active) ---
    if split_point is not None and df_tm is not None:
        # We only calculate stats if we have a clean split block structure
        
        # Calc Stats
        avg_a = df_tm.iloc[:split_point, :split_point].values.mean()
        avg_b = df_tm.iloc[split_point:, split_point:].values.mean()
        avg_inter = df_tm.iloc[:split_point, split_point:].values.mean()
        center_b = split_point + ((len(df_tm) - split_point) / 2)

        # Draw Lines on Both
        for ax in [ax1, ax2]:
            color = 'white' if ax == ax1 else 'black'
            ax.axvline(x=split_point, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axhline(y=split_point, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

        # Add Text Labels (Group A/B)
        for ax, color in [(ax1, 'white'), (ax2, 'black')]:
            ax.text(split_point/2, -0.5, "Group A", ha='center', va='bottom', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(-0.5, split_point/2, "Group A", ha='right', va='center', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(center_b, -0.5, "Group B", ha='center', va='bottom', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')
            ax.text(-0.5, center_b, "Group B", ha='right', va='center', fontsize=11, fontweight='bold', color=color if ax==ax1 else 'black')

        # Add Stats to TM Plot (Right)
        ax2.text(split_point/2, split_point/2, f"Avg: {avg_a:.2f}", 
                 ha='center', va='center', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax2.text(center_b, center_b, f"Avg: {avg_b:.2f}", 
                 ha='center', va='center', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax2.text(center_b, split_point/2, f"Inter: {avg_inter:.2f}", 
                 ha='center', va='center', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
