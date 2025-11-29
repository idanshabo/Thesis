import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tm_heatmap(df, stats, split_pos, output_folder, filename="tm_score_matrix.png"):
    """
    Generates and saves the high-res heatmap.
    """
    n = len(df)
    plt.figure(figsize=(14, 12)) 
    
    # Hide labels if too many proteins
    use_labels = True if n < 50 else False
    
    # Plot
    sns.heatmap(df, cmap="viridis", vmin=0, vmax=1, 
                xticklabels=use_labels, yticklabels=use_labels, 
                cbar_kws={'label': 'TM-Score'})
    
    # Visual Separators
    plt.axhline(split_pos, color='white', linewidth=2, linestyle='--')
    plt.axvline(split_pos, color='white', linewidth=2, linestyle='--')
    
    # Text Annotations
    font_args = {'color': 'white', 'ha': 'center', 'va': 'center', 'fontweight': 'bold', 'fontsize': 14}
    
    # Group A (Top Left)
    plt.text(split_pos/2, split_pos/2, 
             f"Group A\nAvg: {stats['avg_a']:.2f}", **font_args)
    
    # Group B (Bottom Right)
    plt.text(split_pos + (n-split_pos)/2, split_pos + (n-split_pos)/2, 
             f"Group B\nAvg: {stats['avg_b']:.2f}", **font_args)

    # Inter-Group (Bottom Left)
    plt.text(split_pos/2, split_pos + (n-split_pos)/2, 
             f"Inter-Group\nAvg: {stats['avg_inter']:.2f}", **font_args)

    plt.title(f"Structural Similarity Analysis (N={n})", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved High-Res Plot: {save_path}")
    plt.close()
