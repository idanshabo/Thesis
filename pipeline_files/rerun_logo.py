import os
import sys
import json
from Bio import SeqIO

# 1. Point Python to your Thesis repo so it can find your local modules
repo_path = os.path.expanduser("~/Documents/Thesis/pipeline_outputs/Thesis")
sys.path.append(repo_path)

# Import your newly fixed plotting function
from significant_split_evaluation.structures.plot_comparative_logos import generate_comparative_logos

def rerun_just_the_plot():
    # =====================================================================
    # MAC DIRECTORY SETUP
    # =====================================================================
    TARGET_FAMILY = "PF01340"
    SPLIT_FOLDER = "rank2"
    
    # Base path for your data
    base_data_path = os.path.expanduser("~/Documents/Thesis/pipeline_outputs")
    
    # Construct exact paths
    fasta_path = os.path.join(base_data_path, f"{TARGET_FAMILY}_calculations", "sequence_embeddings_subfamilies", "subfamily_1", "subfamily_1.fasta")
    
    split_dir = os.path.join(base_data_path, f"{TARGET_FAMILY}_outputs", "sequence_embeddings", "subfamily_1", "significant_splits", SPLIT_FOLDER)
    dir_predicted = os.path.join(base_data_path, f"{TARGET_FAMILY}_calculations", "structures", "predicted_esm")
    output_png = os.path.join(split_dir, "comparative_sequence_logos_FIXED.png")
    # =====================================================================

    print(f"Loading FASTA from:\n  {fasta_path}")
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print(f"\n[Error] Could not find FASTA. Double check the path!")
        return

    # Auto-detect the JSON file in the split folder
    print(f"Looking for JSON in:\n  {split_dir}")
    try:
        json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
        if not json_files:
            print("\n[Error] No JSON file found in the rank2 folder!")
            return
            
        split_json = os.path.join(split_dir, json_files[0])
        print(f"  -> Found JSON: {json_files[0]}")
        
        with open(split_json, 'r') as f:
            split_data = json.load(f)
    except FileNotFoundError:
        print(f"\n[Error] Folder does not exist. Double check the path!")
        return

    group_a_ids = split_data.get('group_a', [])
    group_b_ids = split_data.get('group_b', [])

    print(f"\nFound {len(group_a_ids)} sequences in Group A and {len(group_b_ids)} in Group B.")
    print("Generating Fixed Logo Plot... (This may take a few seconds)")

    generate_comparative_logos(
        records=records, 
        group_a_ids=group_a_ids, 
        group_b_ids=group_b_ids, 
        dir_predicted=dir_predicted, 
        output_path=output_png, 
        highlight_threshold=0.6
    )
    
    print(f"\nDone! Check your folder for the new image:\n  {output_png}")
    
if __name__ == "__main__":
    rerun_just_the_plot()
