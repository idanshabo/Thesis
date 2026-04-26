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
    
    # Construct exact paths based on your pipeline's output structure
    fasta_path = os.path.join(base_data_path, f"{TARGET_FAMILY}_calculations", "sequence_embeddings_subfamilies", "subfamily_1", "subfamily_1.fasta")
    
    split_json = os.path.join(base_data_path, f"{TARGET_FAMILY}_outputs", "sequence_embeddings", "subfamily_1", "significant_splits", SPLIT_FOLDER, "split_info.json")
    
    dir_predicted = os.path.join(base_data_path, f"{TARGET_FAMILY}_calculations", "structures", "predicted_esm")
    
    # Save the new fixed PNG right next to the old one in the rank2 folder
    output_png = os.path.join(base_data_path, f"{TARGET_FAMILY}_outputs", "sequence_embeddings", "subfamily_1", "significant_splits", SPLIT_FOLDER, "comparative_sequence_logos_FIXED.png")
    # =====================================================================

    print(f"Loading FASTA from:\n  {fasta_path}")
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print(f"\n[Error] Could not find FASTA. Double check the path!")
        return

    print(f"Loading Split Data from:\n  {split_json}")
    try:
        with open(split_json, 'r') as f:
            split_data = json.load(f)
    except FileNotFoundError:
        print(f"\n[Error] Could not find Split JSON. Double check the path!")
        return

    group_a_ids = split_data.get('group_a', [])
    group_b_ids = split_data.get('group_b', [])

    print(f"\nFound {len(group_a_ids)} sequences in Group A and {len(group_b_ids)} in Group B.")
    print("Generating Fixed Logo Plot... (This may take a few seconds)")

    # Run strictly the logo generation function
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
