import os
import sys
import pandas as pd
from Bio import SeqIO
from ete3 import Tree

# --- DIRECTORY PATH RESOLUTION ---
# Get the absolute path of the script's directory (pipeline_files)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Thesis)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to Python's path so it can find sibling folders like 'utils'
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now we can safely import from the utils folder!
from utils.phylogenetic_baselines import evaluate_strict_branch_baselines

def load_fasta_to_dict(fasta_path):
    """Loads a FASTA file into an id_to_seq dictionary."""
    id_to_seq = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        id_to_seq[str(rec.id)] = str(rec.seq)
        id_to_seq[str(rec.id).replace('/', '_')] = str(rec.seq)
    return id_to_seq

def run_all_baselines(base_dir, output_csv):
    """
    Loops through all PF*****_calculations directories, calculates the 
    random baseline, and saves the results to a CSV.
    """
    results = []
    
    # List all items in the pipeline_outputs folder
    for item in os.listdir(base_dir):
        # We only care about folders that look like "PF00000_calculations"
        if item.startswith("PF") and item.endswith("_calculations"):
            calc_folder_path = os.path.join(base_dir, item)
            
            if not os.path.isdir(calc_folder_path):
                continue
                
            # Extract the "PF01340" part from "PF01340_calculations"
            family_name = item.replace("_calculations", "")
            print(f"Processing baseline for {family_name}...")
            
            # Construct the paths to the tree and fasta files
            tree_path = os.path.join(calc_folder_path, f"{family_name}.tree")
            fasta_path = os.path.join(calc_folder_path, f"{family_name}.fasta")
            
            if not os.path.exists(tree_path) or not os.path.exists(fasta_path):
                print(f"  -> Skipping {family_name}: Missing tree or fasta file.")
                continue
                
            # 1. Load the Tree and Sequences
            try:
                tree = Tree(tree_path, format=1)
            except Exception:
                tree = Tree(tree_path, format=0)
                
            id_to_seq = load_fasta_to_dict(fasta_path)
            
            # 2. Run the strict baseline evaluator (100 random valid splits)
            baselines = evaluate_strict_branch_baselines(
                tree_node=tree, 
                id_to_seq=id_to_seq, 
                tree_alpha=0.10, 
                min_absolute_size=20, 
                num_trials=100
            )
            
            if baselines:
                results.append({
                    "Family": family_name,
                    "Valid_Edges_Tested": baselines["total_valid_edges_tested"],
                    "Random_Baseline_Similarity_Pct": round(baselines["mean_random_sim_pct"], 2),
                    "Random_Baseline_Branch_Length": round(baselines["mean_random_branch_len"], 4)
                })
                print(f"  -> Baseline Similarity: {baselines['mean_random_sim_pct']:.2f}%")
            else:
                print(f"  -> No valid arbitrary splits found for {family_name} under strict rules.")

    # 3. Save everything to a CSV in the Thesis repo folder
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nAll baselines completed! Results saved to {output_csv}")

if __name__ == "__main__":
    # Point to your calculation folders
    PIPELINE_OUTPUTS_DIR = os.path.expanduser("~/Documents/Thesis/pipeline_outputs")
    
    # Save the output CSV into the main Thesis folder
    OUTPUT_CSV_PATH = os.path.join(parent_dir, "baseline_comparisons.csv")
    
    run_all_baselines(PIPELINE_OUTPUTS_DIR, OUTPUT_CSV_PATH)
