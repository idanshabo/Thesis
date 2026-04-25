import os
import pandas as pd
from Bio import SeqIO
from ete3 import Tree
from phylogenetic_baselines import evaluate_strict_branch_baselines

def load_fasta_to_dict(fasta_path):
    """Loads a FASTA file into an id_to_seq dictionary."""
    id_to_seq = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        # Store both normal and sanitized IDs to prevent mismatch errors
        id_to_seq[str(rec.id)] = str(rec.seq)
        id_to_seq[str(rec.id).replace('/', '_')] = str(rec.seq)
    return id_to_seq

def run_all_baselines(data_dir, output_csv="baseline_comparisons.csv"):
    """
    Loops through all family directories, calculates the random baseline,
    and saves the results to a CSV.
    """
    results = []
    
    # Assuming your data_dir contains subfolders for each Pfam family (e.g., PF01340)
    for family_name in os.listdir(data_dir):
        family_path = os.path.join(data_dir, family_name)
        
        if not os.path.isdir(family_path):
            continue
            
        print(f"Processing baseline for {family_name}...")
        
        # NOTE: Change these filenames if your global tree/fasta are named differently
        tree_path = os.path.join(family_path, f"{family_name}.tree")
        fasta_path = os.path.join(family_path, f"{family_name}.fasta")
        
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
        # Using alpha=0.10 and min_size=20 to match your pipeline defaults
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

    # 3. Save everything to a CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nAll baselines completed! Results saved to {output_csv}")

if __name__ == "__main__":
    # Change "./data" to the path where your 30 Pfam family folders are stored
    TARGET_DIRECTORY = "./data" 
    run_all_baselines(TARGET_DIRECTORY)
