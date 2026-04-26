import os
import sys
import json
import pandas as pd
from Bio import SeqIO
from ete3 import Tree

# --- DIRECTORY PATH RESOLUTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.phylogenetic_baselines import evaluate_strict_branch_baselines

def load_fasta_to_dict(fasta_path):
    """Loads a FASTA file into an id_to_seq dictionary."""
    id_to_seq = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        id_to_seq[str(rec.id)] = str(rec.seq)
        id_to_seq[str(rec.id).replace('/', '_')] = str(rec.seq)
    return id_to_seq

def get_k_from_json(out_folder):
    """
    Determines total number of groups (K) by checking both:
    1. subfamilies_summary.json (Phase 1 clustering)
    2. results.json (Phase 2 statistical splits)
    """
    subfam_path = os.path.join(out_folder, "subfamilies_summary.json")
    results_path = os.path.join(out_folder, "results.json")
    
    k_total = 1
    
    # Check Phase 1: Subfamily Clustering
    if os.path.exists(subfam_path):
        try:
            with open(subfam_path, 'r') as f:
                subfam_data = json.load(f)
                if subfam_data:
                    # Count keys like "subfamily_1", "subfamily_2"
                    k_total = len(subfam_data.keys())
        except Exception as e:
            print(f"      Warning: Error reading subfamilies_summary.json: {e}")

    # Check Phase 2: Statistical Splits (LRT)
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results_data = json.load(f)
                if results_data:
                    # Add any additional significant splits found in Phase 2
                    sig_splits = sum([1 for item in results_data if item.get('sig') is True])
                    k_total += sig_splits
        except Exception as e:
            print(f"      Warning: Error reading results.json: {e}")
            
    return max(k_total, 1)

def run_all_baselines(base_dir, output_csv):
    """
    Loops through outputs, determines specific K for Seq and Struct, 
    and generates tailored baselines for both.
    """
    results = []
    
    # Iterate looking for the output folders
    for item in os.listdir(base_dir):
        if item.startswith("PF") and item.endswith("_outputs"):
            family_name = item.replace("_outputs", "")
            print(f"\nProcessing baselines for {family_name}...")
            
            # Resolve Paths
            out_folder = os.path.join(base_dir, item)
            calc_folder = os.path.join(base_dir, f"{family_name}_calculations")
            
            tree_path = os.path.join(calc_folder, f"{family_name}.tree")
            fasta_path = os.path.join(calc_folder, f"{family_name}.fasta")
            
            seq_json = os.path.join(out_folder, "sequence_embeddings", "results.json")
            struct_json = os.path.join(out_folder, "structure_embeddings", "results.json")
            
            if not os.path.exists(tree_path) or not os.path.exists(fasta_path):
                print(f"  -> Skipping {family_name}: Missing tree or fasta.")
                continue
                
            # 1. Load Tree and Sequences
            try:
                tree = Tree(tree_path, format=1)
            except Exception:
                tree = Tree(tree_path, format=0)
                
            id_to_seq = load_fasta_to_dict(fasta_path)
            
            # 2. Determine K targets for this specific family
            seq_k = get_k_from_json(os.path.join(out_folder, "sequence_embeddings"))
            struct_k = get_k_from_json(os.path.join(out_folder, "structure_embeddings"))
            
            print(f"  -> Target Groups: Seq (K={seq_k}), Struct (K={struct_k})")
            
            family_data = {
                "Family": family_name,
                "Seq_K_Groups": seq_k,
                "Struct_K_Groups": struct_k,
            }
            
            # 3. Evaluate Sequence Baseline
            if seq_k > 1:
                seq_base = evaluate_strict_branch_baselines(tree, id_to_seq, target_k=seq_k, min_absolute_size=10, num_trials=100)
                if seq_base:
                    family_data["Seq_Baseline_Sim_Pct"] = round(seq_base['mean_random_sim_pct'], 2)
                    family_data["Seq_Baseline_Branch_Len"] = round(seq_base['mean_random_branch_len'], 4) # <-- NEW LINE
                else:
                    family_data["Seq_Baseline_Sim_Pct"] = None
                    family_data["Seq_Baseline_Branch_Len"] = None # <-- NEW LINE
                    print("     [Seq] Could not find enough valid random branch cuts.")
            else:
                family_data["Seq_Baseline_Sim_Pct"] = "N/A (No Splits)"
                family_data["Seq_Baseline_Branch_Len"] = "N/A"
                
            # 4. Evaluate Structure Baseline
            if struct_k > 1:
                if struct_k == seq_k and seq_base:
                    family_data["Struct_Baseline_Sim_Pct"] = round(seq_base['mean_random_sim_pct'], 2)
                    family_data["Struct_Baseline_Branch_Len"] = round(seq_base['mean_random_branch_len'], 4) # <-- NEW LINE
                else:
                    struct_base = evaluate_strict_branch_baselines(tree, id_to_seq, target_k=struct_k, min_absolute_size=10, num_trials=100)
                    if struct_base:
                        family_data["Struct_Baseline_Sim_Pct"] = round(struct_base['mean_random_sim_pct'], 2)
                        family_data["Struct_Baseline_Branch_Len"] = round(struct_base['mean_random_branch_len'], 4) # <-- NEW LINE
                    else:
                        family_data["Struct_Baseline_Sim_Pct"] = None
                        family_data["Struct_Baseline_Branch_Len"] = None # <-- NEW LINE
                        print("     [Struct] Could not find enough valid random branch cuts.")
            else:
                family_data["Struct_Baseline_Sim_Pct"] = "N/A (No Splits)"
                family_data["Struct_Baseline_Branch_Len"] = "N/A"

            results.append(family_data)

    # 5. Save everything
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nAll tailored baselines completed! Results saved to {output_csv}")

if __name__ == "__main__":
    PIPELINE_OUTPUTS_DIR = os.path.expanduser("~/Documents/Thesis/pipeline_outputs")
    OUTPUT_CSV_PATH = os.path.join(parent_dir, "dual_baseline_comparisons.csv")
    
    run_all_baselines(PIPELINE_OUTPUTS_DIR, OUTPUT_CSV_PATH)
