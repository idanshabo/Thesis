import os
import random
from Bio import SeqIO

# Import the local modules
from significant_split_evaluation.structures.structure_predictor import run_prediction_batch
from significant_split_evaluation.structures.structure_analysis import calculate_tm_matrix
from significant_split_evaluation.structures.visualization import plot_tm_heatmap


def normalize_id(identifier):
    """
    Standardizes IDs by replacing typical problem characters.
    Seq/1 -> Seq_1
    """
    return identifier.replace("/", "_")


def visualize_structures_pipeline(fasta_path, split_data):
    """
    Main orchestration function.
    """
    output_folder = os.path.join(os.path.dirname(fasta_path), 'structures')
    
    # 1. Parse FASTA
    print(f"Reading FASTA: {fasta_path}")
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print("Error: FASTA file not found.")
        return

    id_map = {}
    for r in records:
        sanitized = normalize_id(r.id)
        id_map[sanitized] = r.id # Store the original ID so we can look it up later

    total_count = len(records)
    
    # 2. Strategy Logic: Full vs Sampling
    final_processing_list = [] # Stores ORIGINAL FASTA IDs
    
    # Helper to find matching IDs in FASTA regardless of / or _
    def get_valid_ids(input_list):
        valid = []
        for item in input_list:
            # Try 1: Exact match
            if item in id_map.values():
                valid.append(item)
            # Try 2: Sanitized match (The Fix)
            elif normalize_id(item) in id_map:
                valid.append(id_map[normalize_id(item)])
        return valid

    # Get valid IDs for both groups
    valid_a = get_valid_ids(split_data['group_a'])
    valid_b = get_valid_ids(split_data['group_b'])

    if total_count <= 200:
        print(f"Dataset Size: {total_count} (Small). Processing ALL.")
        # We process everyone found in the FASTA
        final_processing_list = list(id_map.values())
    else:
        print(f"Dataset Size: {total_count} (Large). Sampling mode active.")
        
        # Sample from the VALID lists
        sample_a = random.sample(valid_a, min(len(valid_a), 50))
        sample_b = random.sample(valid_b, min(len(valid_b), 50))
        
        final_processing_list = sample_a + sample_b
        print(f"Sampled: {len(sample_a)} from Group A, {len(sample_b)} from Group B.")

    # 3. Predict Structures
    # We pass 'final_processing_list' which now contains IDs exactly as they appear in the FASTA
    run_prediction_batch(records, output_folder, allow_list=final_processing_list)
    
    # 4. Analyze
    # We pass the same list to analysis. 
    # The analysis module handles the file lookup (sanitization) internally.
    
    # Filter groups to only include what we actually processed (intersection)
    processed_set = set(final_processing_list)
    analysis_group_a = [x for x in valid_a if x in processed_set]
    analysis_group_b = [x for x in valid_b if x in processed_set]
    
    df, stats, split_pos = calculate_tm_matrix(analysis_group_a, analysis_group_b, output_folder)
    
    plot_tm_heatmap(df, stats, split_pos, output_folder)
