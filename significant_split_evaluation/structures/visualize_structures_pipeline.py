import os
import random
from Bio import SeqIO

# Import the local modules
import structure_predictor as predictor
import structure_analysis as analyzer
import visualization as viz

def visualize_structures_pipeline(fasta_path, split_data):
    """
    Main orchestration function.
    split_data: {'group_a': ['id1'...], 'group_b': ['id2'...]}
    """
    
    output_folder = os.path.join(os.path.dirname(fasta_path), 'structures')
    
    # 1. Parse FASTA
    print(f"Reading FASTA: {fasta_path}")
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print("Error: FASTA file not found.")
        return

    all_ids_in_fasta = set(r.id for r in records)
    total_count = len(records)
    
    # 2. Strategy Logic: Full vs Sampling
    final_processing_list = []
    
    if total_count <= 200:
        print(f"Dataset Size: {total_count} (Small). Processing ALL.")
        final_processing_list = list(all_ids_in_fasta)
    else:
        print(f"Dataset Size: {total_count} (Large). Sampling mode active.")
        
        # Filter split_data to ensure IDs exist in FASTA
        available_a = [x for x in split_data['group_a'] if x in all_ids_in_fasta]
        available_b = [x for x in split_data['group_b'] if x in all_ids_in_fasta]
        
        # Sample 50 from each (or max available)
        sample_a = random.sample(available_a, min(len(available_a), 50))
        sample_b = random.sample(available_b, min(len(available_b), 50))
        
        final_processing_list = sample_a + sample_b
        print(f"Sampled: {len(sample_a)} from Group A, {len(sample_b)} from Group B.")

    # 3. Predict Structures
    # Skipping proteins not in final_processing_list
    predictor.run_prediction_batch(records, output_folder, allow_list=final_processing_list)
    
    # 4. Analyze
    # Filter groups to only include what we actually processed
    analysis_group_a = [x for x in split_data['group_a'] if x in final_processing_list]
    analysis_group_b = [x for x in split_data['group_b'] if x in final_processing_list]
    
    df, stats, split_pos = analyzer.calculate_tm_matrix(analysis_group_a, analysis_group_b, output_folder)
    
    # 5. Visualize
    viz.plot_tm_heatmap(df, stats, split_pos, output_folder)
