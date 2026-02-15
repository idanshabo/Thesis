from Bio import AlignIO
import numpy as np

def get_msa_stats(msa_path):
    """
    Calculates:
    - Number of sequences
    - Average sequence length
    - Number of sequences with experimental structures (PDB refs)
    """
    try:
        alignment = AlignIO.read(msa_path, "stockholm")
        num_seqs = len(alignment)
        if num_seqs == 0:
            return {}

        avg_len = np.mean([len(record.seq) for record in alignment])
        
        # Count structures based on Stockholm 'DR' tags or description
        structure_count = 0
        for record in alignment:
            has_structure = False
            # Check BioPython's parsed dbxrefs (e.g., PDB; 1abc)
            if hasattr(record, 'dbxrefs'):
                for ref in record.dbxrefs:
                    if 'PDB' in ref or 'pdb' in ref:
                        has_structure = True
                        break
            
            # Fallback: Check description for explicit mention
            if not has_structure and 'PDB' in record.description.upper():
                has_structure = True
                
            if has_structure:
                structure_count += 1

        return {
            "num_sequences": num_seqs,
            "avg_sequence_length": round(avg_len, 2),
            "num_sequences_with_structure": structure_count
        }
    except Exception as e:
        print(f"Error calculating MSA stats: {e}")
        return {}
