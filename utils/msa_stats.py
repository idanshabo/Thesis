import numpy as np
import random
from itertools import combinations
from Bio import SeqIO, AlignIO
from ete3 import Tree

def calc_msa_similarity(fasta_path, max_pairs=1000):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) < 2: return 100.0
    seqs = [str(rec.seq) for rec in records]
    pairs = list(combinations(range(len(seqs)), 2))
    if len(pairs) > max_pairs: pairs = random.sample(pairs, max_pairs)
    total_sim = sum(sum(1 for a, b in zip(seqs[i], seqs[j]) if a == b) / max(len(seqs[i]), 1) for i, j in pairs)
    return (total_sim / len(pairs)) * 100.0

def calc_norm_branch_length(tree_path):
    try: t = Tree(tree_path, format=1)
    except Exception: t = Tree(tree_path, format=0)
    total_dist = sum([node.dist for node in t.traverse() if not node.is_root()])
    num_leaves = len(t.get_leaves())
    return total_dist / num_leaves if num_leaves > 0 else 0.0

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
