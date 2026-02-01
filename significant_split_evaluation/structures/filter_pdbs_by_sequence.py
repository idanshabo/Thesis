import os
import argparse
import sys
from Bio import SeqIO
from Bio.Align import PairwiseAligner
import pymol
from pymol import cmd
from significant_split_evaluation.structures.structure_from_experiments_2 import prepare_global_structure_map


def normalize_id(identifier):
    return identifier.replace("/", "_")


def align_and_crop_single_pdb(pdb_path, target_seq, output_path):
    """
    Aligns a specific PDB file to a target sequence string and saves the cropped domain.
    Returns the path to the cropped file if successful, else returns the original path.
    """
    try:
        # Initialize PyMOL
        pymol.finish_launching(['pymol', '-qc'])
        cmd.reinitialize()
        
        # Load Structure
        cmd.load(pdb_path, "structure")
        
        # Setup Aligner
        aligner = PairwiseAligner()
        aligner.mode = 'local'
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -0.5
        
        chains = cmd.get_chains("structure")
        best_chain = None
        best_score = 0
        best_range = (0, 0) # start, end (residue indices in alignment)

        # Iterate chains to find best sequence match
        for chain in chains:
            # Get sequence from PyMOL
            pdb_seq_str = cmd.get_fastastr(f"structure and chain {chain}")
            if not pdb_seq_str: continue
            
            # fasta string includes header, so skip first line
            clean_pdb_seq = "".join(pdb_seq_str.split('\n')[1:])
            
            # Align
            score = aligner.score(target_seq, clean_pdb_seq)
            if score > best_score:
                best_score = score
                best_chain = chain
                
                alignments = aligner.align(target_seq, clean_pdb_seq)
                if alignments:
                    match = alignments[0]
                    # Get start/end indices in the PDB sequence string
                    pdb_start_idx = match.path[0][1]
                    pdb_end_idx = match.path[-1][1]
                    best_range = (pdb_start_idx, pdb_end_idx)

        # Crop if a good match found
        if best_chain and (best_range[1] - best_range[0] > 10):
            # Map string indices to PDB Residue IDs (resi)
            cmd.select("target_chain", f"structure and chain {best_chain}")
            stored_ids = []
            cmd.iterate(f"target_chain and name CA", "stored_ids.append(resi)", space={'stored_ids': stored_ids})
            
            if stored_ids:
                s_idx = max(0, best_range[0])
                e_idx = min(len(stored_ids), best_range[1])
                
                if e_idx > s_idx:
                    start_resi = stored_ids[s_idx]
                    end_resi = stored_ids[e_idx - 1]
                    
                    # Select the range
                    cmd.select("final_domain", f"chain {best_chain} and resi {start_resi}-{end_resi}")
                    
                    # Save
                    cmd.save(output_path, "final_domain")
                    print(f"    [Cropped] {os.path.basename(pdb_path)} -> Chain {best_chain}:{start_resi}-{end_resi}")
                    return output_path
        
        print(f"    [Crop Warning] No good sequence match found for {os.path.basename(pdb_path)}. Using full.")
        return pdb_path

    except Exception as e:
        print(f"    [Crop Error] {e}")
        return pdb_path
