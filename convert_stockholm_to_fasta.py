from Bio import AlignIO
import os

def convert_stockholm_to_fasta(stockholm_file_path, fasta_file_path=None):
    """Convert Stockholm to FASTA format"""
    if not fasta_file_path:
        if 'alignment' not in os.path.basename(stockholm_file_path):
            raise ValueError(f"❌ Error: The file path must contain 'alignment'. Got: {stockholm_file_path}")
        base_path = os.path.splitext(stockholm_file_path)[0].replace('.alignment', '')
        fasta_file_path = base_path + '.fasta'
    alignment = AlignIO.read(stockholm_file, "stockholm")
    AlignIO.write(alignment, fasta_file_path, "fasta")
    print(f"Converted file in stockholm format to fasta format in path {fasta_file}")
