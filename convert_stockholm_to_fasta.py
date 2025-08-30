from Bio import AlignIO
import os

def convert_stockholm_to_fasta(stockholm_file_path, fasta_file_path=None):
    """Convert Stockholm to FASTA format"""
    print("starting to convert Stockholm to fasta/n"
    # Extract PFAM family name from the filename
    pfam_family = os.path.basename(stockholm_file_path).split('.')[0]  # Assuming PFAM name is before the first '.'
    
    # Create the directory if it doesn't exist
    pfam_folder_path = os.path.join(os.path.dirname(stockholm_file_path), pfam_family)
    os.makedirs(pfam_folder_path, exist_ok=True)
    
    if not fasta_file_path:
        # Check if 'alignment' is in the file name
        if 'alignment' not in os.path.basename(stockholm_file_path):
            raise ValueError(f"❌ Error: The file path must contain 'alignment'. Got: {stockholm_file_path}")
        
        # Create the base path and set the fasta file path
        base_path = os.path.splitext(stockholm_file_path)[0].replace('.alignment', '')
        fasta_file_path = os.path.join(pfam_folder_path, pfam_family + '.fasta')

    
    if os.path.exists(fasta_file_path):
        print(f"fasta format already exists in path {fasta_file_path}")
        return(fasta_file_path)
    alignment = AlignIO.read(stockholm_file_path, "stockholm")
    AlignIO.write(alignment, fasta_file_path, "fasta")
    print(f"Converted file in stockholm format to fasta format in path {fasta_file_path}")
    return(fasta_file_path)
