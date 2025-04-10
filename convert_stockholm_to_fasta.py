def convert_stockholm_to_fasta(stockholm_file, fasta_file):
    """Convert Stockholm to FASTA format"""
    alignment = AlignIO.read(stockholm_file, "stockholm")
    AlignIO.write(alignment, fasta_file, "fasta")
    print(f"Converted file in stockholm format to fasta format in path {fasta_file}")
