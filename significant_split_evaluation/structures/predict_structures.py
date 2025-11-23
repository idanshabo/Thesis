import re
import requests
import os
import time
from Bio import SeqIO


def get_structure_from_esm(sequence, output_filename):
    """
    Fetches predicted 3D structure from ESMFold API for a given sequence.
    Robustly handles sequence cleaning to prevent 422 errors.
    """
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    # 1. Convert to string and upper case
    raw_seq = str(sequence).upper()
    
    # 2. Strictly remove any character that is NOT a letter (removes gaps -, ., numbers)
    clean_seq = re.sub(r'[^A-Z]', '', raw_seq)
    
    # 3. Replace ambiguous amino acids that ESMFold API often rejects (B, J, O, U, Z)
    clean_seq = re.sub(r'[BJOUZ]', 'X', clean_seq)
    
    # 4. Validation
    if len(clean_seq) == 0:
        print(f"  -> Skipping {output_filename}: Sequence is empty after cleaning.")
        return False
        
    if len(clean_seq) > 400:
        print(f"  -> Skipping {output_filename}: Sequence too long ({len(clean_seq)} aa)")
        return False

    try:
        # Send plain string in body
        response = requests.post(url, data=clean_seq, timeout=60)
        
        if response.status_code == 200:
            with open(output_filename, 'w') as f:
                f.write(response.text)
            return True
        else:
            print(f"  -> Error fetching {output_filename}: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  -> Exception fetching {output_filename}: {e}")
        return False


def process_fasta_to_structures(fasta_path, output_folder):
    """
    Iterates over a FASTA file, generates ESMFold structures, 
    and saves them to the output folder.
    """
    
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    # 2. Parse the FASTA file
    # list() loads it all into memory so we know the total count for the progress bar
    print(f"Reading FASTA file: {fasta_path}...")
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print("Error: FASTA file not found.")
        return

    total_seqs = len(records)
    print(f"Found {total_seqs} sequences. Starting processing...\n")

    # 3. Loop through sequences
    for i, record in enumerate(records):
        # Clean the ID to be filename-safe (replacing / with _)
        safe_id = record.id.replace("/", "_")
        
        # Create full output path
        output_path = os.path.join(output_folder, f"{safe_id}.pdb")
        
        # Progress indicator
        print(f"[{i+1}/{total_seqs}] Processing: {safe_id}")

        # Check if file already exists to avoid re-downloading
        if os.path.exists(output_path):
            print(f"  -> File already exists. Skipping.")
            continue

        # Call your function
        success = get_structure_from_esm(record.seq, output_path)
        
        # Polite rate limiting: Sleep briefly if successful to avoid hammering the API
        if success:
            time.sleep(0.5) 

    print("\n--- Batch Processing Complete ---")
