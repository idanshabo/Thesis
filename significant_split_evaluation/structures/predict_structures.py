import re
import requests
import os
import time
from Bio import SeqIO

def get_structure_from_esm(sequence, output_filename, timeout=60):
    """
    Fetches predicted 3D structure from ESMFold API.
    Added 'timeout' parameter to allow flexible wait times.
    """
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    raw_seq = str(sequence).upper()
    clean_seq = re.sub(r'[^A-Z]', '', raw_seq)
    clean_seq = re.sub(r'[BJOUZ]', 'X', clean_seq)
    
    if len(clean_seq) == 0:
        return False
    if len(clean_seq) > 400: # Note: You might want to relax this for the retry if you suspect length is the issue
        return False

    try:
        # Pass the timeout variable here
        response = requests.post(url, data=clean_seq, timeout=timeout)
        
        if response.status_code == 200:
            with open(output_filename, 'w') as f:
                f.write(response.text)
            return True
        else:
            # We explicitly print the status code so we know if it was a 504
            print(f"    [Error {response.status_code}]", end=" ")
            return False
            
    except Exception as e:
        print(f"    [Exception: {e}]", end=" ")
        return False


def process_fasta_to_structures(fasta_path):
    
    output_folder = os.path.join(os.path.dirname(fasta_file_path), 'structures')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Reading FASTA file: {fasta_path}...")
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except FileNotFoundError:
        print("Error: FASTA file not found.")
        return

    total_seqs = len(records)
    print(f"Found {total_seqs} sequences. Starting processing...\n")

    # --- LIST TO STORE FAILURES ---
    failed_records = []

    # 1. MAIN LOOP
    for i, record in enumerate(records):
        safe_id = record.id.replace("/", "_")
        output_path = os.path.join(output_folder, f"{safe_id}.pdb")
        
        # Check existing (Improved logic from previous step)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"[{i+1}/{total_seqs}] {safe_id}: Exists. Skipping.")
            continue

        print(f"[{i+1}/{total_seqs}] Processing: {safe_id}...", end=" ")
        
        success = get_structure_from_esm(record.seq, output_path)
        
        if success:
            print("Done.")
            time.sleep(0.5)
        else:
            print("FAILED.")
            # Add to retry list
            failed_records.append(record)

    # 2. RETRY LOOP (Runs only if there were failures)
    if len(failed_records) > 0:
        print(f"\n--- Retrying {len(failed_records)} failed sequences ---")
        
        for i, record in enumerate(failed_records):
            safe_id = record.id.replace("/", "_")
            output_path = os.path.join(output_folder, f"{safe_id}.pdb")
            
            print(f"[Retry {i+1}/{len(failed_records)}] {safe_id}...", end=" ")
            
            # We increase timeout to 120s for retries in case it was a complexity issue
            success = get_structure_from_esm(record.seq, output_path, timeout=120)
            
            if success:
                print("Recovered!")
                time.sleep(1) # Longer rest after a heavy retry
            else:
                print("Failed again.")

    print("\n--- Batch Processing Complete ---")
