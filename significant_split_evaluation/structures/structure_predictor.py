import os
import re
import time
import requests

def get_structure_from_esm(sequence, output_filename, timeout=60):
    """
    Fetches predicted 3D structure from ESMFold API.
    """
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    raw_seq = str(sequence).upper()
    clean_seq = re.sub(r'[^A-Z]', '', raw_seq)
    clean_seq = re.sub(r'[BJOUZ]', 'X', clean_seq)
    
    if len(clean_seq) == 0: return False
    if len(clean_seq) > 400: return False

    try:
        response = requests.post(url, data=clean_seq, timeout=timeout)
        if response.status_code == 200:
            with open(output_filename, 'w') as f:
                f.write(response.text)
            return True
        else:
            print(f"    [Error {response.status_code}]", end=" ")
            return False
            
    except Exception as e:
        print(f"    [Exception: {e}]", end=" ")
        return False

def run_prediction_batch(records, output_folder, allow_list=None):
    """
    Main loop for processing structures. 
    allow_list: If provided, only IDs in this list are processed.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Filter records based on allow_list
    if allow_list is not None:
        target_records = [r for r in records if r.id in allow_list]
        print(f"--- Prediction Mode: Sampling ({len(target_records)} proteins) ---")
    else:
        target_records = records
        print(f"--- Prediction Mode: Full Batch ({len(target_records)} proteins) ---")

    total_seqs = len(target_records)
    failed_records = []

    # 1. Main Pass
    for i, record in enumerate(target_records):
        safe_id = record.id.replace("/", "_")
        output_path = os.path.join(output_folder, f"{safe_id}.pdb")
        
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
            failed_records.append(record)

    # 2. Retry Pass
    if failed_records:
        print(f"\n--- Retrying {len(failed_records)} failed sequences ---")
        for i, record in enumerate(failed_records):
            safe_id = record.id.replace("/", "_")
            output_path = os.path.join(output_folder, f"{safe_id}.pdb")
            
            print(f"[Retry {i+1}] {safe_id}...", end=" ")
            # Increased timeout for retries
            success = get_structure_from_esm(record.seq, output_path, timeout=120)
            
            if success:
                print("Recovered!")
                time.sleep(1)
            else:
                print("Failed again.")
