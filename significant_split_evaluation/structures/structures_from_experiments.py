import re
import os
import shutil
import requests
from Bio.PDB import PDBList


def get_pdb_from_uniprot(uniprot_id):
    """
    Fetches PDB IDs linked to a UniProt ID using the UniProt API.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)
    
    if response.status_code != 200:
        return []

    data = response.json()
    pdb_list = []
    
    # Check if cross-references exist
    if 'uniProtKBCrossReferences' in data:
        for xref in data['uniProtKBCrossReferences']:
            if xref['database'] == 'PDB':
                pdb_id = xref['id']
                # PDB entries often list resolution and method, usually helpful to keep
                method = next((p['value'] for p in xref.get('properties', []) if p['key'] == 'Method'), "Unknown")
                resolution = next((p['value'] for p in xref.get('properties', []) if p['key'] == 'Resolution'), "N/A")
                pdb_list.append((pdb_id, method, resolution))
                
    return pdb_list


def select_best_pdb(pdb_list):
    """
    Selects the best PDB based on Method (X-ray > EM > NMR) and Resolution.
    """
    if not pdb_list: return None

    def parse_resolution(res_str):
        try:
            if res_str == 'N/A' or res_str is None: return 999.9
            return float(str(res_str).replace('A', '').strip())
        except ValueError: return 999.9

    method_priority = {'X-ray diffraction': 1, 'Electron Microscopy': 2, 'NMR': 3}

    sorted_pdbs = sorted(pdb_list, key=lambda x: (
        method_priority.get(x[1], 4), 
        parse_resolution(x[2])
    ))
    return sorted_pdbs[0][0]


def prepare_experimental_folder(group_a, group_b, output_folder):
    """
    Optimized PDB Fetcher:
    1. Checks smaller group first.
    2. Fails immediately if smaller group has < 2 matches.
    3. Downloads only if BOTH groups have >= 2 matches.
    """
    pdbl = PDBList(verbose=False)
    
    # Internal helper to process a specific list of IDs
    def find_matches_in_list(id_list, group_name):
        print(f"  Checking {group_name} ({len(id_list)} sequences)...", end=" ")
        mappings = {}
        for full_id in id_list:
            # Extract UniProt ID (Assumes format Accession_Organism_...)
            uniprot_id = full_id.split('_')[0] 
            candidates = get_pdb_from_uniprot(uniprot_id)
            
            if candidates:
                best = select_best_pdb(candidates)
                if best:
                    mappings[full_id] = best
        
        print(f"Found {len(mappings)} structures.")
        return mappings

    # --- Step 1: Determine Execution Order (Smaller First) ---
    if len(group_a) <= len(group_b):
        primary_list, primary_name = group_a, "Group A"
        secondary_list, secondary_name = group_b, "Group B"
        is_a_first = True
    else:
        primary_list, primary_name = group_b, "Group B"
        secondary_list, secondary_name = group_a, "Group A"
        is_a_first = False

    # --- Step 2: Check Primary (Smaller) Group ---
    primary_map = find_matches_in_list(primary_list, primary_name)
    
    if len(primary_map) < 2:
        print(f"  Stopping: {primary_name} has insufficient structures (<2).")
        return False, [], []

    # --- Step 3: Check Secondary (Larger) Group ---
    secondary_map = find_matches_in_list(secondary_list, secondary_name)
    
    if len(secondary_map) < 2:
        print(f"  Stopping: {secondary_name} has insufficient structures (<2).")
        return False, [], []

    # --- Step 4: Reassign to A and B for clarity ---
    # We need to ensure we return the lists in the correct order (A, B) for the pipeline
    if is_a_first:
        map_a, map_b = primary_map, secondary_map
    else:
        map_a, map_b = secondary_map, primary_map

    # --- Step 5: Download Files ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    all_mappings = {**map_a, **map_b}
    print(f"  Criteria met. Downloading {len(all_mappings)} experimental PDB structures...")

    for original_id, pdb_id in all_mappings.items():
        safe_id = original_id.replace("/", "_")
        target_path = os.path.join(output_folder, f"{safe_id}.pdb")

        if not os.path.exists(target_path):
            try:
                # Retrieve file (BioPython downloads to pdir/pdbXXXX.ent)
                downloaded = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir=output_folder)
                
                # Rename .ent to .pdb and use the ID our pipeline expects
                if os.path.exists(downloaded):
                    shutil.move(downloaded, target_path)
            except Exception as e:
                print(f"    Failed to download {pdb_id}: {e}")

    # Cleanup junk
    if os.path.exists(os.path.join(output_folder, "obsolete")):
        shutil.rmtree(os.path.join(output_folder, "obsolete"))

    # Return the keys (the IDs) so the matrix calculator knows what to load
    return True, list(map_a.keys()), list(map_b.keys())
