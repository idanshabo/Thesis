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
    Selects the best PDB from the API list.
    Priority: 
    1. Method (X-ray > EM > NMR)
    2. Resolution (Lowest number is best)
    """
    if not pdb_list:
        return None

    # Helper to clean resolution string to float
    def parse_resolution(res_str):
        try:
            if res_str == 'N/A' or res_str is None: return 999.9
            return float(str(res_str).replace('A', '').strip())
        except ValueError:
            return 999.9

    # Priority map (Lower score = Better)
    method_priority = {
        'X-ray diffraction': 1,
        'Electron Microscopy': 2,
        'NMR': 3
    }

    # Sort: Primary key = Method, Secondary key = Resolution
    sorted_pdbs = sorted(pdb_list, key=lambda x: (
        method_priority.get(x[1], 4), 
        parse_resolution(x[2])
    ))

    # Return the PDB ID of the top result
    return sorted_pdbs[0][0]


def prepare_experimental_folder(group_a, group_b, output_folder):
    """
    Checks for PDBs for both groups. 
    Only downloads if BOTH groups have >= 2 valid structures.
    Returns: (bool_success, list_valid_a, list_valid_b)
    """
    pdbl = PDBList(verbose=False)
    
    # 1. Map IDs to best PDBs
    def find_matches(id_list):
        valid_ids = [] # The original IDs that have a PDB
        mappings = {}  # Map Original ID -> PDB ID
        
        for full_id in id_list:
            # Assumes ID format like "A0A0X_9EURY_..." where first part is UniProt
            uniprot_id = full_id.split('_')[0] 
            candidates = get_pdb_from_uniprot(uniprot_id)
            
            if candidates:
                best = select_best_pdb(candidates)
                if best:
                    valid_ids.append(full_id)
                    mappings[full_id] = best
        return valid_ids, mappings

    print("  Checking Group A for PDBs...")
    valid_a, map_a = find_matches(group_a)
    print("  Checking Group B for PDBs...")
    valid_b, map_b = find_matches(group_b)

    # 2. Check Constraints (At least 2 in each group)
    if len(valid_a) < 2 or len(valid_b) < 2:
        print(f"  Skipping Experimental Plot. Found: Group A={len(valid_a)}, Group B={len(valid_b)} (Need >=2 each)")
        return False, [], []

    # 3. Download Files
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    all_mappings = {**map_a, **map_b}
    print(f"  Downloading {len(all_mappings)} experimental PDB structures...")

    for original_id, pdb_id in all_mappings.items():
        # We save the file as the Original/Safe ID so the matrix calculator recognizes it
        safe_id = original_id.replace("/", "_")
        target_path = os.path.join(output_folder, f"{safe_id}.pdb")

        if not os.path.exists(target_path):
            try:
                # Retrieve (downloads to format pdbXXXX.ent)
                downloaded = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir=output_folder)
                if os.path.exists(downloaded):
                    # Rename to match our pipeline's expected ID
                    shutil.move(downloaded, target_path)
            except Exception as e:
                print(f"    Failed to download {pdb_id}: {e}")

    # Cleanup 'obsolete' folder if BioPython created it
    if os.path.exists(os.path.join(output_folder, "obsolete")):
        shutil.rmtree(os.path.join(output_folder, "obsolete"))

    return True, valid_a, valid_b
