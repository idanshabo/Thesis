import os
import shutil
import time
import requests
from Bio import SeqIO
# Note: PDBList is removed because we are using a custom direct downloader

MAX_RETRIES = 3

def get_pdbs_from_pfam(pf_id):
    """Queries RCSB to find all PDB IDs associated with a Pfam ID."""
    print(f"   [API] Searching PDB for structures linked to {pf_id}...")
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_annotation.annotation_id",
                "operator": "exact_match",
                "value": pf_id
            }
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=query, timeout=10)
            if response.status_code == 200:
                result_set = response.json().get('result_set', [])
                if result_set is None: return []
                ids = [item['identifier'] for item in result_set if 'identifier' in item]
                print(f"   [API] Mapping {len(ids)} PDBs to UniProt IDs...")
                return ids
        except Exception as e:
            print(f"   [API Error] RCSB Search failed (Attempt {attempt+1}): {e}")
            time.sleep(1)
    return []

def get_pdb_uniprot_mapping(pdb_ids):
    """Maps PDB IDs to UniProt IDs using RCSB GraphQL."""
    mappings = {} 
    query_template = """
    query structure_data($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        polymer_entities {
          rcsb_polymer_entity_container_identifiers {
            reference_sequence_identifiers {
              database_accession
              database_name
            }
          }
        }
      }
    }
    """
    chunk_size = 50
    for i in range(0, len(pdb_ids), chunk_size):
        chunk = pdb_ids[i:i + chunk_size]
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post("https://data.rcsb.org/graphql", 
                                       json={"query": query_template, "variables": {"ids": chunk}},
                                       timeout=10)
                if response.status_code != 200:
                    time.sleep(1)
                    continue
                
                data = response.json().get('data')
                if data is None: continue
                
                for entry in data.get('entries', []):
                    pdb_id = entry['rcsb_id']
                    for entity in entry.get('polymer_entities', []):
                        identifiers = entity.get('rcsb_polymer_entity_container_identifiers')
                        if not identifiers: continue
                        
                        uniprot_id = next((ref['database_accession'] for ref in identifiers.get('reference_sequence_identifiers', []) 
                                         if ref['database_name'] == 'UniProt'), None)
                        
                        if uniprot_id:
                            if uniprot_id not in mappings: mappings[uniprot_id] = set()
                            mappings[uniprot_id].add(pdb_id)
                break # Success
            except Exception: time.sleep(1)
    return mappings

def get_uniprot_sequences(uniprot_ids):
    """Downloads sequences for UniProt IDs."""
    print(f"   [API] Downloading {len(uniprot_ids)} UniProt sequences...")
    sequences = {}
    for i, uid in enumerate(uniprot_ids):
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta", timeout=10)
                if r.status_code == 200:
                    sequences[uid] = "".join(r.text.split('\n')[1:])
                    break
                time.sleep(1)
            except: time.sleep(1)
        if i % 20 == 0: time.sleep(0.1) 
    return sequences

def normalize_id(identifier):
    """Standardizes IDs."""
    return identifier.replace("/", "_")

def prepare_global_structure_map(pfam_id, msa_path):
    """
    Fetches all data once and maps PDBs to MSA Headers.
    """
    if not pfam_id or not os.path.exists(msa_path): 
        print(f"   [Skipping] Missing Pfam ID ({pfam_id}) or MSA file ({msa_path})")
        return {}
    
    # 1. Fetch PDBs
    pdb_ids = get_pdbs_from_pfam(pfam_id)
    if not pdb_ids: return {}
    
    # 2. Map to UniProt
    pdb_map = get_pdb_uniprot_mapping(pdb_ids)
    
    # 3. Get Sequences
    sequences = get_uniprot_sequences(list(pdb_map.keys()))
    
    # 4. Match to MSA
    print(f"   [Matching] Aligning PDB sequences to MSA ({os.path.basename(msa_path)})...")
    msa_map = {}
    for record in SeqIO.parse(msa_path, "fasta"):
        clean_seq = str(record.seq).replace('.', '').replace('-', '').upper()
        
        matched_pdbs = set()
        for uid, full_seq in sequences.items():
            if clean_seq in full_seq or full_seq in clean_seq:
                matched_pdbs.update(pdb_map[uid])
        
        if matched_pdbs:
            # FIX: Store BOTH the raw ID and normalized ID as keys
            # This ensures lookup works whether the input list uses '/' or '_'
            msa_map[record.id] = matched_pdbs
            msa_map[normalize_id(record.id)] = matched_pdbs
            
    print(f"   -> {len(msa_map)} MSA sequences have experimental structures.")
    return msa_map

def download_pdb_smart(pdb_id, output_folder):
    """
    Downloads structure with fallback. 
    1. Tries .pdb format.
    2. If that fails (404), tries .cif format (required for large structures).
    """
    pdb_id = pdb_id.lower()
    
    # Check if already exists
    target_pdb = os.path.join(output_folder, f"{pdb_id}.pdb")
    target_cif = os.path.join(output_folder, f"{pdb_id}.cif")
    if os.path.exists(target_pdb) or os.path.exists(target_cif):
        return True

    # URLs
    url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    url_cif = f"https://files.rcsb.org/download/{pdb_id}.cif"
    
    # Helper for request
    def fetch(url, dest):
        for _ in range(MAX_RETRIES):
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Python script; Academic)'}
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    with open(dest, 'w') as f: f.write(r.text)
                    return True
                elif r.status_code == 404: return False
                else: time.sleep(1)
            except: time.sleep(1)
        return False

    # Attempt 1: PDB
    if fetch(url_pdb, target_pdb): return True
    
    # Attempt 2: CIF (Fallback)
    print(f"      [Smart Download] {pdb_id}.pdb failed, trying .cif...")
    if fetch(url_cif, target_cif): return True
    
    print(f"      [Error] Failed to download {pdb_id}")
    return False

def check_and_download_structures(global_map, group_a_leaves, group_b_leaves, output_folder):
    """
    Checks if groups have sufficient structures. If yes, downloads them.
    """
    print("\n--- DEBUGGING ID MISMATCH ---")
    # 1. Show what keys are in the map (from MSA)
    if global_map:
        print(f"Example Map Key (Raw): '{list(global_map.keys())[0]}'")
    else:
        print("Map is empty.")

    # 2. Show what keys are in your groups (from JSON)
    if group_a_leaves:
        print(f"Example Group A ID:    '{group_a_leaves[0]}'")
        
    # 3. Test normalization manually
    if group_a_leaves and global_map:
        test_id = group_a_leaves[0]
        norm_id = normalize_id(test_id)
        print(f"Testing ID: '{test_id}'")
        print(f"   -> In Map? {test_id in global_map}")
        print(f"   -> Normalized '{norm_id}' in Map? {norm_id in global_map}")
    print("-----------------------------\n")

    # 1. Filter Map for current groups
    pdbs_a = set()
    for leaf in group_a_leaves:
        if leaf in global_map: pdbs_a.update(global_map[leaf])
        elif normalize_id(leaf) in global_map: pdbs_a.update(global_map[normalize_id(leaf)])
        
    pdbs_b = set()
    for leaf in group_b_leaves:
        if leaf in global_map: pdbs_b.update(global_map[leaf])
        elif normalize_id(leaf) in global_map: pdbs_b.update(global_map[normalize_id(leaf)])

    # 2. Check Criteria (RELAXED to 1 for now based on your testing)
    print(f"      [Structure Check] Group A: {len(pdbs_a)} | Group B: {len(pdbs_b)}")
    
    if len(pdbs_a) < 1:
        print(f"      [Structure Skip] Group A has < 1 structure.")
        return False, [], [] 
    if len(pdbs_b) < 1:
        print(f"      [Structure Skip] Group B has < 1 structure.")
        return False, [], []
        
    # 3. Download
    all_pdbs = list(pdbs_a | pdbs_b)
    print(f"      [Downloading] Retrieving {len(all_pdbs)} unique PDB/CIF files...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    downloaded = 0
    for pdb_id in all_pdbs:
        if download_pdb_smart(pdb_id, output_folder):
            downloaded += 1
            
    # Cleanup obsolete folder if BioPython created it previously
    if os.path.exists(os.path.join(output_folder, "obsolete")):
        shutil.rmtree(os.path.join(output_folder, "obsolete"))
        
    return True, list(pdbs_a), list(pdbs_b)
