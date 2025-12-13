import os
import shutil
import time
import requests
from Bio import SeqIO
from Bio.PDB import PDBList

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
    try:
        response = requests.post(url, json=query)
        if response.status_code == 200:
            ids = [item['identifier'] for item in response.json().get('result_set', []) if 'identifier' in item]
            print(f"   [API] Mapping {len(ids)} PDBs to UniProt IDs...")
            return ids
    except Exception as e:
        print(f"   [API Error] RCSB Search failed: {e}")
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
            auth_asym_ids
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
        try:
            response = requests.post("https://data.rcsb.org/graphql", json={"query": query_template, "variables": {"ids": chunk}})
            data = response.json().get('data', {})
            if not data: continue
            
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
        except Exception: pass
    return mappings

def get_uniprot_sequences(uniprot_ids):
    """Downloads sequences for UniProt IDs."""
    print(f"   [API] Downloading {len(uniprot_ids)} UniProt sequences...")
    sequences = {}
    for i, uid in enumerate(uniprot_ids):
        try:
            r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta")
            if r.status_code == 200:
                sequences[uid] = "".join(r.text.split('\n')[1:])
            if i % 20 == 0: time.sleep(0.1) # Rate limit
        except: pass
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
            msa_map[record.id] = matched_pdbs
            
    print(f"   -> {len(msa_map)} MSA sequences have experimental structures.")
    return msa_map

def check_and_download_structures(global_map, group_a_leaves, group_b_leaves, output_folder):
    """
    Checks if groups have >= 2 structures each. If yes, downloads them.
    RETURNS: (Success_Bool, List_IDs_Group_A, List_IDs_Group_B)
    """
    # 1. Filter Map for current groups
    pdbs_a = set()
    for leaf in group_a_leaves:
        if leaf in global_map: pdbs_a.update(global_map[leaf])
        elif normalize_id(leaf) in global_map: pdbs_a.update(global_map[normalize_id(leaf)])
        
    pdbs_b = set()
    for leaf in group_b_leaves:
        if leaf in global_map: pdbs_b.update(global_map[leaf])
        elif normalize_id(leaf) in global_map: pdbs_b.update(global_map[normalize_id(leaf)])

    # 2. Check Criteria
    print(f"      [Structure Check] Group A: {len(pdbs_a)} | Group B: {len(pdbs_b)}")
    
    # FIX: Must return 3 values even on failure
    if len(pdbs_a) < 2:
        print(f"      [Structure Skip] Group A has < 2 structures ({len(pdbs_a)}).")
        return False, [], [] 
    if len(pdbs_b) < 2:
        print(f"      [Structure Skip] Group B has < 2 structures ({len(pdbs_b)}).")
        return False, [], []
        
    # 3. Download
    all_pdbs = list(pdbs_a | pdbs_b)
    print(f"      [Downloading] Retrieving {len(all_pdbs)} unique PDB files...")
    
    os.makedirs(output_folder, exist_ok=True)
    pdbl = PDBList(verbose=False)
    
    for pdb_id in all_pdbs:
        target = os.path.join(output_folder, f"{pdb_id}.pdb")
        if not os.path.exists(target):
            try:
                # BioPython downloads to <output_folder>/pdb<id>.ent or similar
                f = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir=output_folder)
                if os.path.exists(f): 
                    shutil.move(f, target)
            except Exception as e: 
                print(f"      Failed {pdb_id}: {e}")
            
    # Cleanup obsolete folder
    if os.path.exists(os.path.join(output_folder, "obsolete")):
        shutil.rmtree(os.path.join(output_folder, "obsolete"))
        
    return True, list(pdbs_a), list(pdbs_b)
