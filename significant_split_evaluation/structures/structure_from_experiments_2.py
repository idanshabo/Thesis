import os
import shutil
import json
import time
import requests
import pandas as pd
import numpy as np
import torch
from ete3 import Tree
from Bio import SeqIO
from Bio.PDB import PDBList
from sklearn.decomposition import PCA


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
            return [item['identifier'] for item in response.json().get('result_set', []) if 'identifier' in item]
    except Exception as e:
        print(f"   [API Error] RCSB Search failed: {e}")
    return []

def get_pdb_uniprot_mapping(pdb_ids):
    """Maps PDB IDs to UniProt IDs using RCSB GraphQL."""
    print(f"   [API] Mapping {len(pdb_ids)} PDBs to UniProt IDs...")
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
            if i % 20 == 0: time.sleep(0.1)
        except: pass
    return sequences

def prepare_global_structure_map(pfam_id, msa_path):
    """
    PRE-CALCULATION: Fetches all data once to avoid API spamming during loops.
    Returns: Dict { 'MSA_HEADER': {'1abc', '2xyz'} }
    """
    if not pfam_id or not os.path.exists(msa_path): return {}
    
    # 1. Fetch
    pdb_ids = get_pdbs_from_pfam(pfam_id)
    if not pdb_ids: return {}
    
    pdb_map = get_pdb_uniprot_mapping(pdb_ids)
    sequences = get_uniprot_sequences(list(pdb_map.keys()))
    
    # 2. Match to MSA
    print(f"   [Matching] aligning PDB sequences to MSA ({os.path.basename(msa_path)})...")
    msa_map = {}
    for record in SeqIO.parse(msa_path, "fasta"):
        clean_seq = str(record.seq).replace('.', '').replace('-', '').upper()
        # Find matches
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
    Checks if the specific split has >= 2 structures per group. 
    If yes, downloads them.
    """
    # 1. Filter Map for current groups
    pdbs_a = set()
    for leaf in group_a_leaves:
        if leaf in global_map: pdbs_a.update(global_map[leaf])
        
    pdbs_b = set()
    for leaf in group_b_leaves:
        if leaf in global_map: pdbs_b.update(global_map[leaf])

    # 2. Check Criteria (Smaller first)
    if len(pdbs_a) <= len(pdbs_b):
        p1, n1, p2, n2 = pdbs_a, "Group A", pdbs_b, "Group B"
    else:
        p1, n1, p2, n2 = pdbs_b, "Group B", pdbs_a, "Group A"
        
    if len(p1) < 2:
        print(f"      [Structure Skip] {n1} has < 2 structures ({len(p1)}).")
        return False, []
    if len(p2) < 2:
        print(f"      [Structure Skip] {n2} has < 2 structures ({len(p2)}).")
        return False, []
        
    # 3. Download
    print(f"      [Downloading] Retrieving {len(p1 | p2)} PDB files...")
    os.makedirs(output_folder, exist_ok=True)
    pdbl = PDBList(verbose=False)
    
    downloaded_files = []
    for pdb_id in (p1 | p2):
        target = os.path.join(output_folder, f"{pdb_id}.pdb")
        if not os.path.exists(target):
            try:
                f = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir=output_folder)
                if os.path.exists(f): 
                    shutil.move(f, target)
                    downloaded_files.append(target)
            except: pass
            
    # Cleanup obsolete folder from BioPython
    if os.path.exists(os.path.join(output_folder, "obsolete")):
        shutil.rmtree(os.path.join(output_folder, "obsolete"))
        
    return True, list(p1 | p2)
