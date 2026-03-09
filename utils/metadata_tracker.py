import time
import json
import os
import random
from itertools import combinations
from Bio import Phylo


class MetadataTracker:
    def __init__(self, output_path):
        self.output_path = output_path
        self.metadata = {
            "timings": {},
            "msa_stats": {},
            "pipeline_stats": {},
            "split_analysis": {}
        }
        
        # --- NEW: Load existing metadata to prevent overwriting ---
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    existing_data = json.load(f)
                    for key, val in existing_data.items():
                        if key in self.metadata and isinstance(val, dict):
                            self.metadata[key].update(val)
                        else:
                            self.metadata[key] = val
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
                
        self.current_stage = None
        self.start_time = None

    def start_timer(self, stage_name):
        self.current_stage = stage_name
        self.start_time = time.time()
        print(f"--- [Metadata] Starting: {stage_name} ---")

    def stop_timer(self):
        if self.current_stage and self.start_time:
            elapsed = time.time() - self.start_time
            self.metadata["timings"][self.current_stage] = round(elapsed, 2)
            self.save()
            self.current_stage = None

    def add_stat(self, category, key, value):
        if category not in self.metadata:
            self.metadata[category] = {}
        self.metadata[category][key] = value
        self.save()

    def add_split_stat(self, split_name, stats_dict):
        self.metadata["split_analysis"][split_name] = stats_dict
        self.save()

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

    def calc_and_add_sequence_similarity(self, fasta_path):
        """Calculates avg sequence similarity and adds it to metadata."""
        try:
            sequences = []
            with open(fasta_path, 'r') as f:
                seq = []
                for line in f:
                    if line.startswith(">"):
                        if seq:
                            sequences.append("".join(seq))
                            seq = []
                    else:
                        seq.append(line.strip())
                if seq:
                    sequences.append("".join(seq))

            n_seqs = len(sequences)
            if n_seqs < 2:
                self.add_stat("msa_stats", "avg_sequence_similarity_pct", 100.0)
                return

            if n_seqs <= 500:
                pairs = list(combinations(sequences, 2))
            else:
                pairs = []
                for _ in range(10000):
                    i, j = random.sample(range(n_seqs), 2)
                    pairs.append((sequences[i], sequences[j]))

            total_sim = 0
            for seq1, seq2 in pairs:
                min_len = min(len(seq1), len(seq2))
                if min_len == 0:
                    continue
                # Ignore gap-to-gap matches
                matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
                total_sim += (matches / min_len) * 100

            avg_sim = total_sim / len(pairs) if pairs else 0.0
            self.add_stat("msa_stats", "avg_sequence_similarity_pct", round(avg_sim, 2))
        except Exception as e:
            print(f"Warning: Could not calculate sequence similarity: {e}")

    def calc_and_add_tree_stats(self, tree_path):
        """Calculates normalized branch length and adds it to metadata."""
        try:
            tree = Phylo.read(tree_path, "newick")
            total_length = tree.total_branch_length()
            num_terminals = len(tree.get_terminals())
            
            if num_terminals > 0:
                norm_len = total_length / num_terminals
                self.add_stat("msa_stats", "normalized_total_branch_length", round(norm_len, 4))
        except Exception as e:
            print(f"Warning: Could not calculate normalized branch length: {e}")
