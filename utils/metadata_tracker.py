import time
import json
import os

class MetadataTracker:
    def __init__(self, output_path):
        self.output_path = output_path
        self.metadata = {
            "timings": {},
            "msa_stats": {},
            "pipeline_stats": {},
            "split_analysis": {}
        }
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
