import json
import os


def get_split_info(folder_path):
    sub_files = os.listdir(folder_path)
    json_file = [f for f in sub_files if f.endswith('.json')][0]
    json_file_path = os.path.join(folder_path, json_file)
    with open(json_file_path, 'r') as f:
      split_info = json.load(f)
    return split_info


