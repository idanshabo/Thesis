import json
import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types and torch tensors """
    def default(self, obj):
        # Handle all numpy integers (int8, int16, int32, int64, uint8, etc.)
        if isinstance(obj, np.integer):
            return int(obj)
        # Handle all numpy floats (float16, float32, float64, etc.)
        elif isinstance(obj, np.floating):
            return float(obj)
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle numpy booleans
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle torch tensors
        elif torch.is_tensor(obj):
            if obj.numel() == 1:
                return obj.item()
            return obj.tolist()
            
        return super(NumpyEncoder, self).default(obj)


def save_results_json(results, results_file_path):
    try:
        with open(results_file_path, 'w') as f:
            # cls=NumpyEncoder handles the conversion of float32, int64, bool_, etc.
            json.dump(results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error saving JSON even with Encoder: {e}")
        # Last resort fallback
        with open(results_file_path, 'w') as f:
            f.write(str(results))
