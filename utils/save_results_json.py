import json
import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types and torch tensors """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif torch.is_tensor(obj):
            if obj.numel() == 1:
                return obj.item()
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_results_json(results_file_path)
    try:
        with open(results_file_path, 'w') as f:
            # cls=NumpyEncoder handles the conversion of float32, int64, bool_, etc.
            json.dump(results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error saving JSON even with Encoder: {e}")
        # Last resort fallback
        with open(results_file_path, 'w') as f:
            f.write(str(results))
