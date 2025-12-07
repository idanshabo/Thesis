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
