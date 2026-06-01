# filename: src/utils/helpers.py
# purpose:  Shared utilities — NumpyEncoder for JSON serialization, serialize_ticket_id.
# version:  1.0

import datetime
import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder for numpy scalars, numpy arrays, Path, datetime, and torch.Tensor.

    Order: torch.Tensor -> np.bool_ -> np.integer -> np.floating -> np.ndarray
           -> Path -> datetime -> .item() fallback.
    np.bool_ MUST be before np.integer (subclasses it in numpy < 2.0).
    torch checked via class-level constant — not inside default() — avoids repeated
    sys.modules lookups when serializing large feature batches.

    Usage: json.dumps(obj, cls=NumpyEncoder)
    """
    try:
        import torch as _torch
        _TORCH_AVAILABLE = True
    except ImportError:
        _torch = None
        _TORCH_AVAILABLE = False

    def default(self, obj):
        if self._TORCH_AVAILABLE and isinstance(obj, self._torch.Tensor):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if hasattr(obj, "item"):          # generic numpy scalar fallback for future dtypes
            return obj.item()
        return super().default(obj)


def serialize_ticket_id(tid) -> "int | str":
    """
    Serialize a Ticket ID for JSON storage.
    Returns Python int if possible (cleaner JSON diffs), else str.
    Used in split_indices.json and Airflow retraining DAG — consistent type
    across all callers prevents int vs str key mismatch in Colab .isin() lookups.
    """
    try:
        return int(tid)
    except (ValueError, TypeError):
        return str(tid)
