import numpy as np
from typing import List, Dict

# Utility: safely order features by a declared schema
def extract_features(payload: Dict, feature_order: List[str]) -> np.ndarray:
    values = []
    for key in feature_order:
        if key not in payload:
            raise ValueError(f"Missing required feature: {key}")
        values.append(float(payload[key]))
    return np.array([values], dtype=np.float32)