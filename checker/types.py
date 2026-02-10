from typing import Set, Tuple

import numpy as np

FeatureId = int
FeatureSet = Set[FeatureId]
Instance = Tuple[np.ndarray, int]  # (x, y), x shape (m,)
