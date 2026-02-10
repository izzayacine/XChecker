"""
Feature domain definitions for witness generation.

This module defines the data structures for representing feature domains,
which are needed to generate valid witness points in feature space.
"""

from dataclasses import dataclass
from typing import List, Union, Set

import numpy as np

# FeatureId = int
# FeatureSet = Set[FeatureId]
# Instance = np.ndarray  # shape (m,)

@dataclass(frozen=True)
class NumericDomain:
    min_: float
    max_: float

@dataclass(frozen=True)
class CategoricalDomain:
    values: List[int]  # or str

FeatureDomains = List[Union[NumericDomain, CategoricalDomain]]
# length = m
