"""
Validation Framework for Formal Explainers

This package provides explainer interfaces and implementations for validating
formal explainers (AXp/CXp) as described in the paper.
"""

from .explainer import (
    TargetExplainer,
    ReferenceExplainer,
    SecondaryExplainer
)

from .model import MLModel

from .validator import Validator, ValidationResult, ValidationReport

from .types import (
    FeatureId,
    FeatureSet,
    Instance
)

from .domains import (
    FeatureDomains,
    NumericDomain,
    CategoricalDomain
)

__all__ = [
    'TargetExplainer',
    'ReferenceExplainer',
    'SecondaryExplainer',
    'MLModel',
    'Validator',
    'ValidationResult',
    'ValidationReport',
    'FeatureDomains',
    'NumericDomain',
    'CategoricalDomain',
    'FeatureId',
    'FeatureSet',
    'Instance'
]
