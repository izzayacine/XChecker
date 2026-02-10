"""
Explainer interfaces for explainers and ML models.

This module defines the required queries for each type of explainer
as described in the validation framework paper.
"""

from abc import ABC, abstractmethod
from typing import List, Set, Optional, Tuple, Any
import numpy as np

from .domains import FeatureDomains
from .model import MLModel
from .types import Instance, FeatureSet


class FormalExplainer(ABC):

    def __init__(self, model: MLModel):
        self.model = model

class TargetExplainer(FormalExplainer):
    """
    Interface for untrusted explainer T.
    
    Required queries:
    - findaxp_T: returns a set declared to be an AXp
    - findcxp_T: returns a set declared to be a CXp
    - witness_T: returns a witness point in feature space
    """
    
    def __init__(self, model: MLModel, domains: FeatureDomains):
        super().__init__(model)
        self.feature_domains = domains

    @abstractmethod
    def findaxp(self, instance: Instance) -> Set[int]:
        """
        Find an abductive explanation (AXp) for the given instance.
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            
        Returns:
            Set of feature indices forming the AXp
        """
        pass
    
    @abstractmethod
    def findcxp(self, instance: Instance) -> Set[int]:
        """
        Find a contrastive explanation (CXp) for the given instance.
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            
        Returns:
            Set of feature indices forming the CXp
        """
        pass
    
    @abstractmethod
    def witness(self, 
                expl: Set[int], 
                target_classes: Optional[Set[int]] = None) -> Optional[np.ndarray]:
        """
        Generate a witness point in feature space.
        
        Args:
            expl: Set of feature indices to consider
            target_classes: Optional set of target class labels. If None, can be
                          obtained from the classifier/model.
            
        Returns:
            Witness point (feature vector) or None if not found
        """
        pass


class ReferenceExplainer(TargetExplainer):
    """
    Interface for reference explainer R.
    
    Required queries (superset of T):
    - findaxp_R: returns a set declared to be an AXp
    - findcxp_R: returns a set declared to be a CXp
    - witness_R: returns a witness point in feature space    
    - iswaxp_R: checks whether a set is a weak AXp
    - iswcxp_R: checks whether a set is a weak CXp
    - prooftrace_R: returns a proof trace for WAXp
    """
 
    @abstractmethod
    def iswaxp(self, instance: Instance,
              expl: FeatureSet) -> Tuple[bool, Optional[np.ndarray], Optional[Any]]:
        """
        Check whether a set of features is a weak AXp (WAXp).
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            explanation: Set of feature indices to check
            
        Returns:
            Tuple of (is_waxp, witness, proof_trace)
            - is_waxp: True if explanation form a WAXp
            - witness: If not WAXp, a witness point showing different prediction
            - proof_trace: If WAXp, a proof trace justifying the WAXp
        """
        pass
    
    @abstractmethod
    def iswcxp(self, instance: Instance,
              expl: FeatureSet) -> Tuple[bool, Optional[np.ndarray], Optional[Any]]:
        """
        Check whether a set of features is a weak CXp (WCXp).
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            explanation: Set of feature indices to check
            
        Returns:
            Tuple of (is_wcxp, witness, proof_trace)
            - is_wcxp: True if features form a WCXp
            - witness: If WCXp, a witness point showing different prediction
            - proof_trace: If not WCXp, a proof trace justifying why no witness exists
        """
        pass
    
    @abstractmethod
    def prooftrace(self, instance: Instance,
                  expl: FeatureSet) -> Optional[Any]:
        """
        Get a proof trace for why features form a WAXp.
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            explanation: Set of feature indices
            
        Returns:
            Proof trace (format depends on implementation) or None
        """
        pass


class SecondaryExplainer(FormalExplainer):
    """
    Interface for second (independent) explainer S.
    
    Required queries:
    - iswaxp_S: checks whether a set is a weak AXp
    - iswcxp_S: checks whether a set is a weak CXp
    """
    
    @abstractmethod
    def iswaxp(self, instance: Instance,
              expl: FeatureSet) -> bool:
        """
        Check whether a set of features is a weak AXp (WAXp).
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            explanation: Set of feature indices to check
            
        Returns:
            True if explanation form a WAXp
        """
        pass
    
    @abstractmethod
    def iswcxp(self, instance: Instance,
              expl: FeatureSet) -> bool:
        """
        Check whether a set of features is a weak CXp (WCXp).
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            explantion: Set of feature indices to check
            
        Returns:
            True if explanation form a WCXp
        """
        pass
