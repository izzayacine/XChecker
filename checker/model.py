from abc import ABC, abstractmethod
import numpy as np

class MLModel(ABC):
    """
    Interface for ML model M that can be queried for predictions.
    """
    
    @abstractmethod
    def predict(self, instance: np.ndarray) -> int:
        """
        Query the ML model for a prediction on a given instance.
        
        Args:
            instance: Feature vector (numpy array)
            
        Returns:
            Predicted class label (integer)
        """
        pass
    
    @abstractmethod
    def nfeatures(self) -> int:
        """Return the number of features in the model."""
        pass
    
    @abstractmethod
    def nclasses(self) -> int:
        """Return the number of classes in the model."""
        pass