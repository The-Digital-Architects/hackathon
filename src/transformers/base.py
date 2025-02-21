from typing import Dict, Any, Optional
import numpy as np

class BaseTransformer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.needs_target = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseTransformer':
        """Fit the transformer to the data."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)
