from typing import List, Dict, Type
import numpy as np
from .transformers.base import BaseTransformer
from .transformers.standard_scaler import StandardScaler
from .transformers.feature_selector import FeatureSelector
from .models import BaseModel, RandomForest, AdaBoost, NeuralNetwork

# Registry of available transformers
TRANSFORMER_REGISTRY: Dict[str, Type[BaseTransformer]] = {
    'standard_scaler': StandardScaler,
    'feature_selector': FeatureSelector,
    # Add more transformers here as needed
}

# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    'random_forest': RandomForest,
    'adaboost': AdaBoost,
    'neural_network': NeuralNetwork,
    # Add more models here as needed
}

class Pipeline:
    def __init__(self, transformers: List[BaseTransformer], model: BaseModel):
        self.transformers = transformers
        self.model = model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Pipeline':
        """Fit the pipeline (transformers and model)."""
        X_transformed = X
        for transformer in self.transformers:
            X_transformed = transformer.fit_transform(X_transformed, y)
        
        self.model.fit(X_transformed, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform data and make predictions."""
        X_transformed = X
        for transformer in self.transformers:
            X_transformed = transformer.transform(X_transformed)
        
        return self.model.predict(X_transformed)
    
    def save(self, path: str) -> None:
        """Save pipeline components."""
        import joblib
        pipeline_state = {
            'transformers': self.transformers,
            'model': self.model
        }
        joblib.dump(pipeline_state, path)
    
    @classmethod
    def load(cls, path: str) -> 'Pipeline':
        """Load pipeline from file."""
        import joblib
        pipeline_state = joblib.load(path)
        return cls(
            transformers=pipeline_state['transformers'],
            model=pipeline_state['model']
        )
