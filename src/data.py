import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataHandler:
    def __init__(self, config = None):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self, path):
        """Load and preprocess data."""
        df = pd.read_csv(path)
        
        # Separate features and target
        target_col = self.config.get('target_column', 'target')
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y.values
    
    def split_data(self, X, y):
        """Split data into train and test sets."""
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
