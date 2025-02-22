import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WildfireDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_prepare_data(self, fire_data_path, state_data_path):
        # Load datasets
        fire_df = pd.read_csv(fire_data_path)
        state_df = pd.read_csv(state_data_path)
        
        # Convert percentage strings to floats
        state_df['Percentage of Federal Land'] = state_df['Percentage of Federal Land'].str.rstrip('%').astype(float) / 100
        state_df['Urbanization Rate (%)'] = state_df['Urbanization Rate (%)'].astype(float)
        
        # Create month and year features
        fire_df['year'] = pd.to_datetime(fire_df['month']).dt.year
        fire_df['month_num'] = pd.to_datetime(fire_df['month']).dt.month
        
        # Merge the datasets
        df = fire_df.merge(state_df, left_on='STATE', right_on='State')
        
        # Create features and target
        self.feature_names = ['year', 'month_num', 'mean_elevation', 'Land Area (sq mi)', 
                            'Percentage of Federal Land', 'Urbanization Rate (%)']
        X = df[self.feature_names]
        y = df['total_fire_size']
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_names
        )
        
        return X_scaled, y, df  # Also return original df for plotting
    
    def split_data(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, random_state=42)
