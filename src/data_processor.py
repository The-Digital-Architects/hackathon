import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WildfireData:
    def __init__(self, fire_data_path, state_data_path, weather_data_path):
        self.scaler = StandardScaler()
        self.data, self.X_train, self.X_val, self.X_test, self.y_train, self.y_val = None, None, None, None, None, None

        self.load_data(fire_data_path, state_data_path, weather_data_path)

    def load_data(self, fire_data_path, state_data_path, weather_data_path):
        # Load datasets
        fire_df = pd.read_csv(fire_data_path)
        state_df = pd.read_csv(state_data_path)
        weather_df = pd.read_csv(weather_data_path)

        # Merge datasets
        self.data = combine_data(states_df=state_df, weather_df=weather_df, target_df=fire_df)
        
    def prepare_data(self, val_size=0.2):
        # Convert percentage strings to floats
        self.data['Percentage of Federal Land'] = self.data['Percentage of Federal Land'].str.rstrip('%').astype(float) / 100
        self.data['Urbanization Rate (%)'] = self.data['Urbanization Rate (%)'].astype(float) / 100
        
        # Create month and year features
        self.data['year'] = pd.to_datetime(self.data['month']).dt.year
        self.data['month_num'] = pd.to_datetime(self.data['month']).dt.month

        # split in train & test data
        train_set = self.data[self.data['total_fire_size'].notna()]
        test_set = self.data[self.data['total_fire_size'].isna()]
        
        # Create features and target
        self.X_test = test_set.pop('total_fire_size')

        self.y_train = train_set['total_fire_size']
        self.X_train = train_set.pop('total_fire_size')
        # X = self.data[['year', 'month_num', 'mean_elevation', 'Land Area (sq mi)', 
        #        'Percentage of Federal Land', 'Urbanization Rate (%)']]
        # y = self.data

    def preprocess_data():
        # Scale features
        X_scaled = self.scaler.fit_transform(X)


def split_data(X, y, test_size=0.2, random=False):
    if random: 
        return train_test_split(X, y, test_size=test_size)
    else: 
        return train_test_split(X, y, test_size=test_size, random_state=42)
    

def combine_data(states_df, weather_df, target_df, how='left'):
    target_df.columns = ['State', 'year_month', 'total_fire_size']
    combined_df = pd.merge(weather_df, states_df, on='State', how='inner')
    combined_df = pd.merge(combined_df, target_df, on=['State', 'year_month'], how=how)
    return combined_df
