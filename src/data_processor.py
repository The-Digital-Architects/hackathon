import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WildfireData:
    def __init__(self, fire_data_path, state_data_path, weather_data_path, coordinates_path, zero_submission_path):
        self.scaler = StandardScaler()
        self.data, self.X_train, self.X_val, self.X_test, self.y_train, self.y_val = None, None, None, None, None, None

        self.load_data(fire_data_path, state_data_path, weather_data_path, coordinates_path, zero_submission_path)
        self.target_col = 'total_fire_size'
        self.features = self.data.columns.drop(self.target_col)

    def filter_features(self, features):
        self.X_train = self.X_train.filter(items=features)
        self.X_val = self.X_val.filter(items=features)
        self.X_test = self.X_test.filter(items=features)

    def load_data(self, fire_data_path, state_data_path, weather_data_path, coordinates_path, zero_submission_path):
        # Load datasets
        fire_df = pd.read_csv(fire_data_path)
        state_df = pd.read_csv(state_data_path)
        weather_df = pd.read_csv(weather_data_path)
        coordinates_df = pd.read_csv(coordinates_path)
        zero_submission_df = pd.read_csv(zero_submission_path)
        
        # Clean up coordinates data
        coordinates_df = coordinates_df[['state&teritory', 'latitude', 'longitude']].rename(columns={'state&teritory': 'State'})
        
        missing_year_months = zero_submission_df[~zero_submission_df['month'].isin(fire_df['month'])]
        if not missing_year_months.empty:
            fire_df = pd.concat([fire_df, missing_year_months], ignore_index=True)

        # Merge datasets
        state_df = pd.merge(state_df, coordinates_df, on='State', how='left')
        self.data = combine_data(states_df=state_df, weather_df=weather_df, target_df=fire_df)
        
    def prepare_data(self, val_size=0.2):
        # Convert percentage strings to floats
        self.data['Percentage of Federal Land'] = self.data['Percentage of Federal Land'].str.rstrip('%').astype(float) / 100
        self.data['Urbanization Rate (%)'] = self.data['Urbanization Rate (%)'].astype(float) / 100
        
        # Create month & season features
        self.data = self.add_month_feature(self.data)
        self.data = self.add_season_feature(self.data)

        # Split in train & test data
        train_set = self.data[self.data[self.target_col].notna()]
        test_set = self.data[self.data[self.target_col].isna()]
        test_set = test_set[test_set['month_since_epoch'] > 12*(2011-1970)-1]
        
        # Create X_test
        self.X_test = test_set.drop(columns=[self.target_col]).reset_index(drop=True)
        self.X_test = self.X_test.sort_values('month_since_epoch').reset_index(drop=True)
        
        # Modified these lines to use ravel()
        self.y_train = train_set[self.target_col].values.ravel()
        self.X_train = train_set.drop(columns=[self.target_col]).reset_index(drop=True)
        # X = self.data[['year', 'month_num', 'mean_elevation', 'Land Area (sq mi)', 
        #        'Percentage of Federal Land', 'Urbanization Rate (%)']]
        # y = self.data

        # Split train & validation data
        self.X_train, self.X_val, self.y_train, self.y_val = split_data(self.X_train, self.y_train, test_size=val_size, random=False)

        print("Training set: ", len(self.X_train), "samples")
        print("Validation set: ", len(self.X_val), "samples")
        print("Test set: ", len(self.X_test), "samples")

        print("Done preparing data.")

    def preprocess_data(self):
        # Scale features
        self.scaler = self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.X_val = self.scaler.transform(self.X_val)

        # Set the type to np.float32
        self.X_train = self.X_train.astype(np.float32)
        self.X_train = self.X_val.astype(np.float32)
        self.X_train = self.X_test.astype(np.float32)

    def add_month_feature(self, data): 
        dates = pd.to_datetime(data['year_month'])
        data['month_since_epoch'] = ((dates.dt.year - 1970) * 12 + dates.dt.month - 1).astype(int)
        return data
    
    def add_season_feature(self, data): 
        data['month_in_year'] = pd.to_datetime(data['year_month']).dt.month

        conditions = [
            data['month_in_year'].isin([12, 1, 2]),
            data['month_in_year'].isin([3, 4, 5]),
            data['month_in_year'].isin([6, 7, 8]),
            data['month_in_year'].isin([9, 10, 11])
        ]
        choices = ['winter', 'spring', 'summer', 'fall']

        data['season'] = np.select(conditions, choices, default='unknown')

        return data

def split_data(X, y, test_size=0.2, random=False):
    if random: 
        return train_test_split(X, y, test_size=test_size)
    else: 
        return train_test_split(X, y, test_size=test_size, random_state=42)
    

def combine_data(states_df, weather_df, target_df, how='left'):
    target_df = target_df.rename(columns={'STATE': 'State', 'month': 'year_month'})
    combined_df = pd.merge(weather_df, states_df, on='State', how='right')
    combined_df = pd.merge(combined_df, target_df, on=['State', 'year_month'], how=how)
    return combined_df
