import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WildfireData:
    def __init__(self, fire_data_path, state_data_path, weather_data_path, coordinates_path, zero_submission_path):
        self.scaler = StandardScaler()
        self.data, self.X_train, self.X_val, self.X_test, self.y_train, self.y_val = None, None, None, None, None, None
        self.target_col = 'total_fire_size'

        self.load_data(fire_data_path, state_data_path, weather_data_path, coordinates_path, zero_submission_path)
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
        
        # Clean up coordinates data
        coordinates_df = coordinates_df[['state&teritory', 'latitude', 'longitude']].rename(columns={'state&teritory': 'State'})
        
        # Remove any existing missing or zero rows from fire_df
        fire_df = fire_df[fire_df['total_fire_size'] > 0]
        
        # Merge datasets with filled data
        state_df = pd.merge(state_df, coordinates_df, on='State', how='left')
        self.data = combine_data(states_df=state_df, weather_df=weather_df, target_df=fire_df)

    

    def prepare_data(self, val_size=0.2):
        # Convert percentage strings to floats
        self.data['Percentage of Federal Land'] = self.data['Percentage of Federal Land'].str.rstrip('%').astype(float) / 100
        self.data['Urbanization Rate (%)'] = self.data['Urbanization Rate (%)'].astype(float) / 100
        
        # First create train set from existing data
        train_set = self.data[self.data[self.target_col].notna()].copy()
        print(f"Training set size before split: {len(train_set)}")
        print(f"Training target stats:\n{train_set[self.target_col].describe()}")
        
        # Create test set from submission template
        zero_submission = pd.read_csv(self.zero_submission_path)
        test_set = zero_submission[['STATE', 'month']].rename(columns={'STATE': 'State'})
        test_set['year_month'] = test_set['month']  # Set year_month for merging
        
        # Add state features
        state_columns = ['mean_elevation', 'Land Area (sq mi)', 'Water Area (sq mi)', 
                        'Percentage of Federal Land', 'Urbanization Rate (%)', 
                        'latitude', 'longitude']
        state_data = self.data[['State'] + state_columns].drop_duplicates('State')
        test_set = pd.merge(test_set, state_data, on='State', how='left')
        
        # Add weather features
        weather_cols = ['PRCP', 'EVAP', 'TMIN', 'TMAX']
        weather_data = self.data[['State', 'year_month'] + weather_cols].copy()
        test_set = pd.merge(
            test_set,
            weather_data,
            left_on=['State', 'year_month'],
            right_on=['State', 'year_month'],
            how='left'
        )
        
        # Add time-based features to both sets
        for df in [train_set, test_set]:
            df = self.add_month_feature(df)
            df = self.add_season_feature(df)
        
        # Fill missing weather values in test set
        for col in weather_cols:
            # Calculate averages by state and season
            avg_by_state_season = train_set.groupby(['State', 'season'])[col].mean()
            avg_by_state = train_set.groupby('State')[col].mean()
            
            # Fill nulls with state-season average, then state average
            for state in test_set['State'].unique():
                state_mask = test_set['State'] == state
                for season in test_set.loc[state_mask, 'season'].unique():
                    mask = state_mask & (test_set['season'] == season)
                    try:
                        fill_val = avg_by_state_season.loc[(state, season)]
                    except:
                        fill_val = avg_by_state.loc[state]
                    test_set.loc[mask & test_set[col].isna(), col] = fill_val
        
        # Create final splits
        self.X_train = train_set.drop(columns=[self.target_col]).reset_index(drop=True)
        self.y_train = train_set[self.target_col].values.ravel()
        
        self.X_test = test_set.reset_index(drop=True)
        self.X_test = self.X_test.sort_values(['State', 'month']).reset_index(drop=True)
        
        # Split train & validation
        self.X_train, self.X_val, self.y_train, self.y_val = split_data(
            self.X_train, self.y_train, test_size=val_size, random=False
        )
        
        print("\nDataset splits:")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        print("\nFeature statistics:")
        print(self.X_test.describe())
        
        # Verify no missing values
        missing = self.X_test.isna().sum()
        if missing.any():
            print("\nMissing values in test set:")
            print(missing[missing > 0])

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
        if 'month_in_year' not in data.columns:
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
    

def combine_data(states_df, weather_df, target_df):
    target_df = target_df.rename(columns={'STATE': 'State', 'month': 'year_month'})
    combined_df = pd.merge(target_df, states_df, on='State', how='outer')
    combined_df = pd.merge(combined_df, weather_df, on=['State', 'year_month'], how='outer')
    return combined_df
