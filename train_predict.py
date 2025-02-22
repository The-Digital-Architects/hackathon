from src.data_processor import WildfireDataProcessor
from src.models import WildfirePredictor
import pandas as pd
import numpy as np

def main():

    config = {
        'model_params': {
            'n_estimators': 200,
            'random_state': 42
        }
    }

    # Initialize processors
    data_processor = WildfireDataProcessor()
    model = WildfirePredictor(config).model
    
    # Load and prepare data
    X, y = data_processor.load_and_prepare_data(
        'data/wildfire_sizes_before_2010.csv',
        'data/merged_state_data.csv'
    )
    
    # Train model
    print("Training model...")
    model.fit(X, y)
    
    # Generate future predictions
    state_data = pd.read_csv('data/merged_state_data.csv')  # Changed this line
    predictions = []
    
    for year in range(2011, 2016):
        for month in range(1, 13):
            for _, row in state_data.iterrows():  # Changed this line
                # Create feature vector for prediction
                features = pd.DataFrame([[
                    year,
                    month,
                    row['mean_elevation'],
                    row['Land Area (sq mi)'],
                    float(row['Percentage of Federal Land'].strip('%')) / 100,
                    float(row['Urbanization Rate (%)'])
                ]], columns=X.columns)
                
                # Scale features
                features_scaled = data_processor.scaler.transform(features)
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                
                predictions.append({
                    'ID': len(predictions),
                    'STATE': row['State'],  # Changed this line
                    'month': f"{year}-{month:02d}",
                    'total_fire_size': prediction
                })
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()
