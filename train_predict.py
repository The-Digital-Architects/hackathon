from src.data_processor import WildfireData
from src.models import WildfirePredictor
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def main():
    config = {
        'model_params': {
            'n_estimators': 200,
            'random_state': 42
        }
    }

    # Initialize processors
    data_processor = WildfireData(
        fire_data_path='data/wildfire_sizes_before_2010.csv',
        state_data_path='data/merged_state_data.csv',
        weather_data_path='data/weather_monthly_state_aggregates.csv',
        coordinates_path='data/state_coordinates.csv'
    )

    # Prepare data
    data_processor.prepare_data()

    # drop features: 'year_month', 'month_in_year'
    print(data_processor.data.columns)
    data_processor.filter_features(['PRCP', 'EVAP', 'TMIN', 'TMAX', 'mean_elevation', 'Land Area (sq mi)', 'Water Area (sq mi)', 'Percentage of Federal Land', 'Urbanization Rate (%)', 'latitude', 'longitude', 'total_fire_size'])

    model = WildfirePredictor(config).model

    print(data_processor.X_train.head())
    print(data_processor.y_train)
    print(data_processor.X_val.head())
    print(data_processor.y_val)

    # pipelines
    numerical_features = data_processor.X_train.select_dtypes(include=[np.number]).columns
    # all others are categorical
    categorical_features = data_processor.X_train.columns.drop(numerical_features)
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())
    ])

    transform_pipeline = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    full_pipeline = make_pipeline(transform_pipeline, model)

    # Train model
    full_pipeline.fit(data_processor.X_train, data_processor.y_train)

    # Evaluate model
    train_score = full_pipeline.score(data_processor.X_train, data_processor.y_train)

    val_score = full_pipeline.score(data_processor.X_val, data_processor.y_val)

    print(f"Train score: {train_score}")
    print(f"Validation score: {val_score}")

    # Predict
    y_pred = full_pipeline.predict(data_processor.X_test)
    print(y_pred)

    # Save predictions
    predictions = pd.DataFrame(data_processor.X_test)
    predictions['total_fire_size'] = y_pred
    predictions.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
