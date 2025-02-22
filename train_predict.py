from src.data_processor import WildfireData
from src.models import WildfirePredictor
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

def train_and_evaluate(data_processor, model_config):
    predictor = WildfirePredictor(model_config)
    
    # Create pipeline
    numerical_features = data_processor.X_train.select_dtypes(include=[np.number]).columns
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

    full_pipeline = make_pipeline(transform_pipeline, predictor.model)

    # Train
    full_pipeline.fit(data_processor.X_train, data_processor.y_train)
    
    # Evaluate using predictions from full pipeline
    y_train_pred = full_pipeline.predict(data_processor.X_train)
    y_val_pred = full_pipeline.predict(data_processor.X_val)
    
    # Calculate scores using raw predictions and true values
    train_score = calculate_custom_score(y_train_pred, data_processor.y_train)
    val_score = calculate_custom_score(y_val_pred, data_processor.y_val)
    
    return full_pipeline, train_score, val_score

def calculate_custom_score(y_pred, y_true):
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    y_pred = np.maximum(y_pred, epsilon)
    y_true = np.maximum(y_true, epsilon)
    
    log_array = np.abs(np.log(y_pred/y_true))
    constrained_log_array = np.minimum(log_array, 10)
    sum_logs = np.sum(constrained_log_array)
    
    return -(1/len(y_true) * sum_logs)  # Negative because we want to maximize

def create_stacking_model():
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42))
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5
    )

def main():
    # Enhanced model configurations
    models_config = {
        'random_forest': {
            'model_type': 'rf',
            'model_params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42
            }
        },
        'gradient_boost': {
            'model_type': 'gb',
            'model_params': {
                'n_estimators': 150,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.8,
                'random_state': 42
            }
        },
        'xgboost': {
            'model_type': 'xgb',
            'model_params': {
                'n_estimators': 150,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        },
        'lightgbm': {
            'model_type': 'lgbm',
            'model_params': {
                'n_estimators': 150,
                'learning_rate': 0.05,
                'num_leaves': 32,
                'subsample': 0.8,
                'random_state': 42
            }
        },
        'catboost': {
            'model_type': 'catboost',
            'model_params': {
                'iterations': 150,
                'learning_rate': 0.05,
                'depth': 8,
                'random_state': 42,
                'verbose': False
            }
        },
        'extra_trees': {
            'model_type': 'ext',
            'model_params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42
            }
        },
        'neural_net': {
            'model_type': 'mlp',
            'model_params': {
                'hidden_layer_sizes': (100, 50, 25),
                'activation': 'relu',
                'max_iter': 500,
                'random_state': 42
            }
        },
        'robust_regression': {
            'model_type': 'huber',
            'model_params': {
                'epsilon': 1.35,
                'max_iter': 200,
                'alpha': 0.0001
            }
        }
    }

    # Initialize data processor
    data_processor = WildfireData(
        fire_data_path='data/wildfire_sizes_before_2010.csv',
        state_data_path='data/merged_state_data.csv',
        weather_data_path='data/weather_monthly_state_aggregates.csv',
        coordinates_path='data/state_coordinates.csv'
    )
    
    # Prepare data
    data_processor.prepare_data()

    X_test_original = data_processor.X_test.copy()

    data_processor.filter_features(['PRCP', 'EVAP', 'TMIN', 'TMAX', 'mean_elevation', 
                                  'Land Area (sq mi)', 'Water Area (sq mi)', 
                                  'Percentage of Federal Land', 'Urbanization Rate (%)', 
                                  'latitude', 'longitude'])

    # Try all models
    results = {}
    best_model = None
    best_score = -np.inf
    
    for model_name, config in models_config.items():
        print(f"\nTraining {model_name}...")
        pipeline, train_score, val_score = train_and_evaluate(data_processor, config)
        
        results[model_name] = {
            'train_score': train_score,
            'val_score': val_score
        }
        
        if val_score > best_score:
            best_score = val_score
            best_model = pipeline

    # Print results with better formatting
    print("\nModel Comparison (Custom Evaluation Metric):")
    for model_name, scores in results.items():
        print(f"\n{model_name}:")
        print(f"Train error: {-scores['train_score']:.4f}")
        print(f"Validation error: {-scores['val_score']:.4f}")
        print(f"(Lower error is better)")

    # Use best model for predictions
    print(f"\nUsing best model for predictions...")
    y_pred = best_model.predict(data_processor.X_test)
    
    # Save predictions
    predictions = pd.DataFrame(X_test_original)
    predictions['total_fire_size'] = y_pred
    predictions.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
