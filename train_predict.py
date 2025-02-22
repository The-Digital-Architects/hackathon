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
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureNamePreserver(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = None
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            return X.to_numpy()
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names

def train_and_evaluate(data_processor, model_config):
    predictor = WildfirePredictor(model_config)
    
    # Create pipeline
    numerical_features = data_processor.X_train.select_dtypes(include=[np.number]).columns
    categorical_features = data_processor.X_train.columns.drop(numerical_features)
    
    num_pipeline = Pipeline([
        ('preserves_names', FeatureNamePreserver()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('preserves_names', FeatureNamePreserver()),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False))
    ])

    transform_pipeline = ColumnTransformer(
        [
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        verbose_feature_names_out=False
    )

    full_pipeline = Pipeline([
        ('preprocessor', transform_pipeline),
        ('model', predictor.model)
    ])

    # Train and evaluate
    full_pipeline.fit(data_processor.X_train, data_processor.y_train)
    
    # Get feature names after transformation
    feature_names = transform_pipeline.get_feature_names_out()
    
    # Transform data and convert to numpy arrays
    X_train_transformed = transform_pipeline.transform(data_processor.X_train)
    X_val_transformed = transform_pipeline.transform(data_processor.X_val)
    
    # Train score and validation score using numpy arrays
    train_score = calculate_custom_score(
        predictor.model.predict(X_train_transformed), 
        data_processor.y_train
    )
    val_score = calculate_custom_score(
        predictor.model.predict(X_val_transformed), 
        data_processor.y_val
    )
    
    return full_pipeline, train_score, val_score, feature_names

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
                'max_iter': 1000,  # Increased from 500
                'early_stopping': True,  # Added early stopping
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
    feature_names = None
    
    for model_name, config in models_config.items():
        print(f"\nTraining {model_name}...")
        pipeline, train_score, val_score, feat_names = train_and_evaluate(data_processor, config)
        feature_names = feat_names  # Save feature names for later use
        
        results[model_name] = {
            'train_score': train_score,
            'val_score': val_score,
            'pipeline': pipeline
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

    # Use best model for predictions with numpy arrays
    print(f"\nUsing best model for predictions...")
    X_test_transformed = best_model.named_steps['preprocessor'].transform(data_processor.X_test)
    y_pred = best_model.named_steps['model'].predict(X_test_transformed)
    
    # Save predictions
    predictions = pd.DataFrame(X_test_original)
    predictions['total_fire_size'] = y_pred
    predictions.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
