import argparse
import yaml
from pathlib import Path
import joblib
from sklearn.pipeline import make_pipeline

from src.data import DataHandler
from src.models import RandomForest, AdaBoost, NeuralNetwork
from src.transformers import StandardScaler


# Registry of available transformers
TRANSFORMER_REGISTRY = {
    'standard_scaler': StandardScaler,
    # Add more transformers here as needed
}

# Registry of available models
MODEL_REGISTRY = {
    'random_forest': RandomForest,
    'adaboost': AdaBoost,
    'neural_network': NeuralNetwork,
    # Add more models here as needed
}


def create_pipeline(config: dict):
    """Create a scikit-learn pipeline from the configuration."""
    # Build a list of transformer instances.
    transformers = []
    for t_config in config['pipeline']['transformers']:
        transformer_class = TRANSFORMER_REGISTRY.get(t_config['type'])
        if not transformer_class:
            raise ValueError(f"Unknown transformer type: {t_config['type']}")
        # Pass parameters using ** if provided; assume params is a dict.
        transformer_obj = transformer_class(**t_config.get('params', {}))
        transformers.append(transformer_obj._create_transformer())
    
    # Initialize model based on configuration
    model_type = config['model']['type']
    model_class = MODEL_REGISTRY.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")
    model_obj = model_class()
    model = model_obj._create_model()
    
    # Create and return pipeline
    pipeline = make_pipeline(*transformers, model)
    return pipeline


def main(config_path: str):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading data...")
    data_handler = DataHandler(config['data'])
    X, y = data_handler.load_data(config['data']['train_path'])
    X_train, X_test, y_train, y_test = data_handler.split_data(X, y)
    
    # Create and train pipeline
    pipeline = create_pipeline(config)
    print("Training pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    test_predictions = pipeline.predict(X_test)
    accuracy = (test_predictions == y_test).mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save pipeline using joblib
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = output_dir / f"{config['model']['type']}_pipeline.joblib"
    print(f"Saving pipeline to {pipeline_path}")
    joblib.dump(pipeline, pipeline_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)