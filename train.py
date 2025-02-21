import argparse
import yaml
from pathlib import Path

from src.data import DataHandler
from src.pipeline import Pipeline, TRANSFORMER_REGISTRY, MODEL_REGISTRY

def create_pipeline(config: dict):
    """Create a pipeline from config."""
    # Initialize transformers
    transformers = []
    for t_config in config['pipeline']['transformers']:
        transformer_class = TRANSFORMER_REGISTRY[t_config['type']]
        transformers.append(transformer_class(t_config.get('params', {})))
    
    # Initialize model
    model_type = config['model']['type']
    model_class = MODEL_REGISTRY.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_class(config['model'])
    
    return Pipeline(transformers=transformers, model=model)

def main(config_path: str):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize data handler
    print("Initializing data handler...")
    data_handler = DataHandler(config['data'])
    
    # Load and split data
    print("Loading and preprocessing data...")
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
    
    # Save pipeline
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = config['model']['type']
    pipeline_path = output_dir / f"{model_type}_pipeline.joblib"
    print(f"Saving pipeline to {pipeline_path}")
    pipeline.save(str(pipeline_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)