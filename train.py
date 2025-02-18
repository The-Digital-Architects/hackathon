import argparse
import yaml
import logging
from pathlib import Path

from src.data import DataHandler
from src.models import RandomForest, AdaBoost

def setup_logging(log_path: Path) -> None:
    """Set up basic logging configuration."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler()
        ]
    )

def main(config_path: str):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_path = Path(config['logging']['save_path']) / 'train.log'
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    
    # Initialize data handler
    logger.info("Initializing data handler...")
    data_handler = DataHandler(config['data'])
    
    # Load and split data
    logger.info("Loading and preprocessing data...")
    X, y = data_handler.load_data(config['data']['train_path'])
    X_train, X_test, y_train, y_test = data_handler.split_data(X, y)
    
    # Initialize model
    model_type = config['model']['type']
    logger.info(f"Initializing {model_type} model...")
    
    model_classes = {
        'random_forest': RandomForest,
        'adaboost': AdaBoost
    }
    
    model_class = model_classes.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_class(config['model'])
    
    # Train model
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    test_predictions = model.predict(X_test)
    accuracy = (test_predictions == y_test).mean()
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_type}_model.joblib"
    logger.info(f"Saving model to {model_path}")
    model.save(str(model_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)