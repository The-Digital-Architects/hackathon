import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml_pipeline.base import Pipeline
from ml_pipeline.models import RandomForestWrapper, AdaBoostWrapper

def main():
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train Random Forest pipeline
    rf_model = RandomForestWrapper(n_estimators=100, random_state=42)
    rf_pipeline = Pipeline(rf_model)
    predictions, _ = rf_pipeline.train(X_train, y_train)
    
    # Evaluate
    test_predictions = rf_pipeline.predict(X_test)
    accuracy = np.mean(test_predictions == y_test)
    print(f"Random Forest Test Accuracy: {accuracy:.4f}")
    
    # Try AdaBoost pipeline
    ada_model = AdaBoostWrapper(n_estimators=50, random_state=42)
    ada_pipeline = Pipeline(ada_model)
    ada_pipeline.train(X_train, y_train)
    ada_predictions = ada_pipeline.predict(X_test)
    ada_accuracy = np.mean(ada_predictions == y_test)
    print(f"AdaBoost Test Accuracy: {ada_accuracy:.4f}")

if __name__ == "__main__":
    main()
