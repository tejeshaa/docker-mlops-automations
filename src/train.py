import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Generate sample data for demonstration"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
    return pd.DataFrame(X), pd.Series(y)

def train_model():
    """Train the model and save it"""
    logger.info("Loading data...")
    X, y = load_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    logger.info("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Train accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # Save model
    logger.info("Saving model...")
    joblib.dump(model, 'model.joblib')
    
    return test_score

if __name__ == "__main__":
    train_model() 