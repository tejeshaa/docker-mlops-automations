import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path='model.joblib'):
    """Load the trained model"""
    return joblib.load(model_path)

def predict(data):
    """Make predictions using the trained model"""
    model = load_model()
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    # Sample prediction
    sample_data = np.random.rand(1, 20)
    result = predict(sample_data)
    logger.info(f"Prediction for sample data: {result}") 