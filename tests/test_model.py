import pytest
import numpy as np
from src.train import train_model
from src.predict import predict

def test_model_training():
    """Test if model training works and achieves minimum accuracy"""
    test_score = train_model()
    assert test_score > 0.7, "Model accuracy is too low"

def test_prediction():
    """Test if prediction works"""
    sample_data = np.random.rand(1, 20)
    prediction = predict(sample_data)
    assert prediction.shape == (1,), "Prediction shape is incorrect"
    assert prediction[0] in [0, 1], "Invalid prediction value" 