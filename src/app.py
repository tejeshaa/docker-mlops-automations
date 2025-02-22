from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from predict import predict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model API")

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    prediction: int

@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: PredictionInput):
    try:
        # Convert input features to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = predict(features)
        
        return PredictionOutput(prediction=int(prediction[0]))
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 