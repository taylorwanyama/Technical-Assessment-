import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import uvicorn


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Response Prediction API",
    description="API for predicting clinical response based on patient data",
    version="1.0.0"
)

# Load The best performing model(Logistic regression)
try:
    lr_model = joblib.load('trained_models/logistic_regression_model.joblib')
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    lr_model = None

# Define input schema
class PatientData(BaseModel):
    """Schema for patient input data."""
    DOSE: float = Field(..., description="Drug dose in mg", ge=0)
    AGE: float = Field(..., description="Patient age in years", ge=18, le=100)
    WT: float = Field(..., description="Patient weight in kg", ge=30, le=200)
    SEX: str = Field(..., description="Patient sex (M/F)")
    CMAX: float = Field(..., description="Maximum concentration in ng/mL", ge=0)
    AUC: float = Field(..., description="Area under curve in ng·h/mL", ge=0)
    
  
class PredictionResponse(BaseModel):
    """Schema for prediction output."""
    patient_data: Dict
    logistic_regression: Dict

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Clinical Response Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Get prediction for patient data",
            "/model-info": "GET - Get model information",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    models_loaded = lr_model is not None
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    }

@app.get("/model-info")
def get_model_info():
    """Get information about loaded models."""
    if lr_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Load performance metrics if available
    try:
        performance = pd.read_csv('results/tables/model_performance.csv')
        performance_dict = performance.to_dict('records')
    except:
        performance_dict = []
    
    return {
        "model": ["Logistic Regression"],
        "features": ["DOSE", "AGE", "WT", "SEX", "CMAX", "AUC"],
        "performance_metrics": performance_dict,
        "description": "Binary classification models for clinical response prediction"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    """
    Predict clinical response for a patient.
    
    Parameters:
    -----------
    patient : PatientData
        Patient data including demographics and PK parameters
        
    Returns:
    --------
    PredictionResponse
        Predictions from the model with probabilities
    """
    if lr_model is None :
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate SEX input
    if patient.SEX not in ['M', 'F']:
        raise HTTPException(
            status_code=400,
            detail="SEX must be 'M' or 'F'"
        )
    
    # Prepare features
    sex_encoded = 1 if patient.SEX == 'M' else 0
    features = np.array([[
        patient.DOSE,
        patient.AGE,
        patient.WT,
        sex_encoded,
        patient.CMAX,
        patient.AUC
    ]])
    
    # Get predictions from the model
    lr_pred = lr_model.predict(features)[0]
    lr_proba = lr_model.predict_proba(features)[0]
    
   # Create response
    response = PredictionResponse(
        patient_data=patient.dict(),
        logistic_regression={
            "prediction": int(lr_pred),
            "prediction_label": "Response" if lr_pred == 1 else "No Response",
            "probability_no_response": round(float(lr_proba[0]), 4),
            "probability_response": round(float(lr_proba[1]), 4)
        },
    )    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

