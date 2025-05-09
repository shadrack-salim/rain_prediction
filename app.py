from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import numpy as np
from typing import List
import os

# Create FastAPI app
app = FastAPI(title="Rainfall Prediction API")

# Load models
model_path = os.path.join(os.path.dirname(__file__), "models")
classifier = load(os.path.join(model_path, "rainfall_classifier.joblib"))
regressor = load(os.path.join(model_path, "rainfall_regression.joblib"))

# Define input schema
class WeatherInput(BaseModel):
    day_of_year: int
    month: int
    weekday: int
    year: int

class BatchWeatherInput(BaseModel):
    data: List[WeatherInput]

# Define prediction endpoint
@app.post("/predict/")
async def predict_rainfall(weather_data: WeatherInput):
    # Convert input to format expected by the model
    features = [[
        weather_data.day_of_year,
        weather_data.month,
        weather_data.weekday,
        weather_data.year
    ]]
    
    # Make predictions
    is_wet_day = bool(classifier.predict(features)[0])
    
    if is_wet_day:
        # Predict amount (remember to transform back from log)
        rain_amount_log = regressor.predict(features)[0]
        rain_amount = float(np.expm1(rain_amount_log))
        return {
            "will_rain": True,
            "precipitation_mm": round(rain_amount, 2)
        }
    else:
        return {
            "will_rain": False,
            "precipitation_mm": 0.0
        }

# Batch prediction endpoint
@app.post("/predict-batch/")
async def predict_batch(batch_data: BatchWeatherInput):
    # Extract features
    features = [[
        item.day_of_year,
        item.month,
        item.weekday,
        item.year
    ] for item in batch_data.data]
    
    # Make predictions
    is_wet_day = classifier.predict(features)
    results = []
    
    for i, wet_day in enumerate(is_wet_day):
        if wet_day:
            rain_amount_log = regressor.predict([features[i]])[0]
            rain_amount = float(np.expm1(rain_amount_log))
            results.append({
                "will_rain": True,
                "precipitation_mm": round(rain_amount, 2)
            })
        else:
            results.append({
                "will_rain": False,
                "precipitation_mm": 0.0
            })
    
    return {"predictions": results}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Rainfall Prediction API",
        "usage": "POST /predict/ with day_of_year, month, weekday, and year"
    }
