from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# === Load the trained model ===
model = joblib.load('models/Rain_Classifier_With_Season.pkl')

# === Create FastAPI app ===
app = FastAPI(title="Rain Predictor API")

# === Pydantic model for input ===
class WeatherInput(BaseModel):
    datetime_str: str  # Format: YYYY-MM-DD HH:MM
    temperature_c: float
    humidity_percent: float

# === Season function ===
def get_season(month):
    return 'wet' if month in [5, 6, 7, 8, 9, 10] else 'dry'

# === Root route to verify API is live ===
@app.get("/")
def read_root():
    return {"message": "Rain Predictor API is live ✅"}

# === Prediction endpoint ===
@app.post("/predict_rain")
def predict_rain(data: WeatherInput):
    try:
        dt = datetime.strptime(data.datetime_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return {"error": "datetime_str must be in 'YYYY-MM-DD HH:MM' format"}

    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month
    day = dt.day
    season = get_season(month)
    season_wet = 1 if season == 'wet' else 0

    # Prepare features
    features = pd.DataFrame([{
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'day': day,
        'Temperature_C': data.temperature_c,
        'Humidity_%': data.humidity_percent,
        'season_wet': season_wet
    }])

    # Predict
    prob = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]

    return {
        "rain_probability": round(prob, 4),
        "will_rain": bool(prediction),
        "input": data.dict()
    }
