from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# Load model
model = joblib.load('models/Rain_Classifier_With_Season.pkl')

# Define FastAPI app
app = FastAPI(title="Rain Predictor")

# Input schema
class WeatherInput(BaseModel):
    datetime_str: str  # Format: YYYY-MM-DD HH:MM
    temperature_c: float
    humidity_percent: float

def get_season(month):
    return 'wet' if month in [5, 6, 7, 8, 9, 10] else 'dry'

@app.post("/predict_rain")
def predict_rain(data: WeatherInput):
    dt = datetime.strptime(data.datetime_str, "%Y-%m-%d %H:%M")
    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month
    day = dt.day
    season = get_season(month)
    season_wet = 1 if season == 'wet' else 0

    features = pd.DataFrame([{
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'day': day,
        'Temperature_C': data.temperature_c,
        'Humidity_%': data.humidity_percent,
        'season_wet': season_wet
    }])

    prob = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]

    return {
        "rain_probability": round(prob, 4),
        "will_rain": bool(prediction),
        "input": data.dict()
    }

