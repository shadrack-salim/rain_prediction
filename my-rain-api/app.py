from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load model
model = joblib.load('models/Rain_Classifier_With_Season.pkl')

def get_season(month):
    return 'wet' if month in [5, 6, 7, 8, 9, 10] else 'dry'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    input_datetime_str = data.get('datetime')
    temp = float(data.get('temp'))
    humidity = float(data.get('humidity'))

    dt = datetime.strptime(input_datetime_str, "%Y-%m-%d %H:%M")
    season = get_season(dt.month)
    season_wet = 1 if season == 'wet' else 0

    X_input = pd.DataFrame([{
        'hour': dt.hour,
        'dayofweek': dt.weekday(),
        'month': dt.month,
        'day': dt.day,
        'Temperature_C': temp,
        'Humidity_%': humidity,
        'season_wet': season_wet
    }])

    rain_prob = model.predict_proba(X_input)[0][1]
    rain_pred = model.predict(X_input)[0]

    return jsonify({
        'rain_probability': round(rain_prob, 4),
        'prediction': 'Rain' if rain_pred == 1 else 'No Rain'
    })

@app.route('/')
def home():
    return 'Rain Prediction API is running!'
