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

    # Parse input
    date_str = data.get('date_time')
    temp = float(data.get('temperature'))
    humidity = float(data.get('humidity'))

    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month
    day = dt.day
    season = get_season(month)
    season_wet = 1 if season == 'wet' else 0

    # Create feature DataFrame
    X_input = pd.DataFrame([{
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'day': day,
        'Temperature_C': temp,
        'Humidity_%': humidity,
        'season_wet': season_wet
    }])

    # Predict
    prob = model.predict_proba(X_input)[0][1]
    pred = model.predict(X_input)[0]

    return jsonify({
        'prediction': int(pred),
        'probability': round(prob, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
