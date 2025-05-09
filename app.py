from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model - with better error handling
try:
    model_path = os.getenv('MODEL_PATH', 'model/Rain_Classifier_With_Season.pkl')
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def get_season(month):
    return 'wet' if month in [5, 6, 7, 8, 9, 10] else 'dry'

@app.route('/', methods=['GET'])
def home():
    """Homepage route"""
    return jsonify({
        "status": "online",
        "message": "Rain Prediction API is running. Use /predict endpoint for predictions.",
        "sample_request": {
            "date_time": "2023-05-09 14:30",
            "temperature": 25.5,
            "humidity": 80.0
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the service is running"""
    if model is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    return jsonify({"status": "unhealthy", "model_loaded": False}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to get rain predictions based on weather data"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        required_fields = ['date_time', 'temperature', 'humidity']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Parse input
        try:
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
            
            # Log prediction attempt
            logger.info(f"Prediction attempt: date={date_str}, temp={temp}, humidity={humidity}")
            
            # Predict
            prob = model.predict_proba(X_input)[0][1]
            pred = model.predict(X_input)[0]
            
            return jsonify({
                'prediction': int(pred),
                'probability': round(prob, 4),
                'input_received': {
                    'date_time': date_str,
                    'temperature': temp,
                    'humidity': humidity
                }
            })
            
        except ValueError as ve:
            logger.error(f"Value error in prediction: {ve}")
            return jsonify({"error": f"Invalid input data: {str(ve)}"}), 400
            
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
