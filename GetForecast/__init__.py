import azure.functions as func
import pandas as pd
import joblib
import json
import logging
from datetime import datetime, timezone
import os

# Global variable to store the model (loaded on cold start)
model = None

def load_model():
    """Load the trained Holt-Winters model on cold start"""
    global model
    if model is None:
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'hw_model.pkl')
            model = joblib.load(model_path)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    return model

def get_steps_ahead(target_time, model_end_time):
    """Calculate steps ahead from model's training end time"""
    # Model uses 5-minute intervals, so 12 steps per hour
    time_diff = target_time - model_end_time
    steps_ahead = int(time_diff.total_seconds() / 300)  # 300 seconds = 5 minutes
    return max(1, steps_ahead)  # At least 1 step ahead

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main Azure Function entry point"""
    try:
        # Load model on cold start
        hw_model = load_model()
        
        # Get timezone parameter
        timezone_param = req.params.get('timezone', 'UTC')
        
        # Get current timestamp
        if timezone_param.upper() == 'UTC':
            current_time = datetime.now(timezone.utc)
        else:
            try:
                import pytz
                tz = pytz.timezone(timezone_param)
                current_time = datetime.now(tz)
            except:
                # Fallback to UTC if timezone is invalid
                current_time = datetime.now(timezone.utc)
                timezone_param = 'UTC'
        
        # Get model's training end time
        model_end_time = hw_model.fittedvalues.index[-1]
        
        # Calculate steps ahead
        steps_ahead = get_steps_ahead(current_time, model_end_time)
        
        # Generate forecast
        forecast = hw_model.forecast(steps=steps_ahead)
        predicted_hr = float(forecast.iloc[-1])  # Get the last forecasted value
        
        # Prepare response
        response_data = {
            "timestamp": current_time.isoformat(),
            "predictedHeartRate": round(predicted_hr, 1),
            "timezone": timezone_param,
            "stepsAhead": steps_ahead
        }
        
        return func.HttpResponse(
            json.dumps(response_data),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        error_response = {
            "error": str(e),
            "message": "Failed to generate heart rate forecast"
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            mimetype="application/json",
            status_code=500
        ) 