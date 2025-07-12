#!/usr/bin/env python3

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """Load processed heart rate data and prepare for time series modeling"""
    df = pd.read_parquet(filepath)
    
    # Set datetime as index and sort
    df = df.set_index('datetime').sort_index()
    
    # Resample to 5-minute intervals (reduce noise, manageable for daily seasonality)
    ts = df['heart_rate'].resample('5T').mean()
    
    # Fill any missing values with forward fill
    ts = ts.fillna(method='ffill')
    
    return ts

def fit_holt_winters_model(ts):
    """Fit Holt-Winters model with daily seasonality"""
    # Daily seasonality: 24 hours * 12 periods per hour (5-min intervals) = 288
    seasonal_periods = 288
    
    # Fit ExponentialSmoothing model
    model = ExponentialSmoothing(
        ts,
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods,
        damped_trend=True
    )
    
    fitted_model = model.fit(optimized=True)
    
    # Print model summary
    print("Model fitted successfully!")
    print(f"Training data: {len(ts)} observations")
    print(f"Date range: {ts.index.min()} to {ts.index.max()}")
    print(f"Seasonal periods: {seasonal_periods}")
    print(f"AIC: {fitted_model.aic:.2f}")
    
    return fitted_model

def main():
    """Main training pipeline"""
    print("Loading and preparing data...")
    ts = load_and_prepare_data('processed_heart_rate_data.parquet')
    
    print("Fitting Holt-Winters model...")
    model = fit_holt_winters_model(ts)
    
    print("Saving model...")
    joblib.dump(model, 'hw_model.pkl')
    
    print("Training completed! Model saved to hw_model.pkl")

if __name__ == "__main__":
    main() 