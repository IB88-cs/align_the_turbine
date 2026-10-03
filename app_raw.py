"""
FastAPI service simulating a live wind-turbine pitch prediction pipeline.

How it works:
- Loads the saved wind-forecast model + pitch model (and their scalers)
- Keeps a "cursor" position in a saved slice of historical data to simulate
  a live stream (each call to /predict/next advances the cursor by one step)
- Chains: recent wind window -> forecasted next wind speed -> pitch angle

Run with: uvicorn app:app --reload
Docs available at: http://127.0.0.1:8000/docs
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Turbine Pitch Prediction API")

# ---- Load models + scalers once at startup ----
wind_forecast_model = joblib.load("wind_forecast_model.pkl")
wind_forecast_scaler = joblib.load("wind_forecast_scaler.pkl")
pitch_model = joblib.load("pitch_model.pkl")
pitch_scaler = joblib.load("pitch_scaler.pkl")

# ---- Load the "live stream" data ----
stream_df = pd.read_csv("live_stream_data.csv")
stream_df["Date and time"] = pd.to_datetime(stream_df["Date and time"])

WINDOW_SIZE = 6  # must match what you trained the forecast model with
cursor = {"position": WINDOW_SIZE}  # mutable state tracking "now" in the simulated stream

PITCH_FEATURES = [
    "Wind speed (m/s)",
    "Wind speed, Standard deviation (m/s)",
    "Wind speed, Minimum (m/s)",
    "Wind speed, Maximum (m/s)",
    "Wind direction (°)",
    "Nacelle position (°)",
    "Rotor speed (RPM)",
]


class PredictionResponse(BaseModel):
    timestamp: str
    recent_wind_speeds: list[float]
    forecasted_wind_speed: float
    recommended_pitch_angle: float


@app.get("/")
def root():
    return {"status": "running", "docs": "/docs"}


@app.get("/predict/next", response_model=PredictionResponse)
def predict_next():
    """
    Advances the simulated stream by one step and returns:
    - the last 6 wind speed readings used as input
    - the forecasted next wind speed (10 min ahead)
    - the recommended pitch angle for that forecasted condition
    """
    pos = cursor["position"]
    if pos >= len(stream_df):
        raise HTTPException(status_code=404, detail="End of simulated stream reached")

    # 1. Grab the window of recent wind speeds
    window = stream_df["Wind speed (m/s)"].values[pos - WINDOW_SIZE:pos]
    window_scaled = wind_forecast_scaler.transform(window.reshape(1, -1))

    # 2. Forecast next wind speed
    forecasted_wind = wind_forecast_model.predict(window_scaled)[0]

    # 3. Build pitch-model input using forecasted wind speed + latest known other features
    latest_row = stream_df.iloc[pos - 1]
    pitch_input = pd.DataFrame([{
        "Wind speed (m/s)": forecasted_wind,
        "Wind speed, Standard deviation (m/s)": latest_row["Wind speed, Standard deviation (m/s)"],
        "Wind speed, Minimum (m/s)": latest_row["Wind speed, Minimum (m/s)"],
        "Wind speed, Maximum (m/s)": latest_row["Wind speed, Maximum (m/s)"],
        "Wind direction (°)": latest_row["Wind direction (°)"],
        "Nacelle position (°)": latest_row["Nacelle position (°)"],
        "Rotor speed (RPM)": latest_row["Rotor speed (RPM)"],
    }])[PITCH_FEATURES]

    pitch_input_scaled = pitch_scaler.transform(pitch_input)
    predicted_pitch = pitch_model.predict(pitch_input_scaled)[0]

    # 4. Advance the simulated clock
    cursor["position"] += 1

    return PredictionResponse(
        timestamp=str(latest_row["Date and time"]),
        recent_wind_speeds=window.tolist(),
        forecasted_wind_speed=float(forecasted_wind),
        recommended_pitch_angle=float(predicted_pitch),
    )


@app.post("/reset")
def reset_stream():
    """Resets the simulated stream back to the start."""
    cursor["position"] = WINDOW_SIZE
    return {"status": "reset", "position": cursor["position"]}