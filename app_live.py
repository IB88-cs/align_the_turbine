"""
FastAPI service using LIVE wind data (Open-Meteo) to forecast wind speed
and recommend a turbine pitch angle.

How it works:
- Polls Open-Meteo every 10 minutes in the background for current wind speed,
  direction, and gust at a fixed location, building up a rolling buffer
- /predict/next uses the buffer to forecast the next wind speed, then feeds
  that into the simplified pitch model (wind speed, std-dev proxy, direction)

Run with: uvicorn app_live:app --reload
Docs available at: http://127.0.0.1:8000/docs

NOTE: the buffer needs WINDOW_SIZE readings before predictions are possible.
With a 10-minute poll interval, that's roughly 1 hour of uptime before the
first prediction can be made. For testing, POLL_INTERVAL_MINUTES is set low
(see below) — turn it back up to 10 for anything resembling realistic use.
"""

import joblib
import numpy as np
import requests
from collections import deque
from datetime import datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Turbine Pitch Prediction API (Live)")

# ---- Config ----
LATITUDE = -37.5965   # Kelmarsh, UK — change to your site's coordinates
LONGITUDE = 143.4996
WINDOW_SIZE = 6
POLL_INTERVAL_MINUTES = 1  # set lower (e.g. 1) temporarily while testing

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_PARAMS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "current": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
    "wind_speed_unit": "ms",
}

# ---- Load models + scalers ----
wind_forecast_model = joblib.load("wind_forecast_model.pkl")
wind_forecast_scaler = joblib.load("wind_forecast_scaler.pkl")
pitch_model = joblib.load("simplified_pitch_model.pkl")
pitch_scaler = joblib.load("simplified_pitch_scaler.pkl")

# ---- Rolling buffer of recent readings ----
wind_buffer = deque(maxlen=WINDOW_SIZE)
latest_reading = {"wind_speed": None, "wind_direction": None, "wind_gust": None, "timestamp": None}


def fetch_current_wind():
    """Calls Open-Meteo and returns the current wind reading."""
    response = requests.get(OPEN_METEO_URL, params=OPEN_METEO_PARAMS, timeout=10)
    response.raise_for_status()
    current = response.json()["current"]
    return {
        "wind_speed": current["wind_speed_10m"],
        "wind_direction": current["wind_direction_10m"],
        "wind_gust": current["wind_gusts_10m"],
        "timestamp": current["time"],
    }


def poll_wind():
    """Background job: fetch live wind data and append to the buffer."""
    try:
        reading = fetch_current_wind()
        wind_buffer.append(reading["wind_speed"])
        latest_reading.update(reading)
        print(f"[{datetime.now(timezone.utc).isoformat()}] Polled wind: {reading}")
    except Exception as e:
        print(f"Polling failed: {e}")


# ---- Scheduler: poll automatically in the background ----
scheduler = BackgroundScheduler()
scheduler.add_job(poll_wind, "interval", minutes=POLL_INTERVAL_MINUTES, next_run_time=datetime.now())
scheduler.start()


class PredictionResponse(BaseModel):
    timestamp: str
    recent_wind_speeds: list[float]
    forecasted_wind_speed: float
    recommended_pitch_angle: float


@app.get("/")
def root():
    return {
        "status": "running",
        "buffer_size": len(wind_buffer),
        "buffer_needed": WINDOW_SIZE,
        "docs": "/docs",
    }


@app.get("/predict/next", response_model=PredictionResponse)
def predict_next():
    """
    Uses the current rolling buffer of live wind readings to forecast the
    next wind speed, then recommends a pitch angle for that condition.
    """
    if len(wind_buffer) < WINDOW_SIZE:
        raise HTTPException(
            status_code=425,  # 425 Too Early
            detail=f"Not enough live data yet: {len(wind_buffer)}/{WINDOW_SIZE} readings collected. "
                   f"Wait for more polling cycles (every {POLL_INTERVAL_MINUTES} min)."
        )

    # 1. Forecast next wind speed from the buffer
    window = np.array(wind_buffer)
    window_scaled = wind_forecast_scaler.transform(window.reshape(1, -1))
    forecasted_wind = wind_forecast_model.predict(window_scaled)[0]

    # 2. Build simplified pitch-model input
    #    Using gust as a stand-in for "standard deviation" (both represent
    #    short-term wind variability — a reasonable proxy, not identical)
    gust = latest_reading["wind_gust"] or 0.0
    direction = latest_reading["wind_direction"] or 0.0
    wind_std_proxy = max(gust - forecasted_wind, 0)

    pitch_input = np.array([[forecasted_wind, wind_std_proxy, direction]])
    pitch_input_scaled = pitch_scaler.transform(pitch_input)
    predicted_pitch = pitch_model.predict(pitch_input_scaled)[0]

    return PredictionResponse(
        timestamp=latest_reading["timestamp"],
        recent_wind_speeds=list(wind_buffer),
        forecasted_wind_speed=float(forecasted_wind),
        recommended_pitch_angle=float(predicted_pitch),
    )


@app.post("/poll-now")
def poll_now():
    """Manually trigger a poll immediately, instead of waiting for the schedule — useful for testing."""
    poll_wind()
    return {"status": "polled", "buffer_size": len(wind_buffer), "latest": latest_reading}