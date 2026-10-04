import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root():
    """Serves the user-facing dashboard."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Turbine Pitch Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg: #0f172a;
                --card-bg: #1e293b;
                --accent: #38bdf8;
                --text: #f8fafc;
                --text-dim: #94a3b8;
            }
            body {
                font-family: system-ui, -apple-system, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 2rem;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 2rem;
            }
            h1 { margin: 0; font-size: 1.5rem; font-weight: 600; }
            .subtitle { color: var(--text-dim); font-size: 0.9rem; }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            .card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .metric-label {
                color: var(--text-dim);
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                margin-top: 0.5rem;
                color: var(--accent);
            }
            .controls {
                display: flex;
                gap: 1rem;
                margin-bottom: 2rem;
            }
            button {
                background: #2563eb;
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }
            button:hover { background: #1d4ed8; }
            button.secondary { background: #334155; }
            button.secondary:hover { background: #475569; }
            button.active { background: #16a34a; }
            .chart-card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 1.5rem;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>Wind Turbine Live Monitoring</h1>
                <div class="subtitle" id="timestamp">Timestamp: Waiting for data...</div>
            </div>
            <a href="/docs" target="_blank" style="color: var(--accent); text-decoration: none;">API Docs</a>
        </div>

        <div class="controls">
            <button onclick="fetchNext()">Next Step</button>
            <button id="streamBtn" class="secondary" onclick="toggleStream()">Start Auto-Stream</button>
            <button class="secondary" onclick="resetStream()">Reset</button>
        </div>

        <div class="grid">
            <div class="card">
                <div class="metric-label">Forecasted Wind Speed (10 min ahead)</div>
                <div class="metric-value" id="forecastWind">-- m/s</div>
            </div>
            <div class="card">
                <div class="metric-label">Recommended Pitch Angle</div>
                <div class="metric-value" id="recommendedPitch">-- °</div>
            </div>
        </div>

        <div class="chart-card">
            <canvas id="windChart" height="100"></canvas>
        </div>

        <script>
            let streamInterval = null;
            const ctx = document.getElementById('windChart').getContext('2d');
            
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['-50m', '-40m', '-30m', '-20m', '-10m', 'Now', 'Forecast (+10m)'],
                    datasets: [{
                        label: 'Wind Speed (m/s)',
                        data: [0, 0, 0, 0, 0, 0, 0],
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { labels: { color: '#f8fafc' } } },
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
                    }
                }
            });

            async function fetchNext() {
                try {
                    const res = await fetch('/predict/next');
                    if (!res.ok) {
                        if (res.status === 404) alert("End of data stream reached.");
                        return;
                    }
                    const data = await res.json();
                    
                    document.getElementById('timestamp').innerText = 'Timestamp: ' + data.timestamp;
                    document.getElementById('forecastWind').innerText = data.forecasted_wind_speed.toFixed(2) + ' m/s';
                    document.getElementById('recommendedPitch').innerText = data.recommended_pitch_angle.toFixed(2) + '°';

                    // Update chart with recent wind speeds + forecasted value
                    const chartData = [...data.recent_wind_speeds, data.forecasted_wind_speed];
                    chart.data.datasets[0].data = chartData;
                    chart.update();
                } catch (e) {
                    console.error("Error fetching prediction:", e);
                }
            }

            function toggleStream() {
                const btn = document.getElementById('streamBtn');
                if (streamInterval) {
                    clearInterval(streamInterval);
                    streamInterval = null;
                    btn.innerText = "Start Auto-Stream";
                    btn.classList.remove('active');
                    btn.classList.add('secondary');
                } else {
                    fetchNext();
                    streamInterval = setInterval(fetchNext, 2000);
                    btn.innerText = "Pause Auto-Stream";
                    btn.classList.remove('secondary');
                    btn.classList.add('active');
                }
            }

            async function resetStream() {
                await fetch('/reset', { method: 'POST' });
                if (streamInterval) toggleStream();
                fetchNext();
            }

            // Initial Load
            fetchNext();
        </script>
    </body>
    </html>
    """


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
