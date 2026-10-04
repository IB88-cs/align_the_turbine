import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

app = FastAPI(title="Turbine Pitch Prediction API")

# ---- Setup Assets Directory ----
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

PITCH_FEATURES = [
    "Wind speed (m/s)",
    "Wind speed, Standard deviation (m/s)",
    "Wind speed, Minimum (m/s)",
    "Wind speed, Maximum (m/s)",
    "Wind direction (°)",
    "Nacelle position (°)",
    "Rotor speed (RPM)",
]

# ---- Load models + scalers once at startup ----
try:
    wind_forecast_model = joblib.load("wind_forecast_model.pkl")
    wind_forecast_scaler = joblib.load("wind_forecast_scaler.pkl")
    pitch_model = joblib.load("pitch_model.pkl")
    pitch_scaler = joblib.load("pitch_scaler.pkl")
except FileNotFoundError:
    print("Warning: Model or Scaler files not found. The app will run, but /predict/next requires them.")

# ---- Load the "live stream" data ----
try:
    stream_df = pd.read_csv("live_stream_data.csv")
    stream_df["Date and time"] = pd.to_datetime(stream_df["Date and time"])
except FileNotFoundError:
    print("Warning: live_stream_data.csv not found.")
    stream_df = pd.DataFrame(columns=["Date and time", "Wind speed (m/s)"] + PITCH_FEATURES)

WINDOW_SIZE = 6  
cursor = {"position": WINDOW_SIZE}  


class PredictionResponse(BaseModel):
    timestamp: str
    recent_wind_speeds: list[float]
    forecasted_wind_speed: float
    recommended_pitch_angle: float


@app.get("/", response_class=HTMLResponse)
def root():
    """Serves the full-screen user-facing dashboard."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Turbine Pitch Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
        <style>
            :root {
                --bg: #0f172a;
                --card-bg: #1e293b;
                --accent: #38bdf8;
                --text: #f8fafc;
                --text-dim: #94a3b8;
            }
            * {
                box-sizing: border-box;
            }
            html, body {
                height: 100vh;
                margin: 0;
                padding: 0;
                overflow: hidden; /* Fill full screen without body scrollbars */
                font-family: system-ui, -apple-system, sans-serif;
                background-color: var(--bg);
                color: var(--text);
            }
            
            /* Root Layout Wrapper */
            .app-wrapper {
                display: flex;
                flex-direction: column;
                height: 100vh;
                padding: 1rem 1.5rem;
                gap: 1rem;
            }

            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            h1 { margin: 0; font-size: 1.4rem; font-weight: 600; }
            .subtitle { color: var(--text-dim); font-size: 0.85rem; }
            
            .controls {
                display: flex;
                gap: 0.75rem;
            }
            button {
                background: #2563eb;
                color: white;
                border: none;
                padding: 0.6rem 1.25rem;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }
            button:hover { background: #1d4ed8; }
            button.secondary { background: #334155; }
            button.secondary:hover { background: #475569; }
            button.active { background: #16a34a; }

            /* Full Height Grid */
            .main-content-grid {
                display: grid;
                grid-template-columns: 1.9fr 1fr;
                gap: 1rem;
                flex: 1; /* Stretch to fill all available vertical space */
                min-height: 0; /* Prevents overflow inside flex container */
            }

            .left-column {
                display: flex;
                flex-direction: column;
                gap: 1rem;
                height: 100%;
                min-height: 0;
            }

            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }
            .card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 0.9rem 1.25rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .metric-label {
                color: var(--text-dim);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                margin-top: 0.2rem;
                color: var(--accent);
            }

            .metric-caption {
                color: var(--text-dim);
                font-size: 0.85rem;
                line-height: 1.35;
                margin-top: 0.35rem;
                min-height: 2.7em;
            }

            .chart-card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 1.25rem;
                flex: 1;
                min-height: 0;
                display: flex;
                flex-direction: column;
            }
            .chart-container {
                position: relative;
                flex: 1;
                width: 100%;
                height: 100%;
                min-height: 0;
            }

            .visualization-card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 1.25rem;
                display: flex;
                flex-direction: column;
                height: 100%;
                min-height: 0;
            }
            .visualization-label {
                color: var(--text-dim);
                font-size: 0.8rem;
                margin-bottom: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            #turbineCanvasContainer {
                width: 100%;
                flex: 1;
                min-height: 0;
                border-radius: 8px;
                overflow: hidden;
                background: var(--card-bg);
                position: relative;
            }
            .pitch-badge {
                position: absolute;
                top: 12px;
                right: 12px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(4px);
                padding: 6px 12px;
                border-radius: 6px;
                border: 1px solid #334155;
                font-size: 0.85rem;
                color: var(--accent);
                font-weight: 600;
                z-index: 10;
            }
        </style>
    </head>
    <body>
        <div class="app-wrapper">
            <div class="header">
                <div>
                    <h1>Wind Turbine Live Monitoring</h1>
                    <div class="subtitle" id="timestamp">Simulated live stream (replay of Kelmarsh turbine data) &middot; Waiting for data...</div>
                </div>
                <div class="controls">
                    <button onclick="fetchNext()">Next Step</button>
                    <button id="streamBtn" class="secondary" onclick="toggleStream()">Start Auto-Stream</button>
                    <button class="secondary" onclick="resetStream()">Reset</button>
                    <a href="/docs" target="_blank" style="color: var(--accent); text-decoration: none; align-self: center; font-size: 0.9rem; margin-left: 0.5rem;">API Docs</a>
                </div>
            </div>

            <div class="main-content-grid">
                <div class="left-column">
                    <div class="grid">
                        <div class="card">
                            <div class="metric-label">Forecasted Wind Speed (10 min ahead)</div>
                            <div class="metric-value" id="forecastWind">-- m/s</div>
                            <div class="metric-caption" id="forecastCaption">Waiting for data...</div>
                        </div>
                        <div class="card">
                            <div class="metric-label">Recommended Pitch Angle</div>
                            <div class="metric-value" id="recommendedPitch">-- °</div>
                            <div class="metric-caption" id="pitchCaption">Waiting for data...</div>
                        </div>
                    </div>

                    <div class="chart-card">
                        <div class="visualization-label">Wind Speed History & Forecast</div>
                        <div class="chart-container">
                            <canvas id="windChart"></canvas>
                        </div>
                    </div>
                </div>

                <div class="visualization-card">
                    <div class="visualization-label">Recommended Pitch Angle Visualization (3D Model)</div>
                    <div id="turbineCanvasContainer">
                        <div class="pitch-badge">Active Pitch: <span id="pitchAngleOverlay">0.00°</span></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let streamInterval = null;
            const ctx = document.getElementById('windChart').getContext('2d');
            
            // Light shading behind the forecast interval
            const forecastShade = {
                id: 'forecastShade',
                beforeDatasetsDraw(c) {
                    const { ctx, chartArea, scales } = c;
                    const n = c.data.labels.length;
                    const xStart = scales.x.getPixelForValue(n - 2);
                    ctx.save();
                    ctx.fillStyle = 'rgba(251, 146, 60, 0.10)';
                    ctx.fillRect(xStart, chartArea.top, chartArea.right - xStart, chartArea.bottom - chartArea.top);
                    ctx.restore();
                }
            };

            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['-50m', '-40m', '-30m', '-20m', '-10m', 'Now', 'Forecast (+10m)'],
                    datasets: [
                        {
                            label: 'Measured wind speed',
                            data: [0, 0, 0, 0, 0, 0, null],
                            borderColor: '#38bdf8',
                            backgroundColor: 'rgba(56, 189, 248, 0.1)',
                            fill: true,
                            tension: 0.3,
                            borderWidth: 3,
                            pointRadius: 4
                        },
                        {
                            label: 'Forecast (next 10 min)',
                            data: [null, null, null, null, null, 0, 0],
                            borderColor: '#fb923c',
                            backgroundColor: '#fb923c',
                            borderDash: [6, 5],
                            borderWidth: 3,
                            fill: false,
                            tension: 0,
                            pointRadius: [0, 0, 0, 0, 0, 0, 8],
                            pointHoverRadius: [0, 0, 0, 0, 0, 0, 10],
                            pointStyle: 'rectRot'
                        }
                    ]
                },
                plugins: [forecastShade],
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#f8fafc', usePointStyle: true, font: { size: 14 } } } },
                    scales: {
                        x: { ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: false, font: { size: 13 } }, grid: { color: '#334155' } },
                        y: { title: { display: true, text: 'Wind speed (m/s)', color: '#94a3b8', font: { size: 13 } }, ticks: { color: '#94a3b8', font: { size: 13 } }, grid: { color: '#334155' } }
                    }
                }
            });

            // --- Three.js Setup with 3/4 Perspective ---
            let scene, camera, renderer, turbineModel;
            let bladeMeshes = [];
            let currentPitchAngle = 0;

            function init3D() {
                const container = document.getElementById('turbineCanvasContainer');
                const width = container.clientWidth;
                const height = container.clientHeight;

                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1e293b);

                camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);

                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(width, height);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                // Lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
                scene.add(ambientLight);
                
                const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
                dirLight1.position.set(5, 10, 7);
                scene.add(dirLight1);

                const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.5);
                dirLight2.position.set(-5, -5, -5);
                scene.add(dirLight2);

                // Load 3D Model
                const loader = new THREE.GLTFLoader();
                loader.load(
                    '/static/turbine.glb',
                    (gltf) => {
                        turbineModel = gltf.scene;

                        const box = new THREE.Box3().setFromObject(turbineModel);
                        const center = box.getCenter(new THREE.Vector3());
                        const size = box.getSize(new THREE.Vector3());
                        turbineModel.position.sub(center);

                        bladeMeshes = [];
                        turbineModel.traverse((node) => {
                            if (node.isMesh && (node.name.toLowerCase().includes('blade') || node.name.toLowerCase().includes('rotor'))) {
                                bladeMeshes.push(node);
                            }
                        });

                        scene.add(turbineModel);

                        // Fit model dynamically into full view
                        const maxDim = Math.max(size.x, size.y, size.z);
                        camera.position.set(maxDim * 0.7, maxDim * 0.25, maxDim * 0.95);
                        camera.lookAt(0, size.y * 0.05, 0);
                    },
                    undefined,
                    (error) => {
                        console.error('Error loading 3D model:', error);
                    }
                );

                window.addEventListener('resize', onWindowResize, false);
            }

            function onWindowResize() {
                const container = document.getElementById('turbineCanvasContainer');
                if (!container || !renderer || !camera) return;
                const width = container.clientWidth;
                const height = container.clientHeight;
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            }

            function animate() {
                requestAnimationFrame(animate);
                
                const targetRad = currentPitchAngle * (Math.PI / 180);

                if (bladeMeshes.length > 0) {
                    bladeMeshes.forEach((blade) => {
                        blade.rotation.x = THREE.MathUtils.lerp(blade.rotation.x, targetRad, 0.1);
                    });
                } else if (turbineModel) {
                    turbineModel.rotation.y = THREE.MathUtils.lerp(turbineModel.rotation.y, 0.4 + targetRad * 0.5, 0.1);
                }

                renderer.render(scene, camera);
            }

            // Plain-language captions (thresholds based on this turbine's historical data)
            function describeWind(forecast, recent) {
                const last = recent[recent.length - 1];
                const diff = forecast - last;
                let trend = 'Staying about the same as now.';
                if (diff > 0.3) trend = 'Rising, about ' + diff.toFixed(1) + ' m/s more than now.';
                else if (diff < -0.3) trend = 'Falling, about ' + Math.abs(diff).toFixed(1) + ' m/s less than now.';

                let level;
                if (forecast < 3.5) level = 'Very light wind: little electricity expected.';
                else if (forecast < 6) level = 'Light wind: modest electricity output expected.';
                else if (forecast < 10) level = 'Moderate wind: good electricity output expected.';
                else level = 'Strong wind: the turbine is at or near full output.';
                return trend + ' ' + level;
            }

            function describePitch(pitch, forecastWind) {
                if (pitch < 1) return 'Blades stay flat to catch as much wind as possible.';
                if (pitch < 5) return 'Blades start to tilt, letting some wind spill to keep the rotor speed steady.';
                if (forecastWind < 9) return 'Blades are turned well out of the wind, as when the turbine is idling or restarting.';
                return 'Blades tilt noticeably to hold a safe rotor speed in strong wind.';
            }

            async function fetchNext() {
                try {
                    const res = await fetch('/predict/next');
                    if (!res.ok) {
                        if (res.status === 404) alert("End of data stream reached.");
                        return;
                    }
                    const data = await res.json();
                    
                    document.getElementById('timestamp').innerText = 'Simulated live stream (replay of Kelmarsh turbine data) \u00b7 Data time: ' + data.timestamp;
                    document.getElementById('forecastWind').innerText = data.forecasted_wind_speed.toFixed(2) + ' m/s';
                    document.getElementById('recommendedPitch').innerText = data.recommended_pitch_angle.toFixed(2) + '°';
                    document.getElementById('pitchAngleOverlay').innerText = data.recommended_pitch_angle.toFixed(2) + '°';
                    document.getElementById('forecastCaption').innerText = describeWind(data.forecasted_wind_speed, data.recent_wind_speeds);
                    document.getElementById('pitchCaption').innerText = describePitch(data.recommended_pitch_angle, data.forecasted_wind_speed);

                    const recent = data.recent_wind_speeds;
                    const lastIdx = recent.length - 1;
                    chart.data.datasets[0].data = [...recent, null];
                    chart.data.datasets[1].data = [
                        ...recent.map((v, i) => (i === lastIdx ? v : null)),
                        data.forecasted_wind_speed
                    ];
                    chart.update();

                    currentPitchAngle = data.recommended_pitch_angle;
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

            init3D();
            animate();
            fetchNext();
        </script>
    </body>
    </html>
    """


@app.get("/predict/next", response_model=PredictionResponse)
def predict_next():
    pos = cursor["position"]
    if pos >= len(stream_df):
        raise HTTPException(status_code=404, detail="End of simulated stream reached")

    window = stream_df["Wind speed (m/s)"].values[pos - WINDOW_SIZE:pos]
    window_scaled = wind_forecast_scaler.transform(window.reshape(1, -1))

    forecasted_wind = wind_forecast_model.predict(window_scaled)[0]

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

    cursor["position"] += 1

    return PredictionResponse(
        timestamp=str(latest_row["Date and time"]),
        recent_wind_speeds=window.tolist(),
        forecasted_wind_speed=float(forecasted_wind),
        recommended_pitch_angle=float(predicted_pitch),
    )


@app.post("/reset")
def reset_stream():
    cursor["position"] = WINDOW_SIZE
    return {"status": "reset", "position": cursor["position"]}