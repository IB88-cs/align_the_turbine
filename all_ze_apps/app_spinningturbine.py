import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import time
import threading
import urllib.parse
import urllib.request
from collections import deque

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


# ---- Optional LIVE mode (Open-Meteo, 10 m wind). Never needed for the replay demo. ----
LIVE_LAT, LIVE_LON = -37.5961, 143.4942   # Chepstowe wind farm, Victoria (thewindpower.net)
LIVE_POLL_SECONDS = 300
OPEN_METEO = "https://api.open-meteo.com/v1/forecast?"
live_buf = deque(maxlen=WINDOW_SIZE)       # (timestamp, wind speed m/s)
live_state = {"gust": 0.0, "dir": 0.0, "ok": None}
try:
    live_pitch_model = joblib.load("simplified_pitch_model.pkl")
    live_pitch_scaler = joblib.load("simplified_pitch_scaler.pkl")
except FileNotFoundError:
    live_pitch_model = live_pitch_scaler = None
    print("Live mode disabled: simplified_pitch_model.pkl / simplified_pitch_scaler.pkl not found.")


def _get_json(params):
    base = {"latitude": LIVE_LAT, "longitude": LIVE_LON, "wind_speed_unit": "ms"}
    with urllib.request.urlopen(OPEN_METEO + urllib.parse.urlencode({**base, **params}), timeout=10) as r:
        return json.load(r)


def _poll_live():
    cur = _get_json({"current": "wind_speed_10m,wind_direction_10m,wind_gusts_10m"})["current"]
    if not live_buf or live_buf[-1][0] != cur["time"]:     # source updates ~every 15 min; skip repeats
        live_buf.append((cur["time"], float(cur["wind_speed_10m"])))
    live_state.update(gust=cur["wind_gusts_10m"] or 0.0, dir=cur["wind_direction_10m"] or 0.0, ok=True)


def _backfill_live():
    """Try to pre-fill the buffer with recent 15-min readings so live mode starts quickly. Fails quietly."""
    try:
        m = _get_json({"minutely_15": "wind_speed_10m", "past_minutely_15": WINDOW_SIZE + 2,
                       "forecast_minutely_15": 1})["minutely_15"]
        now = _get_json({"current": "wind_speed_10m"})["current"]["time"]
        for t, v in zip(m["time"], m["wind_speed_10m"]):
            if v is not None and t <= now and (not live_buf or t > live_buf[-1][0]):
                live_buf.append((t, float(v)))
    except Exception as e:
        print("Live backfill skipped:", e)


def _live_loop():
    _backfill_live()
    while True:
        try:
            _poll_live()
        except Exception as e:
            live_state["ok"] = False
            print("Live poll failed:", e)
        time.sleep(LIVE_POLL_SECONDS)


@app.on_event("startup")
def _start_live():
    if live_pitch_model is not None:
        threading.Thread(target=_live_loop, daemon=True).start()


class PredictionResponse(BaseModel):
    timestamp: str
    recent_wind_speeds: list[float]
    recent_timestamps: list[str]
    step_seconds: int = 600
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
                min-height: 100vh;
                margin: 0;
                padding: 0;
                overflow-x: hidden;
                font-family: system-ui, -apple-system, sans-serif;
                background-color: var(--bg);
                color: var(--text);
            }
            
            /* Root Layout Wrapper */
            .app-wrapper {
                display: flex;
                flex-direction: column;
                min-height: 100vh;
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
                min-height: 340px;
                display: flex;
                flex-direction: column;
            }
            .chart-container {
                position: relative;
                flex: 1;
                width: 100%;
                height: 100%;
                min-height: 260px;
            }

            .visualization-card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 1.25rem;
                display: flex;
                flex-direction: column;
                height: 100%;
                min-height: 460px;
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

            .pitch-gauge { position: absolute; left: 12px; bottom: 12px; z-index: 10; background: rgba(15, 23, 42, 0.85);
                           border: 1px solid #334155; border-radius: 8px; padding: 6px 10px 4px; text-align: center; }
            .gauge-caption { color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
            .source-row { display: flex; align-items: center; gap: 0.6rem; margin-top: 0.4rem; flex-wrap: wrap; }
            .source-badge { font-size: 0.8rem; font-weight: 800; letter-spacing: 0.08em; padding: 0.25rem 0.75rem; border-radius: 999px; border: 1px solid; }
            .source-badge.replay { color: #c4b5fd; background: rgba(139, 92, 246, 0.18); border-color: #8b5cf6; }
            .source-badge.live { color: #4ade80; background: rgba(34, 197, 94, 0.18); border-color: #22c55e; }
            .source-badge.live::before { content: ''; display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; background: #22c55e; margin-right: 0.45rem; animation: pulse 1.4s infinite; }
            @keyframes pulse { 50% { opacity: 0.25; } }
            .source-name { font-size: 1rem; font-weight: 600; }
        </style>
    </head>
    <body>
        <div class="app-wrapper">
            <div class="header">
                <div>
                    <h1>Wind Turbine Live Monitoring</h1>
                    <div class="source-row"><span id="sourceBadge" class="source-badge replay">REPLAY</span><span id="sourceName" class="source-name">Simulated stream of recorded Kelmarsh turbine data</span></div>
                    <div class="subtitle" id="timestamp">Waiting for data...</div>
                </div>
                <div class="controls">
                    <button id="modeReplay" onclick="setMode('replay')" class="active">Replay</button>
                    <button id="modeLive" class="secondary" onclick="setMode('live')">Live</button>
                    <button id="nextBtn" onclick="fetchNext()">Next Step</button>
                    <button id="streamBtn" class="secondary" onclick="toggleStream()">Start Auto-Stream</button>
                    <button id="resetBtn" class="secondary" onclick="resetStream()">Reset</button>
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

                    <div class="card">
                        <div class="metric-label">Forecast check: model vs. "same as last reading"</div>
                        <div class="metric-caption" id="accuracyText">Collecting results...</div>
                    </div>

                    <div class="chart-card">
                        <div class="visualization-label">Wind Speed History & Forecast</div>
                        <div class="chart-container">
                            <canvas id="windChart"></canvas>
                        </div>
                    </div>
                </div>

                <div class="visualization-card">
                    <div class="visualization-label" id="vizLabel">Recommended Pitch Angle Visualization</div>
                    <div id="turbineCanvasContainer">
                        <div class="pitch-badge">Active Pitch: <span id="pitchAngleOverlay">0.00°</span></div>
                        <div class="pitch-gauge">
                            <svg viewBox="0 0 120 80" width="160">
                                <path d="M10 62 A50 50 0 0 1 110 62" fill="none" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
                                <path id="gaugeFill" d="M10 62 A50 50 0 0 1 110 62" fill="none" stroke="#38bdf8" stroke-width="8" stroke-linecap="round" pathLength="100" stroke-dasharray="0 100"/>
                                <line id="gaugeNeedle" x1="60" y1="62" x2="60" y2="22" stroke="#f8fafc" stroke-width="3" stroke-linecap="round" style="transform-box: view-box; transform-origin: 60px 62px; transition: transform 0.6s;"/>
                                <circle cx="60" cy="62" r="4" fill="#f8fafc"/>
                                <text x="6" y="77" fill="#94a3b8" font-size="8">0° flat</text>
                                <text x="114" y="77" fill="#94a3b8" font-size="8" text-anchor="end">90° feathered</text>
                            </svg>
                            <div class="gauge-caption">Blade pitch</div>
                        </div>
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
                    labels: [],
                    datasets: [
                        { label: 'Measured wind speed', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.1)',
                          fill: true, tension: 0.2, borderWidth: 3, pointRadius: 3, spanGaps: false },
                        { label: 'Forecast (next 10 min)', data: [], borderColor: '#fb923c', backgroundColor: '#fb923c',
                          borderDash: [6, 5], borderWidth: 3, fill: false, tension: 0, pointRadius: [], pointStyle: 'rectRot' },
                        { label: 'Earlier forecast for that time', data: [], showLine: false, borderColor: '#fb923c',
                          backgroundColor: 'rgba(0,0,0,0)', pointStyle: 'circle', pointRadius: 5, pointBorderWidth: 2 }
                    ]
                },
                plugins: [forecastShade],
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        legend: { labels: { color: '#f8fafc', usePointStyle: true, boxWidth: 8, font: { size: 12 } } },
                        tooltip: {
                            // the orange line starts on the last measured point only to connect the two: hide it there
                            filter: (item) => !(item.datasetIndex === 1 && item.dataIndex < item.chart.data.labels.length - 1),
                            callbacks: { label: (item) => ['Measured: ', 'Forecast for next step: ', 'Earlier forecast for this time: '][item.datasetIndex] + item.parsed.y.toFixed(2) + ' m/s' }
                        }
                    },
                    scales: {
                        x: { ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, maxTicksLimit: 9, font: { size: 12 } }, grid: { color: '#334155' } },
                        y: { title: { display: true, text: 'Wind speed (m/s)', color: '#94a3b8', font: { size: 13 } }, ticks: { color: '#94a3b8', font: { size: 13 } }, grid: { color: '#334155' } }
                    }
                }
            });

            // --- Rolling history kept in the browser (real timestamps, gaps marked) ---
            const MAX_HIST = 24; let STEP_MS = 600000, mode = 'replay', liveTimer = null;
            let hist = [], pending = null, lastTs = null, err = { n: 0, model: 0, base: 0 };

            function addPoint(ts, v) {
                const t = Date.parse(ts.replace(' ', 'T') + 'Z');
                if (lastTs !== null && t <= lastTs) return; // same reading as before: nothing new to add
                let prevF = null;
                if (lastTs !== null && t - lastTs !== STEP_MS) {
                    hist.push({ label: '\u2026', measured: null, prevForecast: null }); // data gap (turbine stopped)
                    pending = null; // a forecast can't be scored across a gap
                }
                if (pending) {
                    prevF = pending.forecast;
                    err.n++;
                    err.model += Math.abs(v - pending.forecast);
                    err.base += Math.abs(v - pending.base);
                }
                hist.push({ label: ts.slice(11, 16), measured: v, prevForecast: prevF });
                lastTs = t;
                while (hist.length > MAX_HIST) hist.shift();
            }

            function renderChart(forecast) {
                const n = hist.length;
                chart.data.labels = [...hist.map(h => h.label), 'Forecast +10m'];
                chart.data.datasets[0].data = [...hist.map(h => h.measured), null];
                chart.data.datasets[1].data = [...hist.map((h, i) => (i === n - 1 ? h.measured : null)), forecast];
                chart.data.datasets[1].pointRadius = [...hist.map(() => 0), 8];
                chart.data.datasets[2].data = [...hist.map(h => h.prevForecast), null];
                chart.update();
            }

            function renderAccuracy() {
                const el = document.getElementById('accuracyText');
                if (err.n < 3) { el.innerText = 'Collecting results... (compares each forecast with what actually happened)'; return; }
                const m = err.model / err.n, b = err.base / err.n, r = m / b;
                const verdict = r < 0.97 ? 'slightly better than' : (r > 1.03 ? 'worse than' : 'about the same as');
                el.innerText = 'Over ' + err.n + ' steps, average forecast error is ' + m.toFixed(2) + ' m/s. Simply repeating the last reading gives ' +
                    b.toFixed(2) + ' m/s, so the model is ' + verdict + ' that baseline at 10 minutes.';
            }

            // --- Three.js Setup with 3/4 Perspective ---
            let scene, camera, renderer, turbineModel;
            let rotorPivot = null, rotorAxis = 'z', spinSpeed = 0;
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

                        scene.add(turbineModel);

                        // Spin the rotor about the hub. The blades are one piece in this model, so true per-blade pitch can't be shown (see the dial).
                        turbineModel.updateMatrixWorld(true);
                        let bladesNode = null;
                        turbineModel.traverse((n) => { if (!bladesNode && n.name.toLowerCase().includes('blades')) bladesNode = n; });
                        if (bladesNode) {
                            const bs = new THREE.Box3().setFromObject(bladesNode).getSize(new THREE.Vector3());
                            rotorAxis = (bs.x <= bs.y && bs.x <= bs.z) ? 'x' : (bs.y <= bs.z ? 'y' : 'z');
                            rotorPivot = new THREE.Group();
                            rotorPivot.position.copy(bladesNode.getWorldPosition(new THREE.Vector3())); // node origin = hub centre
                            scene.add(rotorPivot);
                            rotorPivot.attach(bladesNode);
                        }

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
                
                if (rotorPivot && spinSpeed) rotorPivot.rotation[rotorAxis] += spinSpeed;

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
                    const res = await fetch(mode === 'live' ? '/predict/live' : '/predict/next');
                    if (!res.ok) {
                        if (mode === 'live') {
                            const d = await res.json().catch(() => ({}));
                            if (res.status === 425) {
                                const msg = 'Collecting live data: ' + d.detail + '/6 readings. This takes a while because the weather source updates every ~15 minutes.';
                                document.getElementById('forecastCaption').innerText = msg;
                                document.getElementById('accuracyText').innerText = msg;
                            } else {
                                setMode('replay', (d.detail || 'Live feed unavailable') + ' - back to replay.');
                            }
                            return;
                        }
                        if (res.status === 404) alert("End of data stream reached.");
                        return;
                    }
                    const data = await res.json();
                    STEP_MS = (data.step_seconds || 600) * 1000;
                    
                    updateSource('Data time: ' + data.timestamp + (mode === 'live' ? ' UTC' : ''));
                    document.getElementById('forecastWind').innerText = data.forecasted_wind_speed.toFixed(2) + ' m/s';
                    document.getElementById('recommendedPitch').innerText = data.recommended_pitch_angle.toFixed(2) + '°';
                    document.getElementById('pitchAngleOverlay').innerText = data.recommended_pitch_angle.toFixed(2) + '°';
                    document.getElementById('forecastCaption').innerText = describeWind(data.forecasted_wind_speed, data.recent_wind_speeds);
                    document.getElementById('pitchCaption').innerText = describePitch(data.recommended_pitch_angle, data.forecasted_wind_speed) +
                        (mode === 'live' ? ' (Demo only: live input is 10 m weather wind, not turbine data.)' : '');

                    const recent = data.recent_wind_speeds, stamps = data.recent_timestamps;
                    if (lastTs === null) recent.forEach((v, i) => addPoint(stamps[i], v));
                    else addPoint(stamps[stamps.length - 1], recent[recent.length - 1]);
                    pending = { forecast: data.forecasted_wind_speed, base: recent[recent.length - 1] };
                    renderChart(data.forecasted_wind_speed);
                    renderAccuracy();

                    currentPitchAngle = data.recommended_pitch_angle;
                    updateGauge(currentPitchAngle);
                    // rotor only turns in replay, and only above a rough cut-in wind speed
                    spinSpeed = (mode === 'replay' && data.forecasted_wind_speed >= 3) ? Math.min(data.forecasted_wind_speed, 12) / 12 * 0.03 : 0;
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
                    streamInterval = setInterval(fetchNext, mode === 'live' ? 30000 : 2000);
                    btn.innerText = "Pause Auto-Stream";
                    btn.classList.remove('secondary');
                    btn.classList.add('active');
                }
            }

            function updateGauge(p) {
                const c = Math.max(0, Math.min(90, p));
                document.getElementById('gaugeNeedle').style.transform = 'rotate(' + ((c - 45) * 2) + 'deg)';
                document.getElementById('gaugeFill').setAttribute('stroke-dasharray', (c / 90 * 100) + ' 100');
            }

            function updateSource(timeText) {
                const live = mode === 'live';
                const b = document.getElementById('sourceBadge');
                b.className = 'source-badge ' + (live ? 'live' : 'replay');
                b.innerText = live ? 'LIVE WEATHER' : 'REPLAY';
                document.getElementById('sourceName').innerText = live
                    ? 'Real-time wind near Chepstowe wind farm, Victoria (Open-Meteo, 10 m height, not turbine data)'
                    : 'Simulated stream of recorded Kelmarsh turbine data';
                document.getElementById('timestamp').innerText = timeText;
                document.getElementById('vizLabel').innerText = live ? 'Pitch dial (illustration only: live feed is weather data)' : 'Recommended Pitch Angle Visualization';
            }

            function setMode(m, note) {
                if (streamInterval) toggleStream();
                mode = m;
                clearInterval(liveTimer); liveTimer = null;
                spinSpeed = 0;
                ['nextBtn', 'streamBtn', 'resetBtn'].forEach(id => { document.getElementById(id).style.display = (m === 'live') ? 'none' : ''; });
                if (m === 'live') liveTimer = setInterval(fetchNext, 30000);   // new data only arrives every ~15 min
                hist = []; pending = null; lastTs = null; err = { n: 0, model: 0, base: 0 };
                document.getElementById('modeReplay').className = m === 'replay' ? 'active' : 'secondary';
                document.getElementById('modeLive').className = m === 'live' ? 'active' : 'secondary';
                updateSource(note || 'Waiting for data...');
                renderChart(null);
                renderAccuracy();
                fetchNext();
            }

            async function resetStream() {
                if (mode === 'replay') await fetch('/reset', { method: 'POST' });
                hist = []; pending = null; lastTs = null; err = { n: 0, model: 0, base: 0 };
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
        recent_timestamps=[str(t) for t in stream_df["Date and time"].values[pos - WINDOW_SIZE:pos].astype("datetime64[s]")],
        forecasted_wind_speed=float(forecasted_wind),
        recommended_pitch_angle=float(predicted_pitch),
    )


@app.get("/predict/live", response_model=PredictionResponse)
def predict_live():
    if live_pitch_model is None:
        raise HTTPException(status_code=503, detail="Live mode unavailable: model files missing")
    n = len(live_buf)
    if n < WINDOW_SIZE:
        if n == 0 and live_state["ok"] is False:
            raise HTTPException(status_code=503, detail="Live weather unavailable")
        raise HTTPException(status_code=425, detail=str(n))
    stamps = [t for t, _ in live_buf]
    window = np.array([v for _, v in live_buf])
    forecast = float(wind_forecast_model.predict(wind_forecast_scaler.transform(window.reshape(1, -1)))[0])
    # Std-dev proxy: gust-minus-speed is roughly 3 standard deviations. Training std-dev averages ~0.9 m/s.
    std_proxy = max(live_state["gust"] - forecast, 0.0) / 3.0
    x = np.array([[forecast, std_proxy, live_state["dir"]]])
    z = np.clip(live_pitch_scaler.transform(x), -3, 3)   # keep inputs inside the range the model was trained on
    pitch = float(live_pitch_model.predict(z)[0])
    step = int((pd.Timestamp(stamps[-1]) - pd.Timestamp(stamps[-2])).total_seconds()) or 900
    return PredictionResponse(timestamp=stamps[-1], recent_wind_speeds=window.tolist(), recent_timestamps=stamps,
                              forecasted_wind_speed=forecast, recommended_pitch_angle=pitch, step_seconds=step)


@app.post("/reset")
def reset_stream():
    cursor["position"] = WINDOW_SIZE
    return {"status": "reset", "position": cursor["position"]}