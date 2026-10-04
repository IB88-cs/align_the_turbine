# Align_The_Turbine
Predicting wind conditions to support smarter turbine optimisation and more accessible renewable energy intelligence.

Built for Climate Hack-tion 2026 - Build for 2035

COP31 Priority: **Electrification**

## The Problem
As transport, buildings and industries become increasingly electrified, reducing dependence on fossil fuels requires not only greater electricity genereation, but also more effective use of low-emission energy sources.

Wind energy can support this transition, however wind conditions are inherently variable. Changes in wind speed and direction affect how turbines operate and how much electricity they can generate.

This led us to explore a simple question:

> What if wind turbines could anticipate changing wind conditions instead of only reacting to them?

## Our Solution
**Align the Turbine** is an ML-powered proof of concept that uses real-world wind turbine data to predict future wind conditions.

1. **Forecasts** wind speed 10 minutes ahead from the last hour of readings (a small neural network, scikit-learn `MLPRegressor`).
2. **Estimates the blade pitch** a turbine typically used in similar conditions (a second `MLPRegressor`).
3. **Estimates the electricity output** for the forecast wind, using the average power curve of the turbine we have data for.
4. **Explains it in plain language** in a dashboard, so non-technical viewers can follow wind speed, expected power and blade angle without reading raw data.

[Trained a MLP (multi layer perceptron) model to predict the wind for the next 10 minutes (the next window). Even though predicting the wind and adjusting the wind is an active field of research and field-tested technology called lidar-assisted feedforward pitch control. But the key difference between our model and this technology, is that our model is trained to predict the gust for the next 10 minutes. ] - added by Ishani


Our approach is:

**Predict → Optimise → Understand**

## How It Works
### 1. Real-World Turbine Data

Out model uses historical turbine data including:

- Wind speed
- Wind spped variability
- Wind direction
- Nacelle position
- Rotor speed
- Power output
- Blade pitch angle
- Turbine operating status
- Date and Time (10 minute intervals)

### 2. Machine Learning

The data is processed and used to train a machine-learning model (MLP) that predicts future wind conditions.

### 3. Prediction Optimisation

The predicted wind conditions can be combined with turbine operational data to explore how settings such as blade pitch could be adjusted proactively.

### 4. API

[Using the Open-Meteo API we have taken live data from the Chepstowe Windfarm, and used them to predict the data for the next window, that way we aren't just relying on historic data, but also making predictions based on the current conditions of a wind farm. There is no API key needed and it is free for non-commercial use] 


### 5. Accessible Interface
## What the dashboard shows

| Card / chart | What it means |
|---|---|
| Forecasted wind speed | The model's prediction for 10 minutes from now, with a plain-language label (light, moderate, strong wind) |
| Expected power (estimate) | Forecast wind converted to kilowatts using the average power curve from the recorded turbine data. An estimate only |
| Recommended pitch angle | The blade angle this turbine typically used in similar conditions (0° = flat to the wind, 90° = feathered) |
| Wind speed graph | Recent measured wind (blue line), the next forecast (orange diamond) and, in Replay, what the model had predicted earlier for each time (orange circles) |
| 3D turbine (Replay only) | An illustration. The rotor turns a little with each step, further in stronger wind |

**Two data modes**

- **Replay** (default): steps through recorded 10-minute data from Kelmarsh turbine 1 (a continuous 9-day stretch, 24 Nov to 3 Dec 2017), as if it were arriving live.
- **Live (demo)**: uses real-time wind near the Chepstowe wind farm, Victoria, from the free [Open-Meteo] API. The numbers are real calculations on that feed, but it is weather-model wind for the area, not a measurement at a turbine, so treat Live as a demonstration of the pipeline, not a validated result. Times are shown in Melbourne time (AEST/AEDT).


---
## Model Performance

We compare the model's predicted wind speed against the actual recorded wind speed to evaluate its forecasting performance.

### Actual vs Predicted Wind Speed

Graph to be added here.

### Evaluation

**wind forecast model**
- mean absolute error: 0.563 m/s
- root mean squared error: 0.747 m/s
- prediction horizon: 1 (next 10 mins)

**original pitch model** (with rotor position, nacelle position - in real life these position would not change for the turbine alongside live wind data)
- mean absolute error: 0.195°
- root mean squared error: 0.746°
- prediction horizon: 1 (next 10 mins)

**simplified model** (only wind speed, wind direction) 
- mean absolute error: 0.453°
-  root mean squared error: 2.059°
-  prediction horizon: 1 (next 10 mins)

---
## COP31 Alignment

Our project addresses the **COP31 Electrification** priority.

Electrifying transport, buildings and industry can reduce dependence on fossil fuels, but this transition also increases the important of efficient and reliable low-emission electricity generation.

By exploring how predictive modelling could support more effective utilisation of wind energy, our project demonstrates how technology can contribute to the transition towards an increasingly electrified, low-emission energy system.

---
## Why It Matters

Climate action is not only a technical problem.

From an environmental politics perspective, the transition towards renewable energy also involves **energy security, sustainable development, resource allocation and environmental justice**.

Increasing the effective use of wind energy can contribute to reducing dependence on fossil fuels, it may also reduce pressure on some land-intensive energy pathways, including crop-based biofuels that can compete with agricultural resources and food production.

We also recognise that access to data does not necessarily mean access to knowledge. Complex renewable-energy information can be difficult for people without technical expertise to understand.

Our application therefore aims to make the same evidence accessible to users with different levels of technical knowledge.

So, predicting the wind gust's speed to be able to align the angles before the gust arrives is a field of research i.e. lidar-assisted feedforward pitch control. In this case a lidar is mounted on a nacelle/spinner to scan the wind 50-100 meters ahead and adjust the blades accordingly. However, this method is expensive, and the lidar preview is bounded by its scan range. 

With our method of predicting using a ML model trained on historical wind data, it omits the hardware costs, and can provide longer horizons. Moreover, as an ML approach forecasts based on the data collected on-site, there is no per-turbine capital expenditure (Ajitha et al.)

## Target Users

### Wind Operators & Engineers

Access technical wind forecasts, turbine information and model outputs.

---
## Installation steps/ Run it

```bash
pip install fastapi uvicorn scikit-learn joblib pandas numpy
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

Files the app needs in the same folder: `app.py`, `static/turbine.glb`, `Kelmarsh_wrangled.csv`, `live_stream_data.csv`, `wind_forecast_model.pkl`, `wind_forecast_scaler.pkl`, `pitch_model.pkl`, `pitch_scaler.pkl`, and for Live mode `simplified_pitch_model.pkl` and `simplified_pitch_scaler.pkl`. The page loads Chart.js and Three.js from public CDNs, so the browser needs internet access. **[Note. the scikit-learn version the models were trained with is 1.9]**

The notebooks in `notebooks/` cover data wrangling and model training. `make_simplified_scaler.py` rebuilds the scaler for the simplified pitch model.

---
## Limitations

- mainly that the MLP model isn't as accurate as CNN - an MLP model is just simpler to set up, and thus was used for prototype purposes

- This project is a **proof of concept developed during the hackathon**. 

- The model uses historical turbine data and has not been validated for autonomous control of real wind turbines.

- Real turbine operation involves additional engineering, environmental and safety constraints that are outside the scope of this prototype.

- Further real-world testing and engineering validation would be required before the system could be used operationally.
  
- Live mode uses weather-model wind at a nearby location, not turbine readings. It uses a gust-based stand-in for wind variability, receives about 15-minute spacing while the models were trained on 10-minute data, and takes a while to fill its first readings.
  
- The pitch recommendation is not validated against power output, and the 3D model cannot show real per-blade pitch.
  
- This idea works in theory, but hasn't been tested in practice, so the power outputs may be different to the expectations.
  
- Due to data cleaning, there is a gap in 3% of the forecast model's training window, so closing the gap is an improvement to be made. 

## Future Development

Future development could include:

- Intergration with live turbine and weather data
- Testing across different turbine models and locations
- Further development of pitch-angle optimisation
- Real Time prediction - don't we already hv this??
- Model uncertainty and confidence information
- Testing with wind-energy professionals - wdym??
- Improved accessibility
- Multilingual Support/ better usability
- a CNN model used instead of MLP

---
## Team

- **Henriette Fung** - Documentation, pitch, research
- **Ishani Basu** - Research, backend, training the model, idea design (team lead btw), pitch
- **Charan Pedireddi** - Frontend, demo video
- **Ipsa Chatterjee** - Frontend, demo video, 

---
## Tools, Data & AI Disclosure

This project was developed during **Climate Hack-tion 2026**.

External tools and resouces used include:

- Dataset: [Kelmarsh wind farm data, published by Zenodo on Cubico Sustainable] - using the 2017 dataset
- Machine Learning Libraries: scikit-learn, pandas, NumPy, joblib, FastAPI, Uvicorn, Chart.js, Three.js. 3D turbine model: static/turbine.glb.
- API: Open-Meteo API ((free, non-commercial use), https://open-meteo.com/
- AI coding/generation tools: Claude was used for coding assistance on the dashboard and API
- Other tools: Canva for pitch slides, Jupyter notebooks for data wrangling and model training

All project specific development was completed during the hackathon period.

---
## Demo

**Demo video:** [link to be added]









