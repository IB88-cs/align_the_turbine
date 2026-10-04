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

Our model is connected to the application through an API, allowing predictions to be requested and displayed through the user interface.

### 5. Accessible Interface

Different users need different information from the same data.

Our application therefore aims to provide different views for:

- **Wind operators and engineers** - Detailed technical information and predictions
- **Energy organisations and decision-makers** - System-level renewable energy insights
- **Communities and non-technical users** - clear, accessible explanations

An LLM acts as an interpretation layer to help explain technical model outputs in language appropriate to different users.

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

**original pitch model** (with rotor position, nacelle position - cuz irl we won't have these features in real time)
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
## Technology

### Machine Learning

### Backend

### Frontend

### LLM

### Data

---
## Limitations

- mainly that the MLP model isn't as accurate as CNN - an MLP model is just simpler to set up, and thus was used for prototype purposes

This project is a **proof of concept developed during the hackathon**. 

The model uses historical turbine data and has not been validated for autonomous control of real wind turbines.

Real turbine operation involves additional engineering, environmental and safety constraints that are outside the scope of this prototype.

Further real-world testing and engineering validation would be required before the system could be used operationally.

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
- **Ishani Basu** - Research, backend, training the model, idea design (team lead btw)
- **Charan Pedireddi** - Frontend
- **Ipsa Chatterjee** - Frontend

---
## Tools, Data & AI Disclosure

This project was developed during **Climate Hack-tion 2026**.

External tools and resouces used include:

- Dataset: [Kelmarsh wind farm data, published by Zenodo on Cubico Sustainable] - using the 2017 dataset
- Machine Learning Libraries: scikit-learn, pandas, NumPy, joblib, FastAPI, Uvicorn, Chart.js, Three.js. 3D turbine model: static/turbine.glb.
- API: Open-Meteo API ((free, non-commercial use), https://open-meteo.com/
- AI coding/generation tools: Claude
- Other tools: Canva for pitch slides, Jupyter notebooks for data wrangling and model training

All project specific development was completed during the hackathon period.

---
## Demo

**Demo video:** [link to be added]

**Live application:** [link to be added]

**Repository:** [link to be added]





