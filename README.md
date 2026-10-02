# Align_The_Turbine
Predicting wind conditions to support smarter turbine optimisation and more accessible renewable energy intelligence.

Built for Climate Hack-tion 2026 - Build for 2035

COP31 Priority: **Electrification**

## The Problem
As transport, buildings and industries become increasing electrified, reducing dependence on fossil fuels requires not only greater electricity genereation, but also more effective use of loww-emission energy sources.

Wind energy can support this transition, however wind conditions are inherently variable. Changes in wind speed and direction affect how turbines operate and how much electricity they can generate.

This led us to explore a simple question:

> What if wind turbines could anticipate changing wind conditions instead of only reacting to them?

## Out Solution
**Aligh the Turbine** is an AI-powered proof of concept that uses real-world wind turbine data to predict future wind conditions.

Our machine-learning model analyses historical wind and turbine data to forecast changes in wind conditions. These predictions can be used to explore how turbine settings, such as blade pitch, could respond proactively to improve energy capture.

We are connecting the model to an accessible application through an API, allowing users to interact with the predictions without needing to work directly with raw data or machine-learning tools.

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
- Date and Time

### 2. Machine Learning

The data is processed and used to train a machine-learning model that predicts future wind conditions.

### 3. Prediction Optimisation

The predicted wind conditions can be combined with turbine operational data to explore how settings such as blade pitch could be adjusted proactively.

### 4. API

Our model is connected to the application through an API, allowing predictions to be requested and displayed through the user interface.

### 5. Accessible Interface

Different users need different information from the same data.

Our application therefore aims to provide different views for:

- **Wind operators and engineers** - Detailed technical information and predictions
- **Energy organisations and decision-makers** - System-level renewable energy insights
- **Communities and non-technical users** - clear, accessible explanations

An LLM acts as an interpretation layer to help explain techincal model outputs in language appropriate to different users.

---
## Model Performance

We compare the model's
