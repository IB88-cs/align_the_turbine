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

We compare the model's predicted wind speed against the actual recorded wind speed to evaluate its forecasting performance.

### Actual vs Predicted Wind Spped

Graph to be added here.

### Evaluation

- **MAE:** [add result]
-  **RMSE:** [add result]
-  **Prediction Horizon:** [add result]

**MAE (Mean Absolute Error)** measures the average difference between predicted and actual wind speed.

**RMSE (Root Mean Squared Error)** also measures prediction error but gives greater weight to larger errors.

---
## COP31 Alignment

Our project addresses the **COP31 Electrification** priority.

Electrifying transportm buildings and industry can reduce dependence on fossil fuels, but this transistion also increases the important of efficient and reliable low-emission electricity generation.

By exploring how predictive modelling could support more effective utilisation of wind energy, our project demonstrates how technology can contribute to the trasition towards an increasingly electrified, low-emission energy system.

---
## Why It Matters

Climate action is not only a technical problem.

From an environmental politics perspective, the transition towards renewable energy also involves **energy security, sustainable development, resource allocation and environmental justice**.

Increasing the effective use of wind energy can contribute to reducing dependence on fossil fuels, it may also reduce pressure on some land-intensive energy pathways, including crop-based biofuels that can compete with agricultural resources and food production.

We also recognise that access to data does not necessarily mean access to knowledge. Complex renewable-energy information can be difficlut for people without technical expertise to understnad.

Our application therefore aims to make the same evidence accessible to users with different levels of technical knowledge.

## Target Users

### Wind Operators & Engineers

Access technical wind forecasts, turbine information and model outputs.

### Energy Organisations & Decision-Makers

Understand renewable energy trends and potential system-level implications

### Communities

Access clear explanation of wind energy, forecasts and their relevance to the energy transition.

---
## Technology

### Machine Learning

### Backend

### Frontend

### LLM

### Data

---
## Limitations

This project is a **proof of concept developed during the hackathon**. 

The model uses historical turbine data and has not been validated for autonomous control of real wind turbines.

Real turbine operation involves additional engineering, environmental and safety constraints that are outside the scope of this prototype.

The LLM is used to help interpret and communicate model outputs. It does not independently generate or validate turbine-control decisions.

Further real-world testing and engineering validation would be required before the system could be used operationally.

## Futre Development

Future development could include:

- Intergration with live turbine and weather data
- Testing across different turbine models and locations
- Further development of pitch-angle optimisation
- Read Time prediction
- Model uncertainty and confidence information
- Testing with wind-energy professionals
- Improved accessibility
- Multilingual Support

---
## Team

**Henriette Fung** - Documentation
**name** - 
**name** -
**name** -

---
## Tools, Data & AI Disclosure

This project was developed during **Climate Hack-tion 2026**.

External tools and resouces used include:

- Dataset: [name + source]
- Machine Learning Libraries:
- API:
- LLM:
- AI coding/generation tools:
- Other tools:

All project specific development was completed during the hackathin period.

---
## Demo

**Demo video:** [link to be added]

**Live application:** [link to be added]

**Repository:** [link to be added]





