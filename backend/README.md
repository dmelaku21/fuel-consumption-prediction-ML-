# Fuel Consumption Prediction System for ESLSE 🚛⛽

Overview

The Fuel Consumption Prediction System is a machine learning–based application developed to support Ethiopian Shipping and Logistics Services Enterprise (ESLSE) in accurately estimating fuel usage for vehicle trips.
By analyzing operational parameters such as distance, speed, vehicle load, engine capacity, and vehicle age, the system replaces traditional manual estimation methods with a data-driven predictive approach, improving fuel budgeting, operational planning, and fleet efficiency.

This project demonstrates how predictive analytics and web-based decision support systems can enhance fuel management in large-scale logistics operations.

---
 ## Business Motivation for ESLSE

Fuel represents one of the largest operational costs for ESLSE. Current fuel estimation practices often rely on:

- Fixed fuel-per-kilometer averages

- Historical records and spreadsheets

- Supervisor experience and judgment

## Challenges of current methods:

- Inaccurate fuel budgeting

- Overestimation or underestimation of fuel needs

- Limited identification of inefficient vehicles

- Low transparency and accountability

## Solution

This system predicts fuel consumption at the individual trip level, enabling proactive, data-driven decision-making and better cost control.

### Key Features

- Data-driven fuel prediction
    Machine learning model estimates fuel consumption for each trip

- Feature selection with Lasso Regression
    Identifies the most influential factors affecting fuel usage

- Interpretable model
    Linear Regression allows managers to understand how each variable impacts fuel consumption

- Secure web application
   Streamlit-based UI with login authentication

- Professional dashboard
    Includes analytics, charts, and prediction history

# Dataset

The dataset simulates real ESLSE trip data and includes operationally relevant variables.

Input Features:

Feature	Description
Distance (km):-Total distance traveled in a trip
Average Speed (km/h):- Mean speed during the trip
Vehicle Load (kg):-	Weight of transported cargo
Engine Capacity (liters):-	Vehicle engine size
Vehicle Age (years)	:- Age of the vehicle

### Target Variable:
- **Fuel Consumption (liters)** – Total fuel used for the trip

##  Machine Learning Approach

### Feature Selection
- Feature selection is performed using **Lasso Regression** combined with `SelectFromModel`
- This helps reduce irrelevant features and improves model performance

### Model Training
- Algorithm used: **Linear Regression**
- Data split:
  - 80% Training
  - 20% Testing

### Model Evaluation
The model is evaluated using:
- **Root Mean Squared Error (RMSE)**
- **R² Score**

The trained model achieved a high R² score, indicating strong predictive performance.

---

## . System Architecture
The system consists of three main components:
1. **Training Module** – Trains and saves the machine learning model
2. **Prediction Module** – Loads the trained model and predicts fuel consumption
3. **Web  application** – Streamlit-based UI with authentication and analytics

---

## Web Application (Streamlit)

Secure login system

User-friendly trip data entry

Instant fuel consumption prediction

Prediction history tracking

Interactive charts and analytics
---
## Benefits to ESLSE:

-Accurate fuel budgeting

-Improved route and vehicle efficiency analysis

-Reduced fuel wastage and emissions

-Identification of inefficient vehicles

-Strong decision-support for fleet management

##  Project Structure

```
FUEL_CONSUMPTION_PREDICTION/
│
├── README.md                  # Project documentation
├── requirements.txt           # Required Python libraries
│
├── data/
│   └── fuel_data.csv          # Dataset used for training and evaluation
│
├── backend/
│   ├── train_model.py         # Script to train and evaluate the ML model
│   ├── app.py                 # Basic Streamlit application
│   ├── app_advanced.py        # Advanced Streamlit app (login + dashboard)
│   ├── auth.py                # Authentication logic (login system)
│   ├── predict_fuel_interactive.py  # Terminal-based prediction
│   ├── predict_fuel_advanced.py     # Advanced prediction script
│   │
│   ├── model/
│   │   ├── fuel_model.pkl     # Trained machine learning model
│   │   └── selector.pkl       # Feature selector (Lasso)
│   │
│   └── assets/
│       └── logo.png           # ESLSE application logo
│
├── frontend/
│   ├── index.html             # Optional frontend HTML page
│   ├── script.js              # JavaScript logic
│   └── style.css              # CSS styling
│
└── .gitignore                 # Git ignore rules

```


# Web Application (Streamlit)

The Streamlit app allows ESLSE staff to:

Enter trip details (distance, speed, load, etc.)

Receive predicted fuel consumption instantly

Use a non-technical, user-friendly interface without programming knowledge

##  How to Run the Project

### Step 1: Install Required Libraries
pip install -r requirements.txt
Step 2: Train the Model
python backend/train_model.py
Step 3: Run the Web Application
streamlit run backend/app.py

Open your browser and visit:
http://localhost:8502

###  Technologies Used
Python 3
NumPy
Pandas
Scikit-learn
Joblib
Streamlit

### Future Enhancements for ESLSE
-Integrate real-time GPS and telematics data for dynamic predictions

-Include driver behavior analysis to identify fuel-inefficient driving patterns

-Enable fleet-level optimization and anomaly detection

-Maintain prediction history for reporting and auditing

-Deploy on cloud platforms or Docker for enterprise scalability


## Dependencies Note
Indirect dependencies (such as Werkzeug, Jinja2, and others) are automatically installed by 'pip' when installing the main libraries.

The 'requirements.txt' file intentionally lists **only direct dependencies** used by the application to keep the project clean, portable, and easy to maintain.
