# Fuel Consumption Prediction System for ESLSE 🚛⛽

Overview
The Fuel Consumption Prediction System is a machine learning–based tool designed to help Ethiopian Shipping and Logistics Services Enterprise (ESLSE) accurately estimate fuel usage for each vehicle trip. By analyzing operational data such as distance, speed, vehicle load, engine capacity, and vehicle age, the system replaces manual estimation methods with a data-driven, predictive model, improving cost efficiency and operational planning.

This project demonstrates how predictive analytics can enhance fuel management, route planning, and fleet optimization in large-scale logistics operations.

---
 ## Business Motivation for ESLSE

Fuel is one of the largest operational expenses for ESLSE. Current estimation methods rely on:

- Fixed fuel-per-kilometer averages

- Historical records and spreadsheets

- Supervisor experience and judgment

## Challenges of current methods:

- Inaccurate fuel budgeting

- Overestimation or underestimation of fuel needs

- Limited identification of inefficient vehicles

- Low transparency and accountability

The predictive system addresses these challenges by estimating fuel consumption at the trip level, enabling proactive decision-making and cost control.

### Key Features

- Data-driven fuel prediction: Uses machine learning to estimate fuel consumption for every trip

-Feature selection: Lasso Regression identifies the most influential factors affecting fuel use

-High interpretability: Linear Regression allows ESLSE managers to understand how each factor impacts consumption

-Web-based interface: Streamlit app enables non-technical users to interact with the system

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

## 5. System Architecture
The system consists of three main components:
1. **Training Module** – Trains and saves the machine learning model
2. **Prediction Module** – Loads the trained model and predicts fuel consumption
3. **Web Interface** – Streamlit-based UI for user interaction

---

## 6. Web Application (Streamlit)
The Streamlit web application allows users to:
- Enter vehicle and trip details
- Instantly receive predicted fuel consumption
- Use the system without modifying any code

---
## Benefits to ESLSE:

-Accurate fuel budgeting

-Route and vehicle efficiency analysis

-Reduced fuel wastage and emissions

-Data-driven decision-making support

## 7. Project Structure
## 7. Project Structure

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
│   ├── app.py                 # Main Streamlit web application
│   ├── app_advanced.py        # Extended / advanced application logic
│   ├── predict_fuel_interactive.py  # Terminal-based interactive prediction
│   ├── predict_fuel_advanced.py     # Advanced prediction script
│   │
│   ├── model/
│   │   ├── fuel_model.pkl     # Trained machine learning model
│   │   └── selector.pkl       # Feature selector used during training
│   │
│   └── venv/                  # Virtual environment (ignored in Git)
│
├── frontend/
│   ├── index.html             # Frontend HTML page
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

## 8. How to Run the Project

### Step 1: Install Required Libraries
pip install -r requirements.txt
Step 2: Train the Model
python backend/train_model.py
Step 3: Run the Web Application
streamlit run backend/app.py

Open your browser and visit:
http://localhost:8502

### 9. Technologies Used
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
