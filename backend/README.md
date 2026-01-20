# Fuel Consumption Prediction System 🚗⛽

## 1. Project Overview
The Fuel Consumption Prediction System is a machine learning–based application designed to estimate vehicle fuel consumption using trip and vehicle-related parameters.  
The system helps analyze fuel usage patterns and provides accurate predictions based on historical data.

This project includes:
- Data preprocessing and feature selection
- Machine learning model training and evaluation
- Model persistence using Joblib
- An interactive web-based interface built with Streamlit

---

## 2. Problem Statement
Fuel consumption is influenced by multiple factors such as travel distance, speed, vehicle load, engine capacity, and vehicle age.  
Manually estimating fuel usage can be inaccurate and inefficient.

This project aims to:
- Build a predictive model to estimate fuel consumption
- Reduce manual calculation errors
- Provide a user-friendly interface for quick predictions

---

## 3. Dataset Description
The dataset used in this project contains historical trip and vehicle information.

### Input Features:
- **Distance (km)** – Total distance traveled
- **Average Speed (km/h)** – Mean speed during the trip
- **Vehicle Load (kg)** – Load carried by the vehicle
- **Engine Capacity (liters)** – Engine size
- **Vehicle Age (years)** – Age of the vehicle

### Target Variable:
- **Fuel Consumption (liters)** – Total fuel used for the trip

---

## 4. Machine Learning Approach

### 4.1 Feature Selection
- Feature selection is performed using **Lasso Regression** combined with `SelectFromModel`
- This helps reduce irrelevant features and improves model performance

### 4.2 Model Training
- Algorithm used: **Linear Regression**
- Data split:
  - 80% Training
  - 20% Testing

### 4.3 Model Evaluation
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

## 7. Project Structure
FUEL_CONSUMPTION_PREDICTION/
│
├── README.md                         # Project documentation
├── requirements.txt                  # Required Python libraries
│
├── data/
│   └── fuel_data.csv                 # Dataset used for training and evaluation
│
├── backend/
│   ├── train_model.py                # Script to train and evaluate the ML model
│   ├── app.py                        # Main Streamlit web application
│   ├── app_advanced.py               # Extended/advanced application logic
│   ├── predict_fuel_interactive.py   # Terminal-based interactive prediction
│   ├── predict_fuel_advanced.py      # Advanced prediction script
│   ├── .gitignore                    # Git ignore rules
│   │
│   ├── model/
│   │   ├── fuel_model.pkl           # Trained machine learning model
│   │   └── selector.pkl             # Feature selector used during training
│   │
│   └── venv/                        # Virtual environment (ignored in Git)
│
├── frontend/
│   ├── index.html                  # Frontend HTML page
│   ├── script.js                   # JavaScript logic
│   └── style.css                   # CSS styling



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

### Future Improvements
Support multiple trips and summary reports
Add input validation and error handling
Store prediction history
Deploy the application online (Streamlit Cloud / Docker)


## Dependencies Note
Indirect dependencies (such as Werkzeug, Jinja2, and others) are automatically installed by 'pip' when installing the main libraries.

The 'requirements.txt' file intentionally lists **only direct dependencies** used by the application to keep the project clean, portable, and easy to maintain.
