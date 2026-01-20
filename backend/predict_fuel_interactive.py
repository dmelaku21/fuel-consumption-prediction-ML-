import joblib
import numpy as np

# Load the trained model and selector
model = joblib.load("model/fuel_model.pkl")
selector = joblib.load("model/selector.pkl")

print("=== Fuel Consumption Predictor ===")
print("Please enter the following details:")

# Get user input
try:
    distance = float(input("Distance (km): "))
    avg_speed = float(input("Average speed (km/h): "))
    load = float(input("Vehicle load (kg): "))
    engine_capacity = float(input("Engine capacity (liters): "))
    vehicle_age = float(input("Vehicle age (years): "))
except ValueError:
    print("❌ Invalid input. Please enter numeric values only.")
    exit()

# Create input array
new_data = np.array([[distance, avg_speed, load, engine_capacity, vehicle_age]])

# Transform using selector
new_data_selected = selector.transform(new_data)

# Predict fuel consumption
predicted_fuel = model.predict(new_data_selected)

print(f"\n✅ Predicted fuel consumption: {predicted_fuel[0]:.2f} liters")


# run using;  python predict_fuel_interactive.py
