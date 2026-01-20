import joblib
import numpy as np

# Load the trained model and selector
model = joblib.load("model/fuel_model.pkl")
selector = joblib.load("model/selector.pkl")

print("=== Fuel Consumption Predictor (Advanced) ===")

# Store all trips for summary
trip_history = []

while True:
    try:
        distance = float(input("\nDistance (km): "))
        if distance <= 0: raise ValueError
        avg_speed = float(input("Average speed (km/h): "))
        if avg_speed <= 0: raise ValueError
        load = float(input("Vehicle load (kg): "))
        if load < 0: raise ValueError
        engine_capacity = float(input("Engine capacity (liters): "))
        if engine_capacity <= 0: raise ValueError
        vehicle_age = float(input("Vehicle age (years): "))
        if vehicle_age < 0: raise ValueError
    except ValueError:
        print("❌ Invalid input. Please enter positive numeric values.")
        continue

    # Create input array
    new_data = np.array([[distance, avg_speed, load, engine_capacity, vehicle_age]])
    # Transform using selector
    new_data_selected = selector.transform(new_data)
    # Predict fuel consumption
    predicted_fuel = model.predict(new_data_selected)[0]

    print(f"✅ Predicted fuel consumption: {predicted_fuel:.2f} liters")

    # Store trip in history
    trip_history.append({
        "Distance_km": distance,
        "Avg_speed_kmh": avg_speed,
        "Load_kg": load,
        "Engine_L": engine_capacity,
        "Age_yrs": vehicle_age,
        "Fuel_L": predicted_fuel
    })

    # Ask if user wants to enter another trip
    again = input("\nPredict another trip? (y/n): ").lower()
    if again != "y":
        break

# Display summary
print("\n=== Summary of All Trips ===")
for i, trip in enumerate(trip_history, 1):
    print(f"Trip {i}: Distance={trip['Distance_km']} km, Avg Speed={trip['Avg_speed_kmh']} km/h, "
          f"Load={trip['Load_kg']} kg, Engine={trip['Engine_L']} L, Age={trip['Age_yrs']} yrs, "
          f"Predicted Fuel={trip['Fuel_L']:.2f} L")
print("✅ All trips processed. Thank you!")



# Features added:
#Loops for multiple trips.

#Input validation (no negative or zero values where invalid).
#Keeps a summary of all trips and prints it at the end.