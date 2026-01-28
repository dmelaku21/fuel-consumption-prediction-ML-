import joblib
import numpy as np
import pandas as pd

# Load trained model, scaler, selector
model = joblib.load("model/fuel_model.pkl")
scaler = joblib.load("model/scaler.pkl")
selector = joblib.load("model/selector.pkl")

print("=== ESLSE Fuel Consumption Predictor ===\n")

# Realistic ranges based on training
RANGES = {
    "distance_km": (300, 650),
    "avg_speed_kmh": (40, 75),
    "vehicle_load_kg": (9000, 22000),
    "engine_size": (2.0, 2.5),
    "vehicle_age": (40, 50)
}

trip_history = []

def get_input(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(f"{prompt} ({min_val}-{max_val}): "))
            if not (min_val <= value <= max_val):
                print(f"❌ Value out of realistic range ({min_val}-{max_val}). Try again.")
                continue
            return value
        except ValueError:
            print("❌ Invalid input. Please enter a numeric value.")

while True:
    print("\nEnter trip details:")
    distance_km = get_input("Distance (km)", *RANGES["distance_km"])
    avg_speed_kmh = get_input("Average speed (km/h)", *RANGES["avg_speed_kmh"])
    vehicle_load_kg = get_input("Vehicle load (kg)", *RANGES["vehicle_load_kg"])
    engine_size = get_input("Engine capacity (liters)", *RANGES["engine_size"])
    vehicle_age = get_input("Vehicle age (years)", *RANGES["vehicle_age"])

    # Derived features (must match training)
    distance_km_sq = distance_km ** 2
    avg_speed_kmh_sq = avg_speed_kmh ** 2
    load_per_engine = vehicle_load_kg / engine_size
    log_distance = np.log1p(distance_km)

    # Correct column names for scaler
    features = pd.DataFrame([{
        "distance_km": distance_km,
        "avg_speed_kmh": avg_speed_kmh,
        "vehicle_load_kg": vehicle_load_kg,
        "engine_size": engine_size,
        "vehicle_age": vehicle_age,
        "distance_km_sq": distance_km_sq,
        "avg_speed_kmh_sq": avg_speed_kmh_sq,
        "load_per_engine": load_per_engine,
        "log_distance": log_distance
    }])

    # Scale → Select → Predict
    features_scaled = scaler.transform(features)
    features_selected = selector.transform(features_scaled)
    predicted_fuel = float(model.predict(features_selected)[0])
    predicted_fuel = max(predicted_fuel, 0.0)  # No negative fuel

    print(f"✅ Predicted fuel consumption: {predicted_fuel:.2f} liters")

    trip_history.append({
        "Distance_km": distance_km,
        "Avg_speed_kmh": avg_speed_kmh,
        "Load_kg": vehicle_load_kg,
        "Engine_L": engine_size,
        "Age_yrs": vehicle_age,
        "Fuel_L": predicted_fuel
    })

    again = input("\nPredict another trip? (y/n): ").lower()
    if again != "y":
        break

# Summary
print("\n=== Summary of All Trips ===")
df = pd.DataFrame(trip_history)
print(df.to_string(index=False))
print(f"\nAverage fuel consumption: {df['Fuel_L'].mean():.2f} L")
print("✅ All trips processed successfully")
