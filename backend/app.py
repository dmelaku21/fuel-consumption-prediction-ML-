from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and feature selector
model = joblib.load("model/fuel_model.pkl")
selector = joblib.load("model/selector.pkl")

# HTML template for the web form
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Fuel Consumption Prediction</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        input { padding: 5px; margin: 5px; width: 200px; }
        button { padding: 10px 20px; margin-top: 10px; }
        .result { margin-top: 20px; font-weight: bold; color: green; }
    </style>
</head>
<body>
    <h2>Fuel Consumption Prediction</h2>
    <form method="POST">
        <label>Distance (km):</label><br>
        <input type="number" step="any" name="distance_km" required><br>

        <label>Average Speed (km/h):</label><br>
        <input type="number" step="any" name="avg_speed_kmh" required><br>

        <label>Vehicle Load (kg):</label><br>
        <input type="number" step="any" name="vehicle_load_kg" required><br>

        <label>Engine Capacity (liters):</label><br>
        <input type="number" step="any" name="engine_capacity_l" required><br>

        <label>Vehicle Age (years):</label><br>
        <input type="number" step="any" name="vehicle_age_years" required><br>

        <button type="submit">Predict Fuel Consumption</button>
    </form>

    {% if prediction %}
        <div class="result">
            Predicted Fuel Consumption: {{ prediction }} liters
        </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Get input values from form
            distance_km = float(request.form["distance_km"])
            avg_speed_kmh = float(request.form["avg_speed_kmh"])
            vehicle_load_kg = float(request.form["vehicle_load_kg"])
            engine_capacity_l = float(request.form["engine_capacity_l"])
            vehicle_age_years = float(request.form["vehicle_age_years"])

            # Convert to array and apply selector
            features = np.array([
                distance_km, avg_speed_kmh, vehicle_load_kg,
                engine_capacity_l, vehicle_age_years
            ]).reshape(1, -1)
            selected_features = selector.transform(features)

            # Make prediction
            prediction_value = model.predict(selected_features)[0]
            prediction = round(float(prediction_value), 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(html_form, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

  #Run backend:python app.py 
  #Open browser:http://127.0.0.1:5000
#Backend is running ✅
