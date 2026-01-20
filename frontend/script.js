function predict() {
  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      distance_km: Number(document.getElementById("distance").value),
      avg_speed_kmh: Number(document.getElementById("speed").value),
      vehicle_load_kg: Number(document.getElementById("load").value),
      engine_capacity_l: Number(document.getElementById("engine").value),
      vehicle_age_years: Number(document.getElementById("age").value)
    })
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById("result").innerText =
      "Predicted Fuel Consumption: " +
      data.predicted_fuel_consumption_liters.toFixed(2) + " liters";
  })
  .catch(error => {
    document.getElementById("result").innerText =
      "Error: Unable to get prediction";
    console.error(error);
  });
}
