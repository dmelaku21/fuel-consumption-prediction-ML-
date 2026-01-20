import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# Load model and selector
model = joblib.load("model/fuel_model.pkl")
selector = joblib.load("model/selector.pkl")

# Page config
st.set_page_config(page_title="🚗 Fuel Consumption Predictor", layout="wide", page_icon="🚗")

# Title
st.title("🚗 Fuel Consumption Predictor")
st.markdown("Enter your trip and vehicle details below to predict **fuel consumption**.")

# Initialize session state for trips
if "trips" not in st.session_state:
    st.session_state.trips = []

# Sidebar form for input
with st.sidebar.form(key="trip_form"):
    st.header("Trip Details")
    distance = st.number_input("Distance (km)", min_value=0.0, value=100.0, step=1.0, help="Enter the total distance of the trip in km")
    avg_speed = st.number_input("Average Speed (km/h)", min_value=0.0, value=60.0, step=1.0, help="Enter the average speed during the trip")
    load = st.number_input("Vehicle Load (kg)", min_value=0.0, value=500.0, step=10.0, help="Enter the vehicle load in kilograms")
    engine_capacity = st.number_input("Engine Capacity (liters)", min_value=0.1, value=1.8, step=0.1, help="Enter engine capacity in liters")
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0.0, value=5.0, step=1.0, help="Enter the age of the vehicle in years")
    submit = st.form_submit_button("Predict Fuel Consumption")

# Predict and store
if submit:
    if distance <= 0 or avg_speed <= 0 or engine_capacity <= 0 or load < 0 or vehicle_age < 0:
        st.error("❌ Enter valid positive values for all fields.")
    else:
        features = np.array([[distance, avg_speed, load, engine_capacity, vehicle_age]])
        selected_features = selector.transform(features)
        predicted_fuel = float(model.predict(selected_features)[0])
        predicted_fuel_rounded = round(predicted_fuel, 2)

        st.session_state.trips.append({
            "Distance (km)": distance,
            "Avg Speed (km/h)": avg_speed,
            "Load (kg)": load,
            "Engine (L)": engine_capacity,
            "Age (yrs)": vehicle_age,
            "Predicted Fuel (L)": predicted_fuel_rounded
        })

        # Display latest prediction in a card-style column
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("### ✅ Latest Trip Prediction")
            st.success(f"**{predicted_fuel_rounded} L**")
        with col2:
            st.markdown("#### Trip Details")
            st.write(f"- Distance: {distance} km")
            st.write(f"- Avg Speed: {avg_speed} km/h")
            st.write(f"- Load: {load} kg")
            st.write(f"- Engine: {engine_capacity} L")
            st.write(f"- Vehicle Age: {vehicle_age} yrs")

# Show trips summary
if st.session_state.trips:
    st.markdown("---")
    st.subheader("📊 Summary of All Trips")

    df_trips = pd.DataFrame(st.session_state.trips)
    
    # Highlight max fuel consumption in red
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #ffcccc' if v else '' for v in is_max]

    st.dataframe(df_trips.style.apply(highlight_max, subset=["Predicted Fuel (L)"]), height=300)

    # Show average fuel
    avg_fuel = df_trips["Predicted Fuel (L)"].mean()
    st.metric("💡 Average Fuel Consumption", f"{avg_fuel:.2f} L")

    # Charts
    st.subheader("📈 Fuel Consumption Charts")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(df_trips, x="Distance (km)", y="Predicted Fuel (L)", markers=True, 
                       title="Fuel vs Distance", template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df_trips, x="Avg Speed (km/h)", y="Predicted Fuel (L)", 
                      title="Fuel vs Avg Speed", color="Predicted Fuel (L)", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    # Clear trips button
    if st.button("Clear All Trips"):
        st.session_state.trips = []
        st.success("✅ All trips cleared.")


# run by using; streamlit run app_advanced.py

