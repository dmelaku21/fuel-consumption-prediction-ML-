# ================================================
# ESLSE Fuel Consumption Prediction System
# Streamlit App - Professional Login + Dashboard
# ================================================

# --------- IMPORTS ---------
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px
from PIL import Image
from auth import authenticate  # Your auth module

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="🚛 ESLSE Fuel Consumption Predictor",
    page_icon="🚛",
    layout="wide"
)

# --------- BASE DIRECTORY ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------- SESSION STATE INIT ---------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "trips" not in st.session_state:
    st.session_state.trips = []

# --------- APP BRANDING ---------
logo_path = os.path.join(BASE_DIR, "..", "assets", "logo.png")
logo = Image.open(logo_path) if os.path.exists(logo_path) else None
APP_NAME = "ESLSE Fuel Predictor"

# --------- MODEL LOADING ---------
MODEL_PATH = os.path.join(BASE_DIR, "model", "fuel_model.pkl")
SELECTOR_PATH = os.path.join(BASE_DIR, "model", "selector.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

if not all(os.path.exists(p) for p in [MODEL_PATH, SELECTOR_PATH, SCALER_PATH]):
    st.error("❌ Model, selector, or scaler file not found.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, selector, scaler

model, selector, scaler = load_artifacts()

# --------- LOGIN PAGE ---------
def show_login():
    # Centered title
    st.markdown("<h2 style='text-align:center'>🔐 Secure Login</h2>", unsafe_allow_html=True)

    # Centered layout for login card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo + App Name
        col_logo, col_text = st.columns([1, 3])
        if logo:
            col_logo.image(logo, width=100)
        col_text.markdown(f"<h3 style='margin-top:25px'>{APP_NAME}</h3>", unsafe_allow_html=True)

        # Login card
        st.markdown(
            """
            <div style='padding:25px; border-radius:15px; background-color:#f5f7fa;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top:15px'>
            </div>
            """, unsafe_allow_html=True
        )

        # Input fields
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", type="primary"):
            authenticated, role = authenticate(username, password)
            if authenticated:
                st.session_state.authenticated = True
                st.session_state.user_role = role
                st.success(f"Welcome {username} ({role})")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

# --------- DASHBOARD ---------
def show_dashboard():
    # Sidebar
    st.sidebar.success(f"Logged in as: {st.session_state.user_role}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.trips = []
        st.rerun()

    # Header
    st.header("🚗 Fuel Consumption Prediction Dashboard")
    st.markdown(
        f"""
        ### 👋 Welcome to ESLSE Fuel Analytics  
        **Role:** {st.session_state.user_role}  
        Use the sidebar to enter trip details and predict fuel consumption.
        """
    )

    # --------- INPUT FORM ---------
    with st.sidebar.form("trip_form"):
        st.subheader("📝 Trip Details")
        distance = st.number_input("Distance (km)", 1.0, 100000.0, 100.0)
        avg_speed = st.number_input("Average Speed (km/h)", 1.0, 200.0, 60.0)
        load = st.number_input("Vehicle Load (kg)", 0.0, 50000.0, 500.0)
        engine_size = st.number_input("Engine Capacity (Liters)", 0.1, 10.0, 1.8)
        vehicle_age = st.number_input("Vehicle Age (Years)", 0.0, 50.0, 5.0)
        submit = st.form_submit_button("Predict Fuel Consumption")

    # --------- PREDICTION ---------
    if submit:
        try:
            # Feature engineering
            distance_km_sq = distance ** 2
            avg_speed_kmh_sq = avg_speed ** 2
            load_per_engine = load / engine_size
            log_distance = np.log1p(distance)

            features = np.array([[
                distance, avg_speed, load, engine_size, vehicle_age,
                distance_km_sq, avg_speed_kmh_sq, load_per_engine, log_distance
            ]])

            features_scaled = scaler.transform(features)
            features_selected = selector.transform(features_scaled)
            raw_prediction = float(model.predict(features_selected)[0])
            prediction = round(max(raw_prediction, 0.0), 2)

            # Save trip
            st.session_state.trips.append({
                "Distance (km)": distance,
                "Avg Speed (km/h)": avg_speed,
                "Load (kg)": load,
                "Engine (L)": engine_size,
                "Age (yrs)": vehicle_age,
                "Predicted Fuel (L)": prediction
            })

            st.success(f"✅ Predicted Fuel: **{prediction} liters**")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    # --------- EMPTY STATE ---------
    if not st.session_state.trips:
        st.info("👈 Enter trip details in the sidebar and click **Predict Fuel Consumption**.")
        return

    # --------- SUMMARY ---------
    st.markdown("---")
    st.subheader("📊 Trip Prediction Summary")
    df = pd.DataFrame(st.session_state.trips)
    st.dataframe(df, use_container_width=True)
    st.metric("Average Fuel Consumption", f"{df['Predicted Fuel (L)'].mean():.2f} L")

    # --------- CHARTS ---------
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.line(df, x="Distance (km)", y="Predicted Fuel (L)", markers=True, title="Fuel vs Distance"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            px.bar(df, x="Avg Speed (km/h)", y="Predicted Fuel (L)", title="Fuel vs Speed"),
            use_container_width=True
        )

    # Clear all trips
    if st.button("🗑️ Clear All Trips"):
        st.session_state.trips = []
        st.rerun()

# --------- MAIN ---------
if st.session_state.authenticated:
    show_dashboard()
else:
    show_login()
