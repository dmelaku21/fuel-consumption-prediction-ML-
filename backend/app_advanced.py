# =========================
# ESLSE Fuel Consumption Prediction System
# Advanced Streamlit App (Stable Version)
# =========================

# --------- IMPORTS ---------
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px
from PIL import Image
from auth import authenticate

# --------- PAGE CONFIG (MUST BE FIRST) ---------
st.set_page_config(
    page_title="🚛 ESLSE Fuel Consumption Predictor",
    page_icon="🚛",
    layout="wide"
)

# --------- BASE DIRECTORY ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------- SESSION STATE INIT ---------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "user_role" not in st.session_state:
    st.session_state["user_role"] = None

if "trips" not in st.session_state:
    st.session_state["trips"] = []

# --------- APP BRANDING (LOGO + TITLE) ---------
logo_path = os.path.join(BASE_DIR, "..", "assets", "logo.png")

col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=90)
with col2:
    st.title("🚛 ESLSE Fuel Consumption Predictor")

st.markdown("---")

# --------- MODEL LOADING ---------
MODEL_PATH = os.path.join(BASE_DIR, "model", "fuel_model.pkl")
SELECTOR_PATH = os.path.join(BASE_DIR, "model", "selector.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SELECTOR_PATH):
    st.error("❌ Model or selector file not found.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_model_and_selector(model_path, selector_path):
    model = joblib.load(model_path)
    selector = joblib.load(selector_path)
    return model, selector

model, selector = load_model_and_selector(MODEL_PATH, SELECTOR_PATH)

# --------- LOGIN PAGE ---------
def show_login():
    st.subheader("🔐 Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        authenticated, role = authenticate(username, password)

        if authenticated:
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = role
            st.success(f"Welcome {username} ({role})")
        else:
            st.error("❌ Invalid username or password")

# --------- DASHBOARD ---------
def show_dashboard():
    # Sidebar
    st.sidebar.success(f"Logged in as: {st.session_state['user_role']}")

    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["user_role"] = None
        st.session_state["trips"] = []
        st.stop()  # clean reload → shows login page

    # Main content
    st.header("🚗 Fuel Consumption Prediction Dashboard")
    st.write(
        "Enter trip and vehicle details below to predict **fuel consumption** "
        "for ESLSE fleet operations."
    )

    # --------- INPUT FORM ---------
    with st.sidebar.form("trip_form"):
        st.subheader("📝 Trip Details")

        distance = st.number_input("Distance (km)", 1.0, 100000.0, 100.0)
        avg_speed = st.number_input("Average Speed (km/h)", 1.0, 200.0, 60.0)
        load = st.number_input("Vehicle Load (kg)", 0.0, 50000.0, 500.0)
        engine_capacity = st.number_input("Engine Capacity (L)", 0.1, 10.0, 1.8)
        vehicle_age = st.number_input("Vehicle Age (years)", 0.0, 50.0, 5.0)

        submit = st.form_submit_button("Predict Fuel Consumption")

    # --------- PREDICTION ---------
    if submit:
        features = np.array([[distance, avg_speed, load, engine_capacity, vehicle_age]])
        selected_features = selector.transform(features)
        prediction = round(float(model.predict(selected_features)[0]), 2)

        st.session_state["trips"].append({
            "Distance (km)": distance,
            "Avg Speed (km/h)": avg_speed,
            "Load (kg)": load,
            "Engine (L)": engine_capacity,
            "Age (yrs)": vehicle_age,
            "Predicted Fuel (L)": prediction
        })

        col1, col2 = st.columns([2, 3])
        with col1:
            st.success(f"✅ Predicted Fuel: **{prediction} liters**")
        with col2:
            st.info("📌 Prediction recorded successfully")

    # --------- SUMMARY ---------
    if st.session_state["trips"]:
        st.markdown("---")
        st.subheader("📊 Trip Prediction Summary")

        df = pd.DataFrame(st.session_state["trips"])
        st.dataframe(df, use_container_width=True)

        st.metric(
            "Average Fuel Consumption",
            f"{df['Predicted Fuel (L)'].mean():.2f} L"
        )

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(
                df,
                x="Distance (km)",
                y="Predicted Fuel (L)",
                markers=True,
                title="Fuel vs Distance"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(
                df,
                x="Avg Speed (km/h)",
                y="Predicted Fuel (L)",
                title="Fuel vs Speed"
            )
            st.plotly_chart(fig2, use_container_width=True)

        if st.button("🗑️ Clear All Trips"):
            st.session_state["trips"] = []
            st.success("All trips cleared.")

# --------- MAIN ROUTER ---------
if st.session_state["authenticated"]:
    show_dashboard()
else:
    show_login()
