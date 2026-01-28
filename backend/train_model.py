# ================================================
# ESLSE Fuel Consumption Prediction - Safe Regression + Feature Selection
# ================================================

import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Suppress warnings
# -------------------------------
warnings.filterwarnings(action='ignore', category=UserWarning)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend folder
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "ESLSE_fuel_data_adapted.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Load Dataset Safely
# -------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", skipinitialspace=True)
df.columns = df.columns.str.strip()  # remove hidden spaces

print("✅ Dataset loaded successfully")
print("Columns:", df.columns.tolist())
print("First 5 rows:\n", df.head())

# -------------------------------
# Safe Feature Engineering
# -------------------------------

# Replace zeros in engine_size to prevent division by zero
df["engine_size"] = df["engine_size"].replace(0, 0.1)

# Derived features with clipping to safe range
df["distance_km_sq"] = df["distance_km"].clip(0, 10000)**2
df["avg_speed_kmh_sq"] = df["avg_speed_kmh"].clip(0, 200)**2
df["load_per_engine"] = (df["vehicle_load_kg"] / df["engine_size"]).clip(-1e6, 1e6)
df["log_distance"] = np.log1p(df["distance_km"].clip(lower=0.1))

# Replace Inf / -Inf and drop NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# -------------------------------
# Features and Target
# -------------------------------
feature_cols = [
    "distance_km",
    "avg_speed_kmh",
    "vehicle_load_kg",
    "engine_size",
    "vehicle_age",
    "distance_km_sq",
    "avg_speed_kmh_sq",
    "load_per_engine",
    "log_distance"
]

X = df[feature_cols].clip(-1e6, 1e6)  # safe for float64
y = df["fuel_consumption_liters"]

# Final safety check
assert np.all(np.isfinite(X.values)), "Error: X contains Inf or NaN!"

# -------------------------------
# Scale features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Feature Selection using LassoCV
# -------------------------------
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.05, 0.1, 0.5], cv=5, max_iter=10000)
lasso_cv.fit(X_scaled, y)
print(f"✅ Best Lasso alpha: {lasso_cv.alpha_:.4f}")

selector = SelectFromModel(lasso_cv, prefit=True)
X_selected = selector.transform(X_scaled)
selected_features = np.array(feature_cols)[selector.get_support()]
print("✅ Selected features:", selected_features)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Linear Regression
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation (Test Set):")
print(f"RMSE: {rmse:.2f} liters")
print(f"R² Score: {r2:.4f}")

# Cross-validation R²
cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
print(f"Mean R² (5-Fold CV): {cv_scores.mean():.4f}")

# -------------------------------
# Save Model, Selector & Scaler
# -------------------------------
model_path = os.path.join(MODEL_DIR, "fuel_model.pkl")
selector_path = os.path.join(MODEL_DIR, "selector.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(selector, selector_path)
joblib.dump(scaler, scaler_path)

print("\n✅ Model, selector, and scaler saved successfully:")
print(f"- {model_path}")
print(f"- {selector_path}")
print(f"- {scaler_path}")
