import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os
import warnings

# Suppress specific sklearn warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# Ensure 'model' folder exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/fuel_data.csv")

# Features and target
X = df[
    [
        "distance_km",
        "avg_speed_kmh",
        "vehicle_load_kg",
        "engine_capacity_l",
        "vehicle_age_years"
    ]
]

y = df["fuel_consumption_liters"]

# -------------------------------
# Feature Selection using Lasso
# -------------------------------
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)

selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# -------------------------------
# Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------------
# Save Model & Selector
# -------------------------------
joblib.dump(model, "model/fuel_model.pkl")
joblib.dump(selector, "model/selector.pkl")

print("✅ Model trained and saved successfully")


# run by using python train_model.py  