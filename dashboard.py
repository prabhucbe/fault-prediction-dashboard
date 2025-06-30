# -*- coding: utf-8 -*-
"""
Streamlit Real-Time Dashboard for Fault Prediction using XGBoost
"""

import streamlit as st
st.set_page_config(page_title="Fault Prediction Dashboard", layout="wide")

import pandas as pd
import numpy as np
import joblib

st.title("📈 Real-Time Fault Prediction Dashboard")

# Load trained model
try:
    model = joblib.load("xgb_model.pkl")  # Ensure this model path exists in the same folder
except FileNotFoundError:
    st.error("❌ Model file 'xgb_model.pkl' not found. Please make sure it exists in the working directory.")
    st.stop()

# Define sensors
sensor_cols = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
    'Pressure', 'Temperature', 'Thermocouple', 'Voltage',
    'Volume Flow RateRMS'
]

# Maintain a sliding buffer of last 60 values
data_buffer = {col: [] for col in sensor_cols}

# Input widget (sidebar)
st.sidebar.header("⚙️ Input Sensor Values")
sensor_input = {}
for col in sensor_cols:
    sensor_input[col] = st.sidebar.slider(col, min_value=0.0, max_value=100.0, step=0.1)


import time

if st.sidebar.button("Simulate 60 Readings"):
    for i in range(60):
        for col in sensor_cols:
            simulated_value = np.random.uniform(10, 90)  # Replace with fixed values if needed
            data_buffer[col].append(simulated_value)
            if len(data_buffer[col]) > 60:
                data_buffer[col].pop(0)
        time.sleep(0.01)  # Short delay to simulate time passing
    st.rerun()


# Update buffer
for col in sensor_cols:
    data_buffer[col].append(sensor_input[col])
    if len(data_buffer[col]) > 60:
        data_buffer[col].pop(0)

# Feature extraction from buffer
def extract_features():
    features = {}
    for col in sensor_cols:
        data = data_buffer[col]
        if len(data) < 60:
            return None
        features[f'{col}_mean_5'] = np.mean(data[-5:])
        features[f'{col}_std_5'] = np.std(data[-5:])
        features[f'{col}_lag_5'] = data[-5]
        features[f'{col}_mean_15'] = np.mean(data[-15:])
        features[f'{col}_std_15'] = np.std(data[-15:])
        features[f'{col}_lag_15'] = data[-15]
        features[f'{col}_mean_60'] = np.mean(data[-60:])
        features[f'{col}_std_60'] = np.std(data[-60:])
        features[f'{col}_lag_60'] = data[-60]
    return pd.DataFrame([features])

features_df = extract_features()

if features_df is not None:
    y_proba = model.predict_proba(features_df)[:, 1][0]
    y_pred = int(y_proba >= 0.4)  # Use optimal threshold if available

    st.subheader("📊 Model Prediction")
    st.metric("Fault Probability", f"{y_proba:.3f}")
    if y_pred == 0:
        st.success("✅ No Fault Detected")
    else:
        st.error("⚠️ Fault Predicted")

    with st.expander("🔍 Show Input Features"):
        st.dataframe(features_df.T.rename(columns={0: "Value"}))
else:
    st.warning("⏳ Waiting for more data to fill buffer (minimum 60 readings required)...")

st.markdown("---")
st.caption("© 2025 Real-Time XGBoost Fault Prediction Dashboard")
