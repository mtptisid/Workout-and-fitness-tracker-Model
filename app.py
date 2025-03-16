import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model & preprocessors
model = pickle.load(open("workout_model.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
efficiency_encoder = pickle.load(open("efficiency_encoder.pkl", "rb"))

# Define expected features based on training
expected_features = scaler.feature_names_in_

# Define categorical & numerical columns
categorical_cols = ["Gender", "Workout Type", "Workout Intensity", "Mood Before Workout"]
numerical_cols = ["Age", "Height (cm)", "Weight (kg)", "Workout Duration (mins)", 
                  "Sleep Hours", "Water Intake (liters)", "Daily Calories Intake", 
                  "Resting Heart Rate (bpm)", "VO2 Max", "Body Fat (%)"]

# Streamlit UI
st.title("🏋️ Workout & Fitness Tracker")

# Collect user input safely
input_data = {}

for col in categorical_cols:
    if col in encoders:  # Ensure encoder exists for this column
        input_data[col] = st.selectbox(f"Select {col}", encoders[col].classes_)

for col in numerical_cols:
    input_data[col] = st.number_input(f"Enter {col}", min_value=0.0, format="%.2f")

# Convert input dictionary to DataFrame
input_data = pd.DataFrame([input_data])

# Ensure all expected features exist
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0  # Default value for missing columns

# Reorder columns to match training data
input_data = input_data[expected_features]

# Print expected features and input data columns for debugging
st.write("Expected Features:", expected_features)
st.write("Input Data Columns:", input_data.columns.tolist())

# Encode categorical features safely
for col in categorical_cols:
    if col in input_data.columns and col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

# Scale numerical features
input_data = scaler.transform(input_data)

# Make sure feature count matches the model
if input_data.shape[1] != len(expected_features):
    st.error(f"⚠️ Feature count mismatch: Model expects {len(expected_features)}, but received {input_data.shape[1]}")
    st.stop()

# Make prediction
prediction = model.predict(input_data)[0]
predicted_efficiency = efficiency_encoder.inverse_transform([prediction])[0]

# Display result
st.subheader(f"🔥 Predicted Workout Efficiency: {predicted_efficiency}")