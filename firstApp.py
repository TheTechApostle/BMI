# bmi_ml_app.py
import streamlit as st
import joblib
import numpy as np

st.title("🧠 ML BMI Category Predictor")

# Load your trained model
model = joblib.load("bmi_model.pkl")

# Input
weight = st.number_input("Weight (kg)", min_value=1.0)
height = st.number_input("Height (m)", min_value=0.1)

if st.button("Predict Category"):
    input_data = np.array([[weight, height]])
    prediction = model.predict(input_data)[0]

    bmi = weight / (height ** 2)

    st.write(f"Your BMI: **{bmi:.2f}**")
    st.success(f"Predicted Category: **{prediction}**")
