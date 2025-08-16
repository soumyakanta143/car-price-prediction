import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Predictor")
st.caption("Predict car price using a trained ML model")

# --- Load trained model ---
with open("pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.subheader("Enter feature values")

# 👉 Update these fields according to your dataset columns
make = st.selectbox("Make", ["Honda", "Ford", "BMW", "Toyota", "Other"])
model_name = st.text_input("Model", "Civic")
engine_size = st.number_input("Engine Size (L)", value=1.5, step=0.1)
mileage = st.number_input("Mileage (km)", value=50000, step=1000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
car_age = st.number_input("Car Age (years)", value=5, step=1)
mileage_per_year = st.number_input("Mileage per Year", value=10000, step=500)

# Build input dataframe
input_data = pd.DataFrame([{
    "Make": make,
    "Model": model_name,
    "Engine_Size": engine_size,
    "Mileage": mileage,
    "Fuel_Type": fuel_type,
    "Transmission": transmission,
    "Car_Age": car_age,
    "Mileage_per_Year": mileage_per_year
}])

# --- Predict ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: **₹ {prediction:,.0f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
