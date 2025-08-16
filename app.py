import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model & encoders
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("car_price_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="Car Price Predictor 🚗", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter the details of the car to predict its price")

# Input fields (update according to your dataset)
make = st.selectbox("Make", encoders["Make"].classes_)
model_name = st.selectbox("Model", encoders["Model"].classes_)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1)
mileage = st.number_input("Mileage", min_value=0, step=1000)
fuel_type = st.selectbox("Fuel Type", encoders["Fuel_Type"].classes_)
transmission = st.selectbox("Transmission", encoders["Transmission"].classes_)
car_age = st.number_input("Car Age (years)", min_value=0, step=1)
mileage_per_year = st.number_input("Mileage per Year", min_value=0, step=500)

# Prepare input row
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

# Encode categorical features
for col in ["Make", "Model", "Fuel_Type", "Transmission"]:
    input_data[col] = encoders[col].transform(input_data[col])

# Prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Car Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
