import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders dictionary
with open("car_price_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load dataset to fetch dropdown options
df = pd.read_csv("Car_Price_Cleaned.csv")

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")
st.write("Fill in the details of the car to estimate its price:")

# Input layout
col1, col2, col3 = st.columns(3)

with col1:
    make = st.selectbox("Make", df["Make"].unique().tolist())
    engine_size = st.number_input("Engine Size (Litres)", min_value=0.5, max_value=6.0, step=0.1)
    car_age = st.number_input("Car Age (years)", min_value=0, max_value=30, step=1)

with col2:
    model_name = st.selectbox("Model", df["Model"].unique().tolist())
    mileage = st.number_input("Mileage (kms)", min_value=0, max_value=300000, step=1000)

with col3:
    fuel_type = st.selectbox("Fuel Type", df["Fuel_Type"].unique().tolist())
    transmission = st.selectbox("Transmission", df["Transmission"].unique().tolist())

# Feature engineering
mileage_per_year = mileage / car_age if car_age > 0 else mileage

# Encode categorical features
make_encoded = encoders["Make"].transform([make])[0]
model_encoded = encoders["Model"].transform([model_name])[0]
fuel_encoded = encoders["Fuel_Type"].transform([fuel_type])[0]
trans_encoded = encoders["Transmission"].transform([transmission])[0]

# Match training feature order
input_data = pd.DataFrame([{
    "Make": make_encoded,
    "Model": model_encoded,
    "Engine_Size": engine_size,
    "Mileage": mileage,
    "Fuel_Type": fuel_encoded,
    "Transmission": trans_encoded,
    "Car_Age": car_age,
    "Mileage_per_Year": mileage_per_year
}])

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.markdown("### 💰 Predicted Price:")
    st.success(f"₹ {prediction:,.2f}")
