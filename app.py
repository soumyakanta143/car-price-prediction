import streamlit as st
import pandas as pd
import pickle

# =============================
# Load trained model & encoders
# =============================
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("car_price_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load dataset (for dropdowns)
df = pd.read_csv("Car_Price_Cleaned.csv")

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.title("🚗 Car Price Prediction App")
st.write("Enter the details of your car to predict its price:")

# =============================
# Layout: 3 columns for inputs
# =============================
col1, col2, col3 = st.columns(3)

# Dropdown options from dataset
makes = df["Make"].unique().tolist()
models = df["Model"].unique().tolist()
fuel_types = df["Fuel_Type"].unique().tolist()
transmissions = df["Transmission"].unique().tolist()

with col1:
    make = st.selectbox("Make", makes)
    engine_size = st.number_input("Engine Size (Litres)", min_value=0.5, max_value=6.0, step=0.1)
    car_age = st.number_input("Car Age (years)", min_value=0, max_value=30, step=1)

with col2:
    model_name = st.selectbox("Model", models)
    mileage = st.number_input("Mileage (in kms)", min_value=0, max_value=300000, step=1000)

with col3:
    fuel_type = st.selectbox("Fuel Type", fuel_types)
    transmission = st.selectbox("Transmission", transmissions)

# =============================
# Feature Engineering
# =============================
mileage_per_year = mileage / car_age if car_age > 0 else mileage

# Encode categorical inputs
make_encoded = encoders["Make"].transform([make])[0]
model_encoded = encoders["Model"].transform([model_name])[0]
fuel_encoded = encoders["Fuel_Type"].transform([fuel_type])[0]
trans_encoded = encoders["Transmission"].transform([transmission])[0]

# =============================
# Build input DataFrame (MUST match training)
# =============================
input_data = pd.DataFrame([{
    "Car_Age": car_age,
    "Mileage": mileage,
    "Mileage_per_Year": mileage_per_year,
    "Fuel_Type": fuel_encoded,
    "Transmission": trans_encoded,
    "Engine_Size": engine_size,
    "Make": make_encoded,
    "Model": model_encoded
}])[["Car_Age", "Mileage", "Mileage_per_Year",
    "Fuel_Type", "Transmission", "Engine_Size",
    "Make", "Model"]]

# =============================
# Prediction
# =============================

# Auto prediction whenever inputs change
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.markdown("### 💰 Predicted Price:")
    st.success(f"₹ {prediction:,.2f}")
else:
    st.empty()


