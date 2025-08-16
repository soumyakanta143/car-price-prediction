import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# import training script


st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

MODELS_DIR = Path("models")
PIPE_PATH = MODELS_DIR / "pipeline.pkl"
META_PATH = MODELS_DIR / "metadata.json"

st.title("🚗 Car Price Predictor")
st.caption("Automatically trains if no model is found.")

# --- Train model if missing ---
if not PIPE_PATH.exists() or not META_PATH.exists():
    st.warning("No trained model found. Training a new model...")
    import argparse
    args = argparse.Namespace(
        data="data/your_dataset.csv",   # dataset location in your repo
        target="Price",                 # target column name
        task="auto",
        test_size=0.2,
        random_state=42
    )
    train.main(args)   # runs training and saves model/metadata

# --- Load pipeline + metadata ---
pipe = joblib.load(PIPE_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

st.sidebar.header("Model Info")
st.sidebar.write(f"**Task:** {meta['task']}")
st.sidebar.json(meta.get("metrics", {}))

st.subheader("Enter feature values")

inputs = {}
# numeric fields
for col in meta["numeric_columns"]:
    inputs[col] = st.number_input(f"{col}", value=0.0, step=1.0)

# categorical fields
for col in meta["categorical_columns"]:
    choices = meta["categorical_choices"].get(col, [])
    if choices:
        inputs[col] = st.selectbox(f"{col}", options=choices)
    else:
        inputs[col] = st.text_input(f"{col}", value="")

# Build single-row dataframe
row = {}
for col in meta["numeric_columns"]:
    row[col] = float(inputs[col]) if inputs[col] is not None else np.nan
for col in meta["categorical_columns"]:
    row[col] = str(inputs[col]) if inputs[col] is not None else ""

X_df = pd.DataFrame([row])

if st.button("Predict"):
    try:
        pred = pipe.predict(X_df)[0]
        if meta["task"] == "classification":
            st.success(f"Predicted class: **{pred}**")
        else:
            st.success(f"Predicted price: **{pred:,.0f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

