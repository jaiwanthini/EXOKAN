"""
streamlit_app.py

Interactive demo: enter stellar/transit measurements, get a KOI disposition
prediction. Run with:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import numpy as np
import torch
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from kan import KAN  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

st.set_page_config(page_title="Exoplanet Candidate Classifier", page_icon="🪐")
st.title("🪐 Exoplanet Candidate Classifier")
st.caption("Interpretable classification of Kepler Objects of Interest using a Kolmogorov-Arnold Network")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    config = np.load(MODELS_DIR / "kan_config.npy", allow_pickle=True).item()
    model = KAN(width=config["width"], grid=config["grid"], k=config["k"], seed=42)
    model.load_state_dict(torch.load(MODELS_DIR / "kan_model.pt"))
    model.eval()
    return model, scaler, label_encoder


model, scaler, label_encoder = load_artifacts()

st.subheader("Enter measurements")
col1, col2 = st.columns(2)

with col1:
    koi_period = st.number_input("Orbital period (days)", value=10.0)
    koi_duration = st.number_input("Transit duration (hours)", value=3.0)
    koi_depth = st.number_input("Transit depth (ppm)", value=500.0)
    koi_prad = st.number_input("Planetary radius (Earth radii)", value=2.0)
    koi_teq = st.number_input("Equilibrium temperature (K)", value=800.0)
    koi_insol = st.number_input("Insolation flux (Earth flux)", value=50.0)

with col2:
    koi_model_snr = st.number_input("Transit signal-to-noise ratio", value=15.0)
    koi_steff = st.number_input("Stellar effective temperature (K)", value=5700.0)
    koi_slogg = st.number_input("Stellar surface gravity (log g)", value=4.4)
    koi_srad = st.number_input("Stellar radius (solar radii)", value=1.0)
    koi_impact = st.number_input("Impact parameter", value=0.5)

if st.button("Classify"):
    features = np.array([[
        koi_period, koi_duration, koi_depth, koi_prad, koi_teq,
        koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_impact
    ]])
    features_scaled = scaler.transform(features)

    with torch.no_grad():
        logits = model(torch.tensor(features_scaled, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))

    predicted_label = label_encoder.classes_[pred_idx]

    st.subheader("Prediction")
    st.success(f"**{predicted_label}**")

    st.subheader("Class probabilities")
    for cls, p in zip(label_encoder.classes_, probs):
        st.write(f"{cls}: {p:.2%}")
        st.progress(float(p))

    st.info(
        "Note: run `src/interpret.py` to generate the KAN activation function "
        "plot separately — it shows which learned functions drive predictions "
        "across the whole dataset, not just this one input."
    )
