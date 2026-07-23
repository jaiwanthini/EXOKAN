"""
streamlit_app.py
 
Interactive demo: enter stellar/transit measurements, get a KOI disposition
prediction, with inline reasoning and the KAN interpretability plot embedded.
 
Run with:  streamlit run app/streamlit_app.py
(run from the project root, e.g. the EXOKAN folder - not from inside src/)
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
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
 
FEATURE_KEYS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_steff", "koi_slogg", "koi_srad", "koi_impact",
]
 
FEATURE_LABELS = {
    "koi_period": "Orbital period (days)",
    "koi_duration": "Transit duration (hours)",
    "koi_depth": "Transit depth (ppm)",
    "koi_prad": "Planetary radius (Earth radii)",
    "koi_teq": "Equilibrium temperature (K)",
    "koi_insol": "Insolation flux (Earth flux)",
    "koi_model_snr": "Transit signal-to-noise ratio",
    "koi_steff": "Stellar effective temperature (K)",
    "koi_slogg": "Stellar surface gravity (log g)",
    "koi_srad": "Stellar radius (solar radii)",
    "koi_impact": "Impact parameter",
}
 
# Real KOI rows pulled directly from the NASA Exoplanet Archive cumulative table,
# one per class - not synthetic values. Feature order matches FEATURE_KEYS.
EXAMPLES = {
    "Kepler-227 b (real CONFIRMED)": {
        "koi_period": 9.488035570, "koi_duration": 2.95750, "koi_depth": 615.8,
        "koi_prad": 2.26, "koi_teq": 793.0, "koi_insol": 93.59, "koi_model_snr": 35.80,
        "koi_steff": 5455.00, "koi_slogg": 4.467, "koi_srad": 0.9270, "koi_impact": 0.1460,
    },
    "K00753.01 (real CANDIDATE)": {
        "koi_period": 19.899139950, "koi_duration": 1.78220, "koi_depth": 10829.0,
        "koi_prad": 14.60, "koi_teq": 638.0, "koi_insol": 39.30, "koi_model_snr": 76.30,
        "koi_steff": 5853.00, "koi_slogg": 4.544, "koi_srad": 0.8680, "koi_impact": 0.9690,
    },
    "K00754.01 (real FALSE POSITIVE)": {
        "koi_period": 1.736952453, "koi_duration": 2.40641, "koi_depth": 8079.2,
        "koi_prad": 33.46, "koi_teq": 1395.0, "koi_insol": 891.96, "koi_model_snr": 505.60,
        "koi_steff": 5805.00, "koi_slogg": 4.564, "koi_srad": 0.7910, "koi_impact": 1.2760,
    },
}
 
# What SHAP found drives each class most - used to generate a plain-English
# explanation alongside the prediction (see interpret.py for the full analysis).
TOP_DRIVERS = {
    "CONFIRMED": ["koi_model_snr", "koi_prad"],
    "FALSE POSITIVE": ["koi_prad", "koi_period", "koi_depth"],
    "CANDIDATE": ["koi_model_snr", "koi_period"],
}
 
st.set_page_config(page_title="Exoplanet Candidate Classifier", page_icon="🪐", layout="wide")
 
for key in FEATURE_KEYS:
    if key not in st.session_state:
        st.session_state[key] = EXAMPLES["Kepler-227 b (real CONFIRMED)"][key]
 
 
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
 
st.title("🪐 Exoplanet Candidate Classifier")
st.caption(
    "Interpretable classification of Kepler Objects of Interest (KOIs) using a "
    "Kolmogorov-Arnold Network (KAN), trained on 9,500+ NASA Exoplanet Archive records."
)
 
with st.sidebar:
    st.header("Why is this interpretable?")
    st.write(
        "Unlike a standard neural network, a KAN learns an explicit, visualizable "
        "function on every connection instead of a fixed weight. The plot below is "
        "read directly from the trained model - no separate explanation step needed."
    )
    kan_plot_path = RESULTS_DIR / "kan_activation_functions.png"
    if kan_plot_path.exists():
        st.image(str(kan_plot_path), caption="Learned KAN activation functions", use_container_width=True)
    else:
        st.info("Run `src/interpret.py` first to generate this plot.")
 
st.subheader("Try an example, or enter your own measurements")
example_cols = st.columns(len(EXAMPLES))
for col, (label, values) in zip(example_cols, EXAMPLES.items()):
    if col.button(label):
        for k, v in values.items():
            st.session_state[k] = v
        st.rerun()
 
st.divider()
 
col1, col2 = st.columns(2)
half = len(FEATURE_KEYS) // 2
with col1:
    for key in FEATURE_KEYS[:half]:
        st.number_input(FEATURE_LABELS[key], key=key)
with col2:
    for key in FEATURE_KEYS[half:]:
        st.number_input(FEATURE_LABELS[key], key=key)
 
if st.button("Classify", type="primary"):
    features = np.array([[st.session_state[k] for k in FEATURE_KEYS]])
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
 
    st.subheader("Why this prediction?")
    drivers = TOP_DRIVERS.get(predicted_label, [])
    if drivers:
        driver_text = ", ".join(FEATURE_LABELS[d] for d in drivers)
        st.info(
            f"Based on SHAP analysis of the baseline model (see `results/` for the full "
            f"plots), **{predicted_label}** predictions are most strongly associated with: "
            f"**{driver_text}**. Check the sidebar for the KAN's own learned activation "
            f"functions, which independently show similar patterns for these features."
        )
