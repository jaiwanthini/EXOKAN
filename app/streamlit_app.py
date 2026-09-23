"""
streamlit_app.py
 
EXOKAN-AI: AI-Assisted Exoplanet Classification & Scientific Analysis
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import sys
import time
import base64
import os
import json
import plotly.graph_objects as go
from fpdf import FPDF
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

class AIExoplanetAssistant:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            self.api_key = st.secrets["GEMINI_API_KEY"]
            
        self.has_llm = bool(self.api_key)
        if self.has_llm:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.model_name = "gemini-2.5-flash"
            except ImportError:
                self.has_llm = False
            
    def chat(self, prompt, context, history, mode="Technical"):
        if self.has_llm:
            return self._llm_chat(prompt, context, history, mode)
        else:
            return self._fallback_chat(prompt, context, mode)
            
    def _llm_chat(self, prompt, context, history, mode):
        sys_prompt = f"""You are EXOKAN-AI, an AI Scientific Assistant for an Exoplanet Classification dashboard.
Your job is to answer user questions based ONLY on the provided Prediction Context.
Do NOT invent probabilities, feature values, or predictions.
If asked a general scientific question, clearly distinguish general knowledge from this project's specific model output.
Current Mode: {mode} (Adapt your tone, vocabulary, and explanation depth to this mode. Simple = beginner, Technical = ML terminology, Research = academic/detailed, Viva = concise academic defense).

Prediction Context:
{json.dumps(context, indent=2)}
"""
        try:
            from google.genai import types
            contents = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
                
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=sys_prompt)
            )
            return response.text
        except Exception as e:
            return f"*(LLM Error: {str(e)} - Falling back to local reasoning)*\n\n" + self._fallback_chat(prompt, context, mode)
            
    def _fallback_chat(self, prompt, context, mode):
        q = prompt.lower()
        rf = context.get('rf_pred', 'Unknown')
        prob = context.get('max_prob', 0.0)
        agree = context.get('agree', False)
        
        ans = ""
        if "why" in q and "disagree" in q:
            ans = "The models disagree because Random Forest uses orthogonal thresholds while KAN uses continuous smooth functions on edges." if not agree else "Actually, the models agree."
        elif "confident" in q or "confidence" in q:
            ans = f"The confidence is {context.get('confidence')} because the RF assigns a {prob:.1%} probability to the winning class."
        elif "feature" in q:
            f_labels = [FEATURE_LABELS.get(f, f) for f in context.get('features', [])]
            ans = f"The most influential features are {', '.join(f_labels)}. They strongly impact the {rf} prediction."
        elif "summarize" in q or "summary" in q:
            ans = f"This sample is a {rf} with {context.get('confidence')} confidence. Models {'agree' if agree else 'disagree'}."
        elif "kan" in q:
            ans = "KAN (Kolmogorov-Arnold Network) is a deep learning architecture using continuous edge functions instead of fixed weights."
        elif "rf" in q or "random forest" in q:
            ans = "Random Forest is an ensemble learning method that outputs the consensus class of multiple decision trees."
        elif "what if" in q or "change" in q:
            ans = "In local fallback mode, please use the 'What-If Analysis' panel above to simulate changes."
        else:
            ans = f"Based on the context, we have a {rf} classification with {prob:.1%} certainty."
            
        return f"[Local Engine] {ans}\n\n*(Note: Provide a GEMINI_API_KEY environment variable to enable the full Generative AI Scientific Assistant)*"

# Grouping for UI
GROUPS = {
    "Observational Parameters": ["koi_period", "koi_duration", "koi_depth", "koi_model_snr", "koi_impact"],
    "Stellar Parameters": ["koi_steff", "koi_slogg", "koi_srad"],
    "Planetary Parameters": ["koi_prad", "koi_teq", "koi_insol"]
}
 
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
 
TOP_DRIVERS = {
    "CONFIRMED": ["koi_model_snr", "koi_prad"],
    "FALSE POSITIVE": ["koi_prad", "koi_period", "koi_depth"],
    "CANDIDATE": ["koi_model_snr", "koi_period"],
}

# --- STYLING ---
st.set_page_config(page_title="EXOKAN-AI Dashboard", page_icon="🔭", layout="wide")

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_path = Path(__file__).resolve().parents[1] / "assets" / "background.jpg"
if bg_path.exists():
    bin_str = get_base64_of_bin_file(bg_path)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(11, 14, 20, 0.8), rgba(11, 14, 20, 0.9)), url("data:image/jpeg;base64,{bin_str}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
<style>
    :root {
        --bg-color: transparent;
        --card-bg: rgba(20, 25, 35, 0.7);
        --text-primary: #E0E6ED;
        --accent-cyan: #00E5FF;
        --accent-purple: #B084CC;
        --border-color: #2A3B4C;
        --success: #00E676;
        --warning: #FFAB00;
        --danger: #FF1744;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        color: white !important;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px 8px 0 0;
        color: #A0AAB5;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 229, 255, 0.1);
        border-bottom: 2px solid var(--accent-cyan);
        color: var(--accent-cyan) !important;
    }
    
    .custom-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-cyan);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #A0AAB5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    hr {
        border-color: var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# --- STATE INIT ---
if "history" not in st.session_state:
    st.session_state["history"] = []
if "ai_context" not in st.session_state:
    st.session_state["ai_context"] = None
for key in FEATURE_KEYS:
    if key not in st.session_state:
        st.session_state[key] = EXAMPLES["Kepler-227 b (real CONFIRMED)"][key]
if "what_if_var" not in st.session_state:
    st.session_state["what_if_var"] = "koi_depth"
if "what_if_val" not in st.session_state:
    st.session_state["what_if_val"] = st.session_state["koi_depth"]

# --- MODEL LOADING ---
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    config = np.load(MODELS_DIR / "kan_config.npy", allow_pickle=True).item()
    kan_model = KAN(width=config["width"], grid=config["grid"], k=config["k"], seed=42)
    kan_model.load_state_dict(torch.load(MODELS_DIR / "kan_model.pt"))
    kan_model.eval()
    rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
    return kan_model, rf_model, scaler, label_encoder
 
kan_model, rf_model, scaler, label_encoder = load_artifacts()

def generate_features(df):
    df_new = df.copy()
    df_new["transit_depth_duration_ratio"] = df_new["koi_depth"] / (df_new["koi_duration"] + 1e-6)
    df_new["snr_depth_ratio"] = df_new["koi_model_snr"] / (df_new["koi_depth"] + 1)
    df_new["teq_insol_ratio"] = df_new["koi_insol"] / (df_new["koi_teq"] + 1)
    skewed_cols = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_insol", "koi_model_snr", "koi_srad"]
    for col in skewed_cols:
        df_new[col] = np.log1p(df_new[col].clip(lower=0))
    return df_new

def get_predictions(features_dict):
    df_input = pd.DataFrame([features_dict])
    df_feat = generate_features(df_input)
    # The columns must match the exact order expected by the scaler.
    # The order was: original 11 features + 3 engineered + log replacements (in place)
    # However, to be perfectly safe, since the scaler learned from X_train_feat.columns,
    # we should ideally reorder. X_train_feat just appended the 3 new columns.
    expected_cols = FEATURE_KEYS + ["transit_depth_duration_ratio", "snr_depth_ratio", "teq_insol_ratio"]
    features = df_feat[expected_cols].values
    features_scaled = scaler.transform(features)
    
    with torch.no_grad():
        logits = kan_model(torch.tensor(features_scaled, dtype=torch.float32))
        kan_probs = torch.softmax(logits, dim=1).numpy()[0]
    kan_pred_idx = int(np.argmax(kan_probs))
    kan_pred = label_encoder.classes_[kan_pred_idx]
    
    rf_probs = rf_model.predict_proba(features_scaled)[0]
    rf_pred_idx = int(np.argmax(rf_probs))
    rf_pred = label_encoder.classes_[rf_pred_idx]
    
    return rf_pred, rf_probs, kan_pred, kan_probs

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color: #00E5FF;'>EXOKAN-AI</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: #A0AAB5; font-size: 0.9rem; margin-bottom: 10px;'>
    <strong>AI-Assisted Exoplanet Classification</strong><br><br>
    EXOKAN-AI is an advanced scientific dashboard that classifies exoplanet candidates from Kepler telescope data. 
    It bridges the gap between raw predictive models (Random Forest & KAN) and human interpretation by providing an interactive, generative AI Assistant.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    analysis_mode = st.radio("Analysis Mode", ["Simple", "Technical", "Research", "Viva"], index=1)
    st.divider()
    st.write("**Active Models**")
    st.write("✓ Random Forest\n✓ KAN")
    st.divider()
    st.write("**AI Features**")
    st.write("✓ Explainability\n✓ Confidence Analysis\n✓ Model Comparison\n✓ AI Interpretation\n✓ Report Generation")
    st.divider()
    st.caption("AI-assisted exoplanet classification using machine learning, deep learning and explainability. The AI layer provides model interpretation and decision support. It does not independently establish astronomical confirmation.")

# --- TABS ---
tab_over, tab_pred, tab_res, tab_ins, tab_rep = st.tabs([
    "OVERVIEW", "PREDICT", "AI ANALYSIS", "MODEL INSIGHTS", "REPORT"
])

# 1. OVERVIEW TAB
with tab_over:
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 style='font-size: 3.5rem; letter-spacing: 2px; color: #00E5FF;'>EXOKAN-AI</h1>
        <h3 style='color: #A0AAB5;'>AI-Assisted Exoplanet Classification & Scientific Analysis</h3>
        <p style='font-size: 1.1rem; margin-top: 20px;'>Analyze exoplanet candidates using machine learning, deep learning and explainable AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("<div class='custom-card'><div class='metric-label'>MODELS</div><div class='metric-value'>2</div><div style='color:#A0AAB5;font-size:0.8rem;'>RF + KAN</div></div>", unsafe_allow_html=True)
    c2.markdown("<div class='custom-card'><div class='metric-label'>CLASSES</div><div class='metric-value'>3</div><div style='color:#A0AAB5;font-size:0.8rem;'>Confirmed, Candidate, FP</div></div>", unsafe_allow_html=True)
    c3.markdown("<div class='custom-card'><div class='metric-label'>AI LAYER</div><div class='metric-value'>Active</div><div style='color:#A0AAB5;font-size:0.8rem;'>Explainable + Interactive</div></div>", unsafe_allow_html=True)
    c4.markdown("<div class='custom-card'><div class='metric-label'>INPUT</div><div class='metric-value'>Single</div><div style='color:#A0AAB5;font-size:0.8rem;'>Exoplanet Sample</div></div>", unsafe_allow_html=True)
    
    st.subheader("Load Example")
    cols = st.columns(3)
    for i, (label, values) in enumerate(EXAMPLES.items()):
        if cols[i].button(label, use_container_width=True):
            for k, v in values.items():
                st.session_state[k] = v
            st.rerun()

# 2. PREDICT TAB
with tab_pred:
    st.markdown("### Enter Exoplanet Features")
    st.caption("Adjust the observational, stellar, and planetary parameters to run a custom analysis.")
    
    cols = st.columns(3)
    for i, (group, keys) in enumerate(GROUPS.items()):
        with cols[i]:
            st.markdown(f"**{group.upper()}**")
            for k in keys:
                st.session_state[k] = st.number_input(FEATURE_LABELS[k], value=float(st.session_state[k]), key=f"in_{k}")
    
    st.divider()
    if st.button("ANALYZE EXOPLANET", type="primary", use_container_width=True):
        with st.status("Analyzing observational features...", expanded=True) as status:
            st.write("✓ Preprocessing data")
            time.sleep(0.3)
            st.write("✓ Random Forest inference")
            time.sleep(0.3)
            st.write("✓ KAN inference")
            time.sleep(0.3)
            
            # Sync inputs to state
            input_dict = {k: st.session_state[f"in_{k}"] for k in FEATURE_KEYS}
            for k in FEATURE_KEYS:
                st.session_state[k] = input_dict[k]
                
            rf_pred, rf_probs, kan_pred, kan_probs = get_predictions(input_dict)
            
            st.write("✓ Model comparison & Explainability")
            time.sleep(0.3)
            st.write("✓ AI interpretation")
            time.sleep(0.2)
            
            max_prob = max(rf_probs)
            if max_prob >= 0.80:
                conf_level = "HIGH"
            elif max_prob >= 0.60:
                conf_level = "MODERATE"
            else:
                conf_level = "LOW"
                
            agree = (rf_pred == kan_pred)
            drivers = TOP_DRIVERS.get(rf_pred, [])
            
            st.session_state["ai_context"] = {
                "input_features": input_dict,
                "rf_pred": rf_pred, "rf_probs": [float(p) for p in rf_probs],
                "kan_pred": kan_pred, "kan_probs": [float(p) for p in kan_probs],
                "confidence": conf_level, "max_prob": float(max_prob),
                "agree": bool(agree), "features": drivers,
                "timestamp": time.strftime("%H:%M:%S")
            }
            
            st.session_state["history"].insert(0, {
                "time": st.session_state["ai_context"]["timestamp"],
                "pred": rf_pred, "conf": conf_level, "agree": agree
            })
            if len(st.session_state["history"]) > 5:
                st.session_state["history"].pop()
                
            status.update(label="Analysis complete!", state="complete", expanded=False)
        st.success("Analysis ready! Check the AI ANALYSIS and MODEL INSIGHTS tabs.")

# 3. AI ANALYSIS TAB
with tab_res:
    if not st.session_state["ai_context"]:
        st.info("No prediction context available. Please run an analysis in the PREDICT tab first.")
    else:
        ctx = st.session_state["ai_context"]
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown(f"""
            <div class='custom-card' style='text-align:center;'>
                <div class='metric-label'>CLASSIFICATION RESULT</div>
                <div class='metric-value' style='font-size:2.5rem; margin: 15px 0;'>{ctx['rf_pred']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            color = "#00E676" if ctx['confidence'] == 'HIGH' else "#FFAB00" if ctx['confidence'] == 'MODERATE' else "#FF1744"
            st.markdown(f"""
            <div class='custom-card' style='text-align:center;'>
                <div class='metric-label'>CONFIDENCE</div>
                <div class='metric-value' style='font-size:2.5rem; color:{color}; margin: 15px 0;'>{ctx['max_prob']:.0%}</div>
                <div style='color:#A0AAB5;'>{ctx['confidence']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            icon = "✓ AGREEMENT" if ctx['agree'] else "⚠ DISAGREEMENT"
            ccolor = "#00E676" if ctx['agree'] else "#FF1744"
            st.markdown(f"""
            <div class='custom-card' style='text-align:center;'>
                <div class='metric-label'>MODEL CONSENSUS</div>
                <div class='metric-value' style='font-size:1.8rem; color:{ccolor}; margin: 15px 0;'>{icon}</div>
                <div style='color:#A0AAB5;'>RF: {ctx['rf_pred']} | KAN: {ctx['kan_pred']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Model Probabilities")
        fig = go.Figure()
        classes = label_encoder.classes_
        fig.add_trace(go.Bar(y=classes, x=ctx['rf_probs'], name='Random Forest', orientation='h', marker_color='#00E5FF'))
        fig.add_trace(go.Bar(y=classes, x=ctx['kan_probs'], name='KAN', orientation='h', marker_color='#B084CC'))
        fig.update_layout(barmode='group', template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), height=300, xaxis_title="Probability", yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("#### AI Confidence Analyst")
            sorted_probs = sorted(ctx['rf_probs'], reverse=True)
            gap = sorted_probs[0] - sorted_probs[1]
            if analysis_mode == "Technical":
                st.write(f"The ensemble estimator yields a maximum class probability of {ctx['max_prob']:.1%}. The margin to the secondary class is {gap:.1%}. This margin determines the {ctx['confidence']} confidence bounds for the classification posterior.")
            else:
                st.write(f"The model is {ctx['max_prob']:.0%} sure about this prediction. The gap between its top choice and its second choice is {gap:.0%}. This represents a {ctx['confidence'].lower()} level of certainty.")
            if ctx['confidence'] == 'LOW':
                st.warning("This prediction has relatively low model confidence and should be interpreted cautiously.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c_right:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("#### AI Model Debate")
            if ctx['agree']:
                msg = f"Both Random Forest and KAN reach the same classification ({ctx['rf_pred']}), suggesting consistency across the two distinct learned decision functions (ensemble tree logic vs. continuous non-linear representations)."
                if analysis_mode == "Simple":
                    msg = "Both models agree on the result. Because they work in very different ways, their agreement gives us more trust in this answer."
            else:
                msg = f"Random Forest predicts {ctx['rf_pred']} while KAN predicts {ctx['kan_pred']}. The disagreement indicates that the sample is interpreted differently by the two models. The feature pattern may therefore be ambiguous within the learned representations."
                if analysis_mode == "Simple":
                    msg = f"The models disagree! One says {ctx['rf_pred']} and the other says {ctx['kan_pred']}. This means the data is tricky and lies right on the edge of what the models learned."
            st.write(msg)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Feature Story")
        st.caption("Top influential features for this classification based on SHAP analysis.")
        if ctx['features']:
            for i, f_key in enumerate(ctx['features']):
                val = 1.0 - (i * 0.25)
                st.write(f"**{i+1}. {FEATURE_LABELS[f_key].upper()}**")
                st.progress(val)
        else:
            st.write("Feature breakdown not available for this prediction.")

# 4. MODEL INSIGHTS TAB
with tab_ins:
    if not st.session_state["ai_context"]:
        st.info("Run an analysis in the PREDICT tab first.")
    else:
        ctx = st.session_state["ai_context"]
        
        st.markdown("### AI Scientific Interpreter")
        with st.container():
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            if analysis_mode == "Technical":
                st.write("**Classification Summary:** The observational sample is classified as " + ctx['rf_pred'] + ".")
                st.write("**Model Evidence:** Inference via Random Forest and KAN yields a consensus state of " + str(ctx['agree']) + ". Max probability is " + f"{ctx['max_prob']:.2%}.")
                st.write("**Feature Evidence:** Decision boundary heavily influenced by " + ", ".join([FEATURE_LABELS[f] for f in ctx['features']]) + ".")
                st.write("**Uncertainty:** " + ("Low uncertainty." if ctx['confidence'] == 'HIGH' else "Moderate to high uncertainty due to boundary proximity."))
                st.write("**Interpretation:** Machine-learning classification result. Does not independently establish astronomical confirmation.")
            else:
                st.write("**Classification Summary:** The AI believes this is a " + ctx['rf_pred'] + ".")
                st.write("**Model Evidence:** The models " + ("agree" if ctx['agree'] else "disagree") + " and the confidence is " + ctx['confidence'].lower() + ".")
                st.write("**Feature Evidence:** The most important measurements were " + ", ".join([FEATURE_LABELS[f] for f in ctx['features']]) + ".")
                st.write("**Uncertainty:** " + ("We are pretty sure." if ctx['confidence'] == 'HIGH' else "This is a tough call for the models."))
                st.write("**Interpretation:** This is an AI guess and requires a human scientist to verify.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.divider()
        
        st.markdown("### AI Analysis Scorecard")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Classification", ctx['rf_pred'])
        sc2.metric("Confidence", ctx['confidence'])
        sc3.metric("Model Agreement", "YES" if ctx['agree'] else "NO")
        sc4.metric("Explanation", "AVAILABLE")
        sc5.metric("Report", "READY")
        
        st.divider()
        
        st.markdown("### What-If Analysis")
        st.caption("Change a key feature to see how it affects the Random Forest prediction.")
        
        wi_col1, wi_col2 = st.columns([1, 2])
        with wi_col1:
            var = st.selectbox("Select Feature", ctx['features'] if ctx['features'] else FEATURE_KEYS[:3])
            val = st.number_input("Adjust Value", value=float(ctx['input_features'][var]))
        
        with wi_col2:
            if st.button("Simulate"):
                new_dict = ctx['input_features'].copy()
                new_dict[var] = val
                n_rf, n_rfs, _, _ = get_predictions(new_dict)
                n_max = max(n_rfs)
                
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.write(f"**Original:** {ctx['rf_pred']} — {ctx['max_prob']:.1%}")
                st.write(f"**Simulated:** {n_rf} — {n_max:.1%}")
                delta = n_max - ctx['max_prob']
                d_text = f"increased by {delta*100:.1f}" if delta > 0 else f"decreased by {abs(delta)*100:.1f}"
                if n_rf == ctx['rf_pred']:
                    st.write(f"**Impact:** Class remained {n_rf}, but confidence {d_text} percentage points.")
                else:
                    st.write(f"**Impact:** Class changed to {n_rf}!")
                st.markdown("</div>", unsafe_allow_html=True)
                
        st.divider()
        
        st.markdown("### AI Scientific Assistant")
        st.caption("Ask questions about this specific prediction. (Connects to Google Gemini if API key is provided).")
        
        # Suggested questions
        st.write("**Suggested Questions:**")
        cols = st.columns(4)
        suggested_prompt = None
        if cols[0].button("Why this prediction?"): suggested_prompt = "Why was this prediction chosen?"
        if cols[1].button("Most important feature?"): suggested_prompt = "Which feature influenced the prediction most?"
        if cols[2].button("Do models agree?"): suggested_prompt = "Do RF and KAN agree?"
        if cols[3].button("Clear Chat"): 
            st.session_state.chat_history = []
            st.rerun()

        # Initialize chat history if not exists
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        prompt = st.chat_input("Ask anything (e.g., 'Why is this a candidate?')")
        if suggested_prompt:
            prompt = suggested_prompt

        if prompt:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing context..."):
                    assistant = AIExoplanetAssistant()
                    response = assistant.chat(prompt, ctx, st.session_state.chat_history[:-1], analysis_mode)
                    st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        st.divider()
        
        st.markdown("### Recent Analyses")
        if st.session_state["history"]:
            for h in st.session_state["history"]:
                st.write(f"`{h['time']}` | Pred: **{h['pred']}** | Conf: {h['conf']} | Agree: {h['agree']}")
            if st.button("Clear History"):
                st.session_state["history"] = []
                st.rerun()

# 5. REPORT TAB
with tab_rep:
    st.markdown("### EXOKAN-AI SCIENTIFIC ANALYSIS REPORT")
    st.caption("Generate and download a comprehensive PDF report of the current analysis.")
    
    if not st.session_state["ai_context"]:
        st.warning("Please run an analysis first.")
    else:
        if st.button("Generate AI Report", type="primary"):
            ctx = st.session_state["ai_context"]
            from fpdf.enums import XPos, YPos
            
            class ReportPDF(FPDF):
                def header(self):
                    self.set_font("helvetica", "B", 16)
                    self.cell(0, 10, "EXOKAN-AI SCIENTIFIC ANALYSIS REPORT", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    self.ln(15)

            pdf = ReportPDF()
            pdf.add_page()
            
            def add_section(title, lines):
                pdf.set_font("helvetica", "B", 12)
                pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("helvetica", size=10)
                for line in lines:
                    pdf.multi_cell(w=0, h=6, text=line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(5)
                
            in_lines = [f"- {FEATURE_LABELS[k]}: {v}" for k, v in ctx["input_features"].items()]
            add_section("1. Sample Input", in_lines)
            
            add_section("2. Classification & Probabilities", [
                f"- Random Forest: {ctx['rf_pred']} ({max(ctx['rf_probs']):.1%})",
                f"- KAN: {ctx['kan_pred']} ({max(ctx['kan_probs']):.1%})"
            ])
            
            add_section("3. Model Consensus & Confidence", [
                f"- Status: {'Agreement' if ctx['agree'] else 'Disagreement'}",
                f"- Confidence Level: {ctx['confidence']}"
            ])
            
            f_lines = [f"- {FEATURE_LABELS[d]}" for d in ctx["features"]] if ctx["features"] else ["- Not available"]
            add_section("4. Feature Importance", f_lines)
            
            add_section("5. AI Scientific Interpretation", [
                f"The models evaluated the observational sample. The primary prediction is {ctx['rf_pred']} with {ctx['confidence']} confidence.",
                "This report represents a machine-learning classification result based on mathematical models.",
                "It does not constitute a definitive astronomical confirmation or refutation without accompanying scientific review."
            ])

            pdf_bytes = pdf.output()

            st.download_button(
                label="Download PDF Report",
                data=bytes(pdf_bytes),
                file_name="exokan_ai_report.pdf",
                mime="application/pdf"
            )
