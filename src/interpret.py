"""
interpret.py

This is the "payoff" script for your project's core argument:
KAN is interpretable BY DESIGN (visualize learned edge functions directly),
vs. XGBoost/RF which need POST-HOC explanation via SHAP.

Run this after train_kan.py and train_baselines.py have produced saved models.
"""

import joblib
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from kan import KAN

from preprocess import clean_and_split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_trained_kan():
    """
    Reconstructs the KAN with the exact same architecture used at training
    time (saved in kan_config.npy by train_kan.py), then loads the trained
    weights. This mirrors what streamlit_app.py does to load the model.
    """
    config = np.load(MODELS_DIR / "kan_config.npy", allow_pickle=True).item()
    model = KAN(width=config["width"], grid=config["grid"], k=config["k"], seed=42)
    model.load_state_dict(torch.load(MODELS_DIR / "kan_model.pt"))
    model.eval()
    return model


def plot_kan_activations(model, sample_input):
    """
    pykan's built-in plotting shows the learned activation function on
    every edge of the network - this is the core "interpretable by design"
    visual for your README/demo.

    Important: model.plot() reads activation values that get cached during
    a forward pass, so we must run the model on some input BEFORE plotting,
    otherwise pykan has nothing to visualize yet.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model(sample_input)  # populates internal activation cache

    model.plot(beta=100)
    plt.savefig(RESULTS_DIR / "kan_activation_functions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved KAN activation function plot to {RESULTS_DIR / 'kan_activation_functions.png'}")


def shap_analysis_on_baseline():
    """
    Post-hoc explanation on XGBoost, for direct contrast against KAN's
    native interpretability.

    Note: for multi-class models, newer versions of SHAP's TreeExplainer
    return a single 3D array of shape (n_samples, n_features, n_classes)
    instead of a list of per-class arrays. If you pass that 3D array
    straight into summary_plot(), SHAP auto-detects the 3D shape as
    INTERACTION values (not per-class importance) and plots a confusing
    feature x feature interaction grid instead. We explicitly slice out
    each class's values below to avoid that.
    """
    X_train, X_test, y_train, y_test, label_encoder, feature_names = clean_and_split()
    xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(shap_values, list):
        # Older SHAP versions: list of (n_samples, n_features) arrays, one per class
        class_shap_values = shap_values
    elif shap_values.ndim == 3:
        # Newer SHAP versions: single (n_samples, n_features, n_classes) array
        class_shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        # Binary classification / already 2D
        class_shap_values = [shap_values]

    for class_idx, class_name in enumerate(label_encoder.classes_):
        plt.figure()
        shap.summary_plot(
            class_shap_values[class_idx], X_test, feature_names=feature_names, show=False
        )
        plt.title(f"SHAP summary — class: {class_name}")
        safe_name = class_name.replace(" ", "_").lower()
        out_path = RESULTS_DIR / f"xgboost_shap_summary_{safe_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot for class '{class_name}' to {out_path}")


if __name__ == "__main__":
    # 1. KAN activation function plot
    X_train, X_test, y_train, y_test, label_encoder, feature_names = clean_and_split()
    kan_model = load_trained_kan()
    sample_input = torch.tensor(X_test, dtype=torch.float32)
    plot_kan_activations(kan_model, sample_input)

    # 2. SHAP summary plots on the XGBoost baseline, for contrast
    shap_analysis_on_baseline()

