"""
interpret.py

This is the "payoff" script for your project's core argument:
KAN is interpretable BY DESIGN (visualize learned edge functions directly),
vs. XGBoost/RF which need POST-HOC explanation via SHAP.

Run this after train_kan.py and train_baselines.py have produced saved models.
"""

import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from preprocess import clean_and_split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def plot_kan_activations(model):
    """
    pykan's built-in plotting shows the learned activation function on
    every edge of the network - this is the core "interpretable by design"
    visual for your README/demo.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.plot(beta=100)
    plt.savefig(RESULTS_DIR / "kan_activation_functions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved KAN activation function plot to {RESULTS_DIR / 'kan_activation_functions.png'}")


def shap_analysis_on_baseline():
    """
    Post-hoc explanation on XGBoost, for direct contrast against KAN's
    native interpretability.
    """
    X_train, X_test, y_train, y_test, label_encoder, feature_names = clean_and_split()
    xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(RESULTS_DIR / "xgboost_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {RESULTS_DIR / 'xgboost_shap_summary.png'}")


if __name__ == "__main__":
    # Note: reloading a pykan model needs the same KAN(width=...) call
    # used at training time before loading state_dict - see train_kan.py.
    # For a quick run, call plot_kan_activations() right after model.fit()
    # inside train_kan.py instead of reloading here.
    shap_analysis_on_baseline()
