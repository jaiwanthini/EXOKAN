
"""
evaluate_models.py

Controlled comparison of:
    - Random Forest
    - XGBoost
    - Kolmogorov-Arnold Network (KAN)

All models use:
    - the same dataset
    - the same feature set
    - the same stratified train/test split
    - the same leakage-free preprocessing
    - the same evaluation metrics

The test set is used ONLY for final evaluation.
"""

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from xgboost import XGBClassifier
from kan import KAN

from data_loader import load_raw_koi_data


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"


# ============================================================
# Dataset configuration
# ============================================================

FEATURE_COLUMNS = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_impact",
]

LABEL_COLUMN = "koi_disposition"

VALID_LABELS = [
    "CONFIRMED",
    "CANDIDATE",
    "FALSE POSITIVE",
]


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

KAN_STEPS = 100
KAN_WIDTH = [11, 6, 3]
KAN_GRID = 8
KAN_K = 3


# ============================================================
# Data preparation
# ============================================================

def prepare_data(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Load and preprocess the dataset.

    The train/test split happens BEFORE fitting the imputer
    and scaler, preventing test-set preprocessing leakage.
    """

    df = load_raw_koi_data()

    # Keep only the three target classes
    df = df[df[LABEL_COLUMN].isin(VALID_LABELS)].copy()

    # Keep only selected features + target
    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()

    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[LABEL_COLUMN].values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # --------------------------------------------------------
    # Train/test split FIRST
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # --------------------------------------------------------
    # Imputation fitted ONLY on training data
    # --------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # --------------------------------------------------------
    # Scaling fitted ONLY on training data
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        imputer,
        scaler,
    )


# ============================================================
# Evaluation utilities
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred,
    label_encoder,
):
    """
    Calculate the common evaluation metrics.
    """

    labels = np.arange(len(label_encoder.classes_))

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


# ============================================================
# Random Forest
# ============================================================

def train_random_forest(
    X_train,
    y_train,
    random_state=RANDOM_STATE,
):
    """
    Train the fixed Random Forest baseline.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# XGBoost
# ============================================================

def train_xgboost(
    X_train,
    y_train,
    random_state=RANDOM_STATE,
):
    """
    Train the fixed XGBoost baseline.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="mlogloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# KAN
# ============================================================

def train_kan(
    X_train,
    X_test,
    y_train,
    y_test,
    n_classes,
    width=None,
    grid=KAN_GRID,
    k=KAN_K,
    steps=KAN_STEPS,
    seed=RANDOM_STATE,
):
    """
    Train a KAN using the same train/test data as the baselines.
    """

    if width is None:
        width = [
            X_train.shape[1],
            6,
            n_classes,
        ]

    # Set PyTorch seed for reproducibility
    torch.manual_seed(seed)

    train_input = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    train_label = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    test_input = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    test_label = torch.tensor(
        y_test,
        dtype=torch.long,
    )

    dataset = {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": test_input,
        "test_label": test_label,
    }

    model = KAN(
        width=width,
        grid=grid,
        k=k,
        seed=seed,
    )

    def train_acc():
        with torch.no_grad():
            logits = model(train_input)
            preds = torch.argmax(logits, dim=1)

            return (
                preds == train_label
            ).float().mean()

    def test_acc():
        with torch.no_grad():
            logits = model(test_input)
            preds = torch.argmax(logits, dim=1)

            return (
                preds == test_label
            ).float().mean()

    print("\nTraining KAN...")
    print(f"Architecture: {width}")
    print(f"Grid:         {grid}")
    print(f"k:            {k}")
    print(f"Steps:        {steps}")

    model.fit(
        dataset,
        opt="LBFGS",
        steps=steps,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=(train_acc, test_acc),
    )

    return model


# ============================================================
# Main experiment
# ============================================================

def run_experiment():

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("CONTROLLED KOI MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Prepare data ONCE
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        imputer,
        scaler,
    ) = prepare_data()

    classes = label_encoder.classes_

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print(f"Features:      {X_train.shape[1]}")
    print(f"Classes:       {list(classes)}")
    print(f"Random state:  {RANDOM_STATE}")

    results = []

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RANDOM FOREST")
    print("=" * 70)

    rf = train_random_forest(
        X_train,
        y_train,
    )

    rf_preds = rf.predict(X_test)

    rf_results = evaluate_predictions(
        y_test,
        rf_preds,
        label_encoder,
    )

    print(
        f"Accuracy: {rf_results['accuracy']:.4f}"
    )

    print(
        f"Macro F1: {rf_results['macro_f1']:.4f}"
    )

    print(rf_results["report"])

    results.append({
        "Model": "Random Forest",
        "Accuracy": rf_results["accuracy"],
        "Macro_F1": rf_results["macro_f1"],
    })

    joblib.dump(
        rf,
        MODELS_DIR / "random_forest.joblib",
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("XGBOOST")
    print("=" * 70)

    xgb = train_xgboost(
        X_train,
        y_train,
    )

    xgb_preds = xgb.predict(X_test)

    xgb_results = evaluate_predictions(
        y_test,
        xgb_preds,
        label_encoder,
    )

    print(
        f"Accuracy: {xgb_results['accuracy']:.4f}"
    )

    print(
        f"Macro F1: {xgb_results['macro_f1']:.4f}"
    )

    print(xgb_results["report"])

    results.append({
        "Model": "XGBoost",
        "Accuracy": xgb_results["accuracy"],
        "Macro_F1": xgb_results["macro_f1"],
    })

    joblib.dump(
        xgb,
        MODELS_DIR / "xgboost.joblib",
    )

    # --------------------------------------------------------
    # KAN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("KAN")
    print("=" * 70)

    kan = train_kan(
        X_train,
        X_test,
        y_train,
        y_test,
        n_classes=len(classes),
        width=KAN_WIDTH,
        grid=KAN_GRID,
        k=KAN_K,
        steps=KAN_STEPS,
        seed=RANDOM_STATE,
    )

    with torch.no_grad():
        kan_logits = kan(
            torch.tensor(
                X_test,
                dtype=torch.float32,
            )
        )

        kan_preds = torch.argmax(
            kan_logits,
            dim=1,
        ).numpy()

    kan_results = evaluate_predictions(
        y_test,
        kan_preds,
        label_encoder,
    )

    print(
        f"Accuracy: {kan_results['accuracy']:.4f}"
    )

    print(
        f"Macro F1: {kan_results['macro_f1']:.4f}"
    )

    print(kan_results["report"])

    results.append({
        "Model": "KAN",
        "Accuracy": kan_results["accuracy"],
        "Macro_F1": kan_results["macro_f1"],
    })

    # --------------------------------------------------------
    # Save preprocessing objects
    # --------------------------------------------------------

    joblib.dump(
        imputer,
        MODELS_DIR / "imputer.joblib",
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.joblib",
    )

    joblib.dump(
        label_encoder,
        MODELS_DIR / "label_encoder.joblib",
    )

    # --------------------------------------------------------
    # Save KAN
    # --------------------------------------------------------

    torch.save(
        kan.state_dict(),
        MODELS_DIR / "kan_model.pt",
    )

    np.save(
        MODELS_DIR / "kan_config.npy",
        {
            "width": KAN_WIDTH,
            "grid": KAN_GRID,
            "k": KAN_K,
        },
    )

    # --------------------------------------------------------
    # Save complete summary
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save detailed text results
    # --------------------------------------------------------

    with open(
        RESULTS_DIR / "model_comparison.txt",
        "w",
    ) as f:

        f.write(
            "CONTROLLED KOI MODEL EVALUATION\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Random state: {RANDOM_STATE}\n"
        )

        f.write(
            f"Test size: {TEST_SIZE}\n"
        )

        f.write(
            f"Features: {len(FEATURE_COLUMNS)}\n"
        )

        f.write(
            f"KAN architecture: {KAN_WIDTH}\n"
        )

        f.write(
            f"KAN grid: {KAN_GRID}\n"
        )

        f.write(
            f"KAN k: {KAN_K}\n"
        )

        f.write(
            f"KAN steps: {KAN_STEPS}\n\n"
        )

        for name, model_results in [
            ("Random Forest", rf_results),
            ("XGBoost", xgb_results),
            ("KAN", kan_results),
        ]:

            f.write(
                f"=== {name} ===\n"
            )

            f.write(
                f"Accuracy: "
                f"{model_results['accuracy']:.4f}\n"
            )

            f.write(
                f"Macro F1: "
                f"{model_results['macro_f1']:.4f}\n\n"
            )

            f.write(
                model_results["report"]
            )

            f.write("\n\n")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved:")
    print("  results/model_comparison.csv")
    print("  results/model_comparison.txt")
    print("  models/random_forest.joblib")
    print("  models/xgboost.joblib")
    print("  models/kan_model.pt")
    print("  models/kan_config.npy")
    print("  models/imputer.joblib")
    print("  models/scaler.joblib")
    print("  models/label_encoder.joblib")

    return {
        "results": results_df,
        "rf": rf,
        "xgb": xgb,
        "kan": kan,
        "y_test": y_test,
        "rf_preds": rf_preds,
        "xgb_preds": xgb_preds,
        "kan_preds": kan_preds,
    }


if __name__ == "__main__":
    run_experiment()

