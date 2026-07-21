"""
train_baselines.py

Trains Random Forest and XGBoost baselines on the same train/test split
used for the KAN model, so results are directly comparable.
"""

import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, ConfusionMatrixDisplay

from preprocess import clean_and_split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def evaluate_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    classes: np.ndarray
) -> Tuple[float, float, str, np.ndarray]:
    """Evaluates the model and returns metrics and predictions."""
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=classes)
    return accuracy, macro_f1, report, preds


def save_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    classes: np.ndarray, 
    model_name: str, 
    save_path: Path
) -> None:
    """Generates and saves a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, 
        y_pred, 
        display_labels=classes, 
        cmap="Blues", 
        ax=ax,
        colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def save_feature_importance(
    importances: np.ndarray, 
    feature_names: List[str], 
    csv_path: Path, 
    plot_path: Path
) -> None:
    """Saves feature importance to a CSV and generates a bar chart for the top 10."""
    # Create DataFrame and sort
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    # Save CSV
    df_imp.to_csv(csv_path, index=False)
    
    # Plot top 10
    top_10 = df_imp.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bar chart, we invert y-axis so highest is on top
    ax.barh(top_10["Feature"], top_10["Importance"], color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Feature Importances (Random Forest)")
    
    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def print_model_results(model_name: str, accuracy: float, macro_f1: float, report: str) -> None:
    """Prints the evaluation results in a formatted manner."""
    print("=" * 50)
    print(f"{model_name} Results")
    print("=" * 50)
    print()
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print()
    print(report)


def train_and_evaluate() -> Dict[str, Any]:
    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, label_encoder, features = clean_and_split()
    classes = label_encoder.classes_

    results = {}
    csv_data = []

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    rf_acc, rf_f1, rf_report, rf_preds = evaluate_model(rf, X_test, y_test, classes)
    
    results["Random Forest"] = {
        "accuracy": rf_acc,
        "f1_macro": rf_f1,
        "report": rf_report,
    }
    csv_data.append({"Model": "Random Forest", "Accuracy": rf_acc, "Macro_F1": rf_f1})
    
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    save_confusion_matrix(y_test, rf_preds, classes, "Random Forest", RESULTS_DIR / "random_forest_confusion_matrix.png")
    save_feature_importance(
        rf.feature_importances_, 
        features, 
        RESULTS_DIR / "random_forest_feature_importance.csv",
        RESULTS_DIR / "random_forest_feature_importance.png"
    )
    print_model_results("Random Forest", rf_acc, rf_f1, rf_report)

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="mlogloss", n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    
    xgb_acc, xgb_f1, xgb_report, xgb_preds = evaluate_model(xgb, X_test, y_test, classes)
    
    results["XGBoost"] = {
        "accuracy": xgb_acc,
        "f1_macro": xgb_f1,
        "report": xgb_report,
    }
    csv_data.append({"Model": "XGBoost", "Accuracy": xgb_acc, "Macro_F1": xgb_f1})
    
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")
    save_confusion_matrix(y_test, xgb_preds, classes, "XGBoost", RESULTS_DIR / "xgboost_confusion_matrix.png")
    print_model_results("XGBoost", xgb_acc, xgb_f1, xgb_report)

    # --- Save Baseline Results ---
    # Text file
    with open(RESULTS_DIR / "baseline_results.txt", "w") as f:
        for name, res in results.items():
            f.write(f"=== {name} ===\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"Macro F1: {res['f1_macro']:.4f}\n")
            f.write(res["report"])
            f.write("\n\n")

    # CSV file
    df_results = pd.DataFrame(csv_data)
    df_results.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

    # --- Print Summary ---
    print("=" * 50)
    print("Files Saved")
    print("=" * 50)
    print()
    print("✓ models/random_forest.joblib")
    print("✓ models/xgboost.joblib")
    print("✓ results/baseline_results.txt")
    print("✓ results/baseline_results.csv")
    print("✓ confusion matrices")
    print("✓ feature importance CSV")
    print("✓ feature importance plot")
    print()

    return results


if __name__ == "__main__":
    train_and_evaluate()
