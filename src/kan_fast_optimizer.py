import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import json
import itertools
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from kan import KAN
from data_loader import load_raw_koi_data
from preprocess import FEATURE_COLUMNS, VALID_LABELS, LABEL_COLUMN

RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

def generate_features(X, feature_set_type):
    X_new = X.copy()
    if feature_set_type in ["B", "C"]:
        X_new["transit_depth_duration_ratio"] = X_new["koi_depth"] / (X_new["koi_duration"] + 1e-6)
        X_new["snr_depth_ratio"] = X_new["koi_model_snr"] / (X_new["koi_depth"] + 1)
        X_new["teq_insol_ratio"] = X_new["koi_insol"] / (X_new["koi_teq"] + 1)
    if feature_set_type == "C":
        skewed_cols = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_insol", "koi_model_snr", "koi_srad"]
        for col in skewed_cols:
            X_new[col] = np.log1p(X_new[col].clip(lower=0))
    return X_new

def get_data(feature_set_type="C"):
    df = load_raw_koi_data()
    df = df[df[LABEL_COLUMN].isin(VALID_LABELS)].copy()
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[LABEL_COLUMN].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_feat = generate_features(X_train_raw, feature_set_type)
    X_test_feat = generate_features(X_test_raw, feature_set_type)
    
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_feat)
    X_test_imp = imputer.transform(X_test_feat)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler, imputer, list(X_train_feat.columns)

def train_eval_kan(X_train, y_train, X_val, y_val, config, seed=42):
    torch.manual_seed(seed)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    width = [n_features] + config["hidden"] + [n_classes]
    
    model = KAN(width=width, grid=config["grid"], k=config["k"], seed=seed)
    
    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32),
        "train_label": torch.tensor(y_train, dtype=torch.long),
        "test_input": torch.tensor(X_val, dtype=torch.float32),
        "test_label": torch.tensor(y_val, dtype=torch.long),
    }
    
    loss_fn = nn.CrossEntropyLoss()
    
    def train_acc():
        with torch.no_grad():
            return (torch.argmax(model(dataset["train_input"]), dim=1) == dataset["train_label"]).float().mean()
            
    def test_acc():
        with torch.no_grad():
            return (torch.argmax(model(dataset["test_input"]), dim=1) == dataset["test_label"]).float().mean()
            
    try:
        model.fit(
            dataset, opt="LBFGS", steps=config.get("steps", 100),
            loss_fn=loss_fn, metrics=(train_acc, test_acc), lr=config.get("lr", 1.0)
        )
    except Exception as e:
        print(f"Error training with config {config}: {e}")
        return None, 0, 0, None
        
    with torch.no_grad():
        logits = model(dataset["test_input"])
        preds = torch.argmax(logits, dim=1).numpy()
        
    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    return model, acc, macro_f1, preds

def run_targeted_search():
    print("Starting KAN Targeted Fast Optimization...")
    X_train, X_test, y_train, y_test, label_encoder, scaler, imputer, features = get_data("C")
    
    # We will test only a few highly promising configs to save time
    configs = [
        {"hidden": [6], "grid": 3, "k": 3, "steps": 50, "lr": 1.0},
        {"hidden": [10], "grid": 3, "k": 3, "steps": 50, "lr": 1.0},
        {"hidden": [6], "grid": 5, "k": 3, "steps": 50, "lr": 1.0},
        {"hidden": [10], "grid": 5, "k": 3, "steps": 50, "lr": 1.0},
    ]
    
    best_config = configs[0]
    best_f1 = 0
    results = []
    
    # Stratified Split for Validation during Tuning
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    for c in configs:
        print(f"\nTesting Config: {c}")
        _, acc, f1, _ = train_eval_kan(X_tr, y_tr, X_val, y_val, c)
        print(f"-> Val Acc: {acc:.4f}, Val Macro F1: {f1:.4f}")
        results.append({"Config": str(c), "Val_Accuracy": acc, "Val_Macro_F1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_config = c
            
    print(f"\n=== Best Config Found: {best_config} ===")
    
    print("\n--- FINAL TEST EVALUATION ---")
    # Train best config on full training set, evaluate on pristine test set
    final_model, test_acc, test_f1, test_preds = train_eval_kan(X_train, y_train, X_test, y_test, best_config, seed=42)
    
    report = classification_report(y_test, test_preds, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_test, test_preds)
    
    print("\nTest Report:")
    print(report)
    
    with open(RESULTS_DIR / "kan_optimized_report.txt", "w") as f:
        f.write(f"Optimized KAN Configuration:\n{json.dumps(best_config, indent=2)}\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro F1: {test_f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), MODELS_DIR / "kan_model.pt")
    
    width_full = [len(features)] + best_config["hidden"] + [len(label_encoder.classes_)]
    np.save(MODELS_DIR / "kan_config.npy", {
        "width": width_full, 
        "grid": best_config["grid"], 
        "k": best_config["k"]
    })
    
    joblib.dump(imputer, MODELS_DIR / "imputer.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    
    pd.DataFrame(results).to_csv(RESULTS_DIR / "kan_optimization_search.csv", index=False)
    print("\nOptimization Complete. Model and config saved.")

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_targeted_search()
