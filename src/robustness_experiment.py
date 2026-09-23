import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import numpy as np

# from kan import KAN
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"
import joblib

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
VALID_LABELS = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]

SEEDS = [42, 7, 21, 100, 123]

KAN_WIDTH = [11, 6, 3]
KAN_GRID = 8
KAN_K = 3
KAN_STEPS = 100
KAN_OPTIMIZER = "LBFGS"

# Function to load raw data
def load_and_encode_data():
    from data_loader import load_raw_koi_data
    df = load_raw_koi_data()
    df = df[df[LABEL_COLUMN].isin(VALID_LABELS)].copy()
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[LABEL_COLUMN].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    return X, y, label_encoder

def train_and_eval_models(X_train_scaled, X_test_scaled, y_train, y_test, seed, label_encoder):
    from kan import KAN
    results = []
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed, eval_metric="mlogloss", n_jobs=-1)
    xgb.fit(X_train_scaled, y_train)
    xgb_preds = xgb.predict(X_test_scaled)
    
    # KAN
    torch.manual_seed(seed)
    kan = KAN(width=KAN_WIDTH, grid=KAN_GRID, k=KAN_K, seed=seed)
    train_input = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_label = torch.tensor(y_train, dtype=torch.long)
    test_input = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_label = torch.tensor(y_test, dtype=torch.long)
    
    dataset = {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": test_input,
        "test_label": test_label,
    }
    
    def train_acc():
        with torch.no_grad():
            return (torch.argmax(kan(train_input), dim=1) == train_label).float().mean()
    def test_acc():
        with torch.no_grad():
            return (torch.argmax(kan(test_input), dim=1) == test_label).float().mean()
            
    kan.fit(dataset, opt=KAN_OPTIMIZER, steps=KAN_STEPS, loss_fn=nn.CrossEntropyLoss(), metrics=(train_acc, test_acc))
    
    with torch.no_grad():
        kan_preds = torch.argmax(kan(test_input), dim=1).numpy()
    
    preds_dict = {
        "Random Forest": rf_preds,
        "XGBoost": xgb_preds,
        "KAN": kan_preds
    }
    
    for model_name, preds in preds_dict.items():
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        report = classification_report(y_test, preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
        
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Macro_F1": macro_f1,
            "Candidate_Precision": report.get("CANDIDATE", {}).get("precision", 0),
            "Candidate_Recall": report.get("CANDIDATE", {}).get("recall", 0),
            "Candidate_F1": report.get("CANDIDATE", {}).get("f1-score", 0),
            "Confirmed_Precision": report.get("CONFIRMED", {}).get("precision", 0),
            "Confirmed_Recall": report.get("CONFIRMED", {}).get("recall", 0),
            "Confirmed_F1": report.get("CONFIRMED", {}).get("f1-score", 0),
            "False_Positive_Precision": report.get("FALSE POSITIVE", {}).get("precision", 0),
            "False_Positive_Recall": report.get("FALSE POSITIVE", {}).get("recall", 0),
            "False_Positive_F1": report.get("FALSE POSITIVE", {}).get("f1-score", 0),
        })
    return results

def compute_summary(df):
    summary = []
    models = df["Model"].unique()
    for model in models:
        m_df = df[df["Model"] == model]
        summary.append({
            "Model": model,
            "Mean_Accuracy": m_df["Accuracy"].mean(),
            "Std_Accuracy": m_df["Accuracy"].std(ddof=1),
            "Mean_Macro_F1": m_df["Macro_F1"].mean(),
            "Std_Macro_F1": m_df["Macro_F1"].std(ddof=1),
            "Mean_Candidate_F1": m_df["Candidate_F1"].mean(),
            "Std_Candidate_F1": m_df["Candidate_F1"].std(ddof=1),
            "Mean_Confirmed_F1": m_df["Confirmed_F1"].mean(),
            "Std_Confirmed_F1": m_df["Confirmed_F1"].std(ddof=1),
            "Mean_False_Positive_F1": m_df["False_Positive_F1"].mean(),
            "Std_False_Positive_F1": m_df["False_Positive_F1"].std(ddof=1),
        })
    return pd.DataFrame(summary)

def run_part_a(X, y, label_encoder):
    print("\n--- PART A: MULTIPLE RANDOM SEEDS ---")
    all_results = []
    for seed in SEEDS:
        print(f"Running Seed {seed}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
        
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        res = train_and_eval_models(X_train_scaled, X_test_scaled, y_train, y_test, seed, label_encoder)
        for r in res:
            r["Seed"] = seed
        all_results.extend(res)
        
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "multiple_seed_results.csv", index=False)
    
    summary_df = compute_summary(df)
    summary_df.to_csv(RESULTS_DIR / "multiple_seed_summary.csv", index=False)
    
    with open(RESULTS_DIR / "multiple_seed_results.txt", "w") as f:
        f.write("MULTIPLE RANDOM SEEDS (Exploratory held-out test evaluation)\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        for idx, row in summary_df.iterrows():
            f.write(f"=== {row['Model']} ===\n")
            f.write(f"Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}\n")
            f.write(f"Macro F1: {row['Mean_Macro_F1']:.4f} ± {row['Std_Macro_F1']:.4f}\n")
            f.write(f"Candidate F1: {row['Mean_Candidate_F1']:.4f} ± {row['Std_Candidate_F1']:.4f}\n")
            f.write(f"Confirmed F1: {row['Mean_Confirmed_F1']:.4f} ± {row['Std_Confirmed_F1']:.4f}\n")
            f.write(f"False Positive F1: {row['Mean_False_Positive_F1']:.4f} ± {row['Std_False_Positive_F1']:.4f}\n\n")
    return df, summary_df

def run_part_b(X, y, label_encoder):
    print("\n--- PART B: STRATIFIED 5-FOLD CROSS-VALIDATION ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_id = fold + 1
        print(f"Running Fold {fold_id}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        res = train_and_eval_models(X_train_scaled, X_val_scaled, y_train, y_val, 42, label_encoder)
        for r in res:
            r["Fold"] = fold_id
        all_results.extend(res)
        
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "cross_validation_results.csv", index=False)
    
    summary_df = compute_summary(df)
    summary_df.to_csv(RESULTS_DIR / "cross_validation_summary.csv", index=False)
    
    with open(RESULTS_DIR / "cross_validation_results.txt", "w") as f:
        f.write("5-FOLD CROSS-VALIDATION ROBUSTNESS EVALUATION\n")
        f.write("Preprocessing performed INSIDE each fold to prevent leakage.\n\n")
        for idx, row in summary_df.iterrows():
            f.write(f"=== {row['Model']} ===\n")
            f.write(f"Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}\n")
            f.write(f"Macro F1: {row['Mean_Macro_F1']:.4f} ± {row['Std_Macro_F1']:.4f}\n")
            f.write(f"Candidate F1: {row['Mean_Candidate_F1']:.4f} ± {row['Std_Candidate_F1']:.4f}\n")
            f.write(f"Confirmed F1: {row['Mean_Confirmed_F1']:.4f} ± {row['Std_Confirmed_F1']:.4f}\n")
            f.write(f"False Positive F1: {row['Mean_False_Positive_F1']:.4f} ± {row['Std_False_Positive_F1']:.4f}\n\n")
    return df, summary_df

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, label_encoder = load_and_encode_data()
    run_part_a(X, y, label_encoder)
    run_part_b(X, y, label_encoder)
    print("Done!")

if __name__ == "__main__":
    main()
