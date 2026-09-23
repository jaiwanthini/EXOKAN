import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

FEATURE_COLUMNS_ORIGINAL = [
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

KAN_GRID = 8
KAN_K = 3
KAN_STEPS = 100
KAN_OPTIMIZER = "LBFGS"

# Function to load raw data
def load_and_encode_data():
    from data_loader import load_raw_koi_data
    df = load_raw_koi_data()
    df = df[df[LABEL_COLUMN].isin(VALID_LABELS)].copy()
    X = df[FEATURE_COLUMNS_ORIGINAL].copy()
    y_raw = df[LABEL_COLUMN].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    return X, y, label_encoder

def generate_features(X, feature_set_type):
    # Create a copy so we don't modify the original during fold logic
    X_new = X.copy()
    
    if feature_set_type in ["B", "C"]:
        # Domain Engineered features
        X_new["transit_depth_duration_ratio"] = X_new["koi_depth"] / (X_new["koi_duration"] + 1e-6)
        X_new["snr_depth_ratio"] = X_new["koi_model_snr"] / (X_new["koi_depth"] + 1)
        X_new["teq_insol_ratio"] = X_new["koi_insol"] / (X_new["koi_teq"] + 1)
        
    if feature_set_type == "C":
        # Log1p transforms on highly skewed variables
        skewed_cols = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_insol", "koi_model_snr", "koi_srad"]
        for col in skewed_cols:
            X_new[col] = np.log1p(X_new[col].clip(lower=0)) # Ensure no negative values for log1p
            
    return X_new

def train_and_eval_models(X_train_scaled, X_test_scaled, y_train, y_test, seed, label_encoder):
    from kan import KAN
    results = []
    
    num_features = X_train_scaled.shape[1]
    
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
    # The KAN baseline width is [number_of_features, 6, 3]
    kan_width = [num_features, 6, 3]
    kan = KAN(width=kan_width, grid=KAN_GRID, k=KAN_K, seed=seed)
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
            "Candidate_F1": report.get("CANDIDATE", {}).get("f1-score", 0),
            "Confirmed_F1": report.get("CONFIRMED", {}).get("f1-score", 0),
            "False_Positive_F1": report.get("FALSE POSITIVE", {}).get("f1-score", 0),
        })
    return results

def compute_summary(df, group_cols):
    summary = df.groupby(group_cols).agg(
        Mean_Accuracy=("Accuracy", "mean"),
        Std_Accuracy=("Accuracy", lambda x: x.std(ddof=1)),
        Mean_Macro_F1=("Macro_F1", "mean"),
        Std_Macro_F1=("Macro_F1", lambda x: x.std(ddof=1)),
        Mean_Candidate_F1=("Candidate_F1", "mean"),
        Std_Candidate_F1=("Candidate_F1", lambda x: x.std(ddof=1)),
        Mean_Confirmed_F1=("Confirmed_F1", "mean"),
        Std_Confirmed_F1=("Confirmed_F1", lambda x: x.std(ddof=1)),
        Mean_False_Positive_F1=("False_Positive_F1", "mean"),
        Std_False_Positive_F1=("False_Positive_F1", lambda x: x.std(ddof=1))
    ).reset_index()
    return summary

def run_experiment():
    print("\n--- FEATURE ENGINEERING 5-FOLD CV ---")
    X_raw, y, label_encoder = load_and_encode_data()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    
    for fs_type in ["A", "B", "C"]:
        print(f"\nEvaluating Feature Set {fs_type}")
        
        # We generate features inside the loop to ensure no row-leakage, 
        # though these specific features are single-row deterministic so it's safe.
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
            fold_id = fold + 1
            print(f"  Fold {fold_id}...")
            
            X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Apply feature engineering
            X_train = generate_features(X_train_raw, fs_type)
            X_val = generate_features(X_val_raw, fs_type)
            
            imputer = SimpleImputer(strategy="median")
            X_train_imputed = imputer.fit_transform(X_train)
            X_val_imputed = imputer.transform(X_val)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_val_scaled = scaler.transform(X_val_imputed)
            
            res = train_and_eval_models(X_train_scaled, X_val_scaled, y_train, y_val, 42, label_encoder)
            for r in res:
                r["Feature_Set"] = fs_type
                r["Fold"] = fold_id
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "feature_engineering_results.csv", index=False)
    
    summary_df = compute_summary(df, ["Feature_Set", "Model"])
    summary_df.to_csv(RESULTS_DIR / "feature_engineering_summary.csv", index=False)
    
    with open(RESULTS_DIR / "feature_engineering_report.txt", "w") as f:
        f.write("FEATURE ENGINEERING 5-FOLD CV ROBUSTNESS EVALUATION\n")
        f.write("Feature Set A: Original 11 features\n")
        f.write("Feature Set B: Set A + 3 engineered ratio features\n")
        f.write("Feature Set C: Set B + Log1p transformations of skewed features\n\n")
        
        for fs_type in ["A", "B", "C"]:
            f.write(f"=== FEATURE SET {fs_type} ===\n")
            fs_summary = summary_df[summary_df["Feature_Set"] == fs_type]
            for idx, row in fs_summary.iterrows():
                f.write(f"  [{row['Model']}]\n")
                f.write(f"    Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}\n")
                f.write(f"    Macro F1: {row['Mean_Macro_F1']:.4f} ± {row['Std_Macro_F1']:.4f}\n")
                f.write(f"    Candidate F1: {row['Mean_Candidate_F1']:.4f} ± {row['Std_Candidate_F1']:.4f}\n")
                f.write(f"    Confirmed F1: {row['Mean_Confirmed_F1']:.4f} ± {row['Std_Confirmed_F1']:.4f}\n")
                f.write(f"    False Positive F1: {row['Mean_False_Positive_F1']:.4f} ± {row['Std_False_Positive_F1']:.4f}\n")
            f.write("\n")
            
    print("Feature engineering experiment completed successfully!")

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_experiment()
