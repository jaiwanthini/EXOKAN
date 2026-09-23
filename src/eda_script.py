import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT_DIR = Path("c:/Users/jaiwa/OneDrive/Desktop/EXOKAN")
sys_path_added = False
import sys
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))
from data_loader import load_raw_koi_data

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

def perform_eda():
    df = load_raw_koi_data()
    df = df[df[LABEL_COLUMN].isin(VALID_LABELS)].copy()
    
    analysis = {}
    
    # 1. Basic Stats & Skewness
    features_df = df[FEATURE_COLUMNS]
    stats = {}
    for col in FEATURE_COLUMNS:
        stats[col] = {
            "missing_pct": float(df[col].isna().mean() * 100),
            "unique_vals": int(df[col].nunique()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "skew": float(df[col].skew()),
            "has_zero_or_neg": bool((df[col] <= 0).any())
        }
    analysis["basic_stats"] = stats
    
    # 2. Correlation
    corr_matrix = features_df.corr().to_dict()
    analysis["correlation"] = corr_matrix
    
    # 3. Class-wise Analysis
    class_stats = {}
    for label in VALID_LABELS:
        sub_df = df[df[LABEL_COLUMN] == label]
        class_stats[label] = {}
        for col in FEATURE_COLUMNS:
            class_stats[label][col] = {
                "mean": float(sub_df[col].mean()),
                "median": float(sub_df[col].median()),
                "std": float(sub_df[col].std()),
                "missing_pct": float(sub_df[col].isna().mean() * 100)
            }
    analysis["class_stats"] = class_stats
    
    # 4. Correlation with Target (Mutual Info / ANOVA F-value approximation)
    from sklearn.feature_selection import f_classif
    # Dropna for quick F-score
    clean_df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].dropna()
    X_clean = clean_df[FEATURE_COLUMNS]
    y_clean = clean_df[LABEL_COLUMN].map({"CONFIRMED": 0, "CANDIDATE": 1, "FALSE POSITIVE": 2})
    f_vals, p_vals = f_classif(X_clean, y_clean)
    
    f_scores = {col: float(f) for col, f in zip(FEATURE_COLUMNS, f_vals)}
    analysis["f_scores"] = f_scores
    
    with open(ROOT_DIR / "eda_results.json", "w") as f:
        json.dump(analysis, f, indent=4)
        
    print("EDA completed and saved to eda_results.json")

if __name__ == "__main__":
    perform_eda()
