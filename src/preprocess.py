"""
preprocess.py

Cleans the raw KOI dataframe and produces a model-ready feature/label set.

Key decisions (worth explaining in your README/interview):
- We DROP columns that leak the label indirectly, e.g. NASA's own vetting
  flags (koi_fpflag_*) and score columns, since the goal is to predict
  disposition FROM raw physical/transit measurements, not from NASA's own
  pre-computed verdict flags.
- We keep only CONFIRMED, CANDIDATE, FALSE POSITIVE rows (drop anything else).
- Missing values are median-imputed per column - astronomy data commonly has
  gaps where a measurement couldn't be made reliably.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import joblib

from data_loader import load_raw_koi_data

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Physical/transit measurement columns - the actual "signal" we want the
# model to learn from (no leakage from NASA's own vetting pipeline).
FEATURE_COLUMNS = [
    "koi_period",       # orbital period (days)
    "koi_duration",      # transit duration (hours)
    "koi_depth",         # transit depth (ppm)
    "koi_prad",           # planetary radius (Earth radii)
    "koi_teq",             # equilibrium temperature (K)
    "koi_insol",             # insolation flux (Earth flux)
    "koi_model_snr",           # transit signal-to-noise ratio
    "koi_steff",                 # stellar effective temperature (K)
    "koi_slogg",                   # stellar surface gravity
    "koi_srad",                      # stellar radius (solar radii)
    "koi_impact",                      # impact parameter
]

LABEL_COLUMN = "koi_disposition"
VALID_LABELS = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]


def clean_and_split(test_size: float = 0.2, random_state: int = 42):
    df = load_raw_koi_data()

    df = df[df[LABEL_COLUMN].isin(VALID_LABELS)].copy()

    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()

    # Median imputation per column
    for col in FEATURE_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLUMNS].values
    y_raw = df[LABEL_COLUMN].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)  # CONFIRMED/CANDIDATE/FALSE POSITIVE -> 0/1/2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, FEATURE_COLUMNS


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, features = clean_and_split()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Classes: {list(le.classes_)}")
    print(f"Features used: {features}")
