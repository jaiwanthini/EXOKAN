
"""
preprocess.py

Cleans the raw KOI dataframe and produces a model-ready feature/label set.

Key decisions:
- We use only physical/transit measurements as features.
- NASA's own vetting flags and verdict-related columns are excluded.
- Only CONFIRMED, CANDIDATE, and FALSE POSITIVE rows are retained.
- Missing values are median-imputed using statistics learned ONLY from
  the training set.
- Standardization is also fitted ONLY on the training set.

This prevents preprocessing leakage from the test set into training.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from pathlib import Path
import joblib

from data_loader import load_raw_koi_data


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


# Physical/transit measurement columns - the actual signal we want
# the model to learn from.
FEATURE_COLUMNS = [
    "koi_period",       # orbital period (days)
    "koi_duration",     # transit duration (hours)
    "koi_depth",        # transit depth (ppm)
    "koi_prad",         # planetary radius (Earth radii)
    "koi_teq",          # equilibrium temperature (K)
    "koi_insol",        # insolation flux (Earth flux)
    "koi_model_snr",    # transit signal-to-noise ratio
    "koi_steff",        # stellar effective temperature (K)
    "koi_slogg",        # stellar surface gravity
    "koi_srad",         # stellar radius (solar radii)
    "koi_impact",       # impact parameter
]

LABEL_COLUMN = "koi_disposition"

VALID_LABELS = [
    "CONFIRMED",
    "CANDIDATE",
    "FALSE POSITIVE",
]


def clean_and_split(test_size: float = 0.2, random_state: int = 42):
    """
    Load, clean, split, impute, and scale the KOI dataset.

    Important:
    - The train/test split happens BEFORE fitting the imputer.
    - The imputer and scaler are fitted only on X_train.
    - X_test is transformed using statistics learned from X_train.

    Returns
    -------
    X_train_scaled : numpy.ndarray
    X_test_scaled : numpy.ndarray
    y_train : numpy.ndarray
    y_test : numpy.ndarray
    label_encoder : LabelEncoder
    feature_columns : list
    """

    # ---------------------------------------------------------
    # 1. Load raw data
    # ---------------------------------------------------------
    df = load_raw_koi_data()

    # Keep only the three target classes
    df = df[df[LABEL_COLUMN].isin(VALID_LABELS)].copy()

    # Keep only selected features + target
    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()

    # ---------------------------------------------------------
    # 2. Separate features and labels
    # ---------------------------------------------------------
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[LABEL_COLUMN].values

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # ---------------------------------------------------------
    # 3. Train/test split BEFORE learning preprocessing stats
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ---------------------------------------------------------
    # 4. Fit imputer ONLY on training data
    # ---------------------------------------------------------
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # ---------------------------------------------------------
    # 5. Fit scaler ONLY on training data
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # ---------------------------------------------------------
    # 6. Save preprocessing objects
    # ---------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        imputer,
        MODELS_DIR / "imputer.joblib"
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.joblib"
    )

    joblib.dump(
        label_encoder,
        MODELS_DIR / "label_encoder.joblib"
    )

    # ---------------------------------------------------------
    # 7. Return model-ready data
    # ---------------------------------------------------------
    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        label_encoder,
        FEATURE_COLUMNS,
    )


if __name__ == "__main__":

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        features,
    ) = clean_and_split()

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Classes:     {list(label_encoder.classes_)}")
    print(f"Features:    {features}")




