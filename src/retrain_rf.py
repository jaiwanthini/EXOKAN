import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from kan_optimizer import get_data

MODELS_DIR = ROOT_DIR / "models"

def retrain_rf():
    print("Loading data and engineered features (Set C)...")
    X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, class_weights, scaler, imputer, features = get_data("C")
    
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    print("Saving Random Forest...")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    
    acc = rf.score(X_test_scaled, y_test)
    print(f"Random Forest Test Accuracy on Feature Set C: {acc:.4f}")

if __name__ == "__main__":
    retrain_rf()
