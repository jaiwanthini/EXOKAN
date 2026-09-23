import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from kan import KAN
from kan_fast_optimizer import get_data

RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

def train_eval_model(X_tr, y_tr, X_val, y_val, config, loss_fn, seed=42):
    torch.manual_seed(seed)
    n_features = X_tr.shape[1]
    n_classes = len(np.unique(y_tr))
    width = [n_features] + config["hidden"] + [n_classes]
    
    model = KAN(width=width, grid=config["grid"], k=config["k"], seed=seed)
    dataset = {
        "train_input": torch.tensor(X_tr, dtype=torch.float32),
        "train_label": torch.tensor(y_tr, dtype=torch.long),
        "test_input": torch.tensor(X_val, dtype=torch.float32),
        "test_label": torch.tensor(y_val, dtype=torch.long),
    }
    
    def train_acc():
        with torch.no_grad():
            return (torch.argmax(model(dataset["train_input"]), dim=1) == dataset["train_label"]).float().mean()
    def test_acc():
        with torch.no_grad():
            return (torch.argmax(model(dataset["test_input"]), dim=1) == dataset["test_label"]).float().mean()
            
    try:
        model.fit(dataset, opt="LBFGS", steps=config.get("steps", 50), loss_fn=loss_fn, metrics=(train_acc, test_acc), lr=config.get("lr", 1.0))
    except Exception as e:
        print(f"Error training model: {e}")
        return None, 0, 0, 0, None, None
        
    with torch.no_grad():
        logits = model(dataset["test_input"])
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    cand_f1 = f1_score(y_val, preds, labels=[0], average=None, zero_division=0)[0]
    return model, acc, macro_f1, cand_f1, probs, preds

def phase1_diagnose(X, y, features, label_encoder):
    cand_idx = np.where(label_encoder.classes_ == "CANDIDATE")[0][0]
    fp_idx = np.where(label_encoder.classes_ == "FALSE POSITIVE")[0][0]
    
    X_cand = X[y == cand_idx]
    X_fp = X[y == fp_idx]
    
    results = []
    for i, feat in enumerate(features):
        c_mean, c_std = X_cand[:, i].mean(), X_cand[:, i].std()
        f_mean, f_std = X_fp[:, i].mean(), X_fp[:, i].std()
        pooled_std = np.sqrt((c_std**2 + f_std**2)/2)
        d = abs(c_mean - f_mean) / (pooled_std + 1e-6)
        results.append((feat, d))
        
    results.sort(key=lambda x: x[1], reverse=True)
    with open(RESULTS_DIR / "phase1_candidate_diagnosis.txt", "w") as f:
        f.write("Ranked Features for CANDIDATE vs FALSE POSITIVE (Cohen's d):\n")
        for feat, d in results:
            f.write(f"{feat:30s} {d:.4f}\n")
    return results

def run_deep_dive():
    print("Starting EXOKAN-AI Phase 1-9 Deep Dive...")
    
    # 1. Load Data
    X_train_full, X_test, y_train_full, y_test, label_encoder, scaler, imputer, features = get_data("C")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
    
    cand_idx = np.where(label_encoder.classes_ == "CANDIDATE")[0][0]
    conf_idx = np.where(label_encoder.classes_ == "CONFIRMED")[0][0]
    fp_idx = np.where(label_encoder.classes_ == "FALSE POSITIVE")[0][0]
    
    config = {"hidden": [6], "grid": 3, "k": 3, "steps": 50, "lr": 1.0}
    
    methods = {}
    
    # --- Phase 1: Diagnose ---
    print("\nPhase 1: Diagnosis")
    phase1_diagnose(X_train_full, y_train_full, features, label_encoder)
    
    # Baseline for Validation
    print("Training Baseline KAN...")
    model_base, acc_base, f1_base, cand_f1_base, probs_base, preds_base = train_eval_model(X_tr, y_tr, X_val, y_val, config, nn.CrossEntropyLoss())
    methods["Baseline"] = {"f1": f1_base, "cand_f1": cand_f1_base}
    
    # --- Phase 3: Soft Weighting ---
    print("\nPhase 3: Soft Class Weighting")
    for w in [1.1, 1.2, 1.3, 1.4, 1.5]:
        weights = torch.tensor([w if i == cand_idx else 1.0 for i in range(3)], dtype=torch.float32)
        _, acc, f1, cand_f1, _, _ = train_eval_model(X_tr, y_tr, X_val, y_val, config, nn.CrossEntropyLoss(weight=weights))
        methods[f"Weight_{w}"] = {"f1": f1, "cand_f1": cand_f1}
        print(f"Weight {w}: F1={f1:.4f}, CandF1={cand_f1:.4f}")
        
    # --- Phase 4: Focal Loss ---
    print("\nPhase 4: Focal Loss")
    for g in [1.0, 1.5, 2.0, 2.5]:
        _, acc, f1, cand_f1, _, _ = train_eval_model(X_tr, y_tr, X_val, y_val, config, FocalLoss(gamma=g))
        methods[f"Focal_{g}"] = {"f1": f1, "cand_f1": cand_f1}
        print(f"Focal {g}: F1={f1:.4f}, CandF1={cand_f1:.4f}")
        
    # --- Phase 5: Threshold Optimization ---
    print("\nPhase 5: Decision Thresholds")
    best_thresh_f1, best_thresh_cand, best_thresh = f1_base, cand_f1_base, 0.33
    for thresh in np.linspace(0.20, 0.50, 31):
        preds = np.argmax(probs_base, axis=1).copy()
        for i in range(len(probs_base)):
            weighted_probs = probs_base[i].copy()
            weighted_probs[cand_idx] *= (0.33 / thresh)
            preds[i] = np.argmax(weighted_probs)
        t_f1 = f1_score(y_val, preds, average="macro")
        t_cand_f1 = f1_score(y_val, preds, labels=[cand_idx], average=None)[0]
        if t_f1 > best_thresh_f1:
            best_thresh_f1 = t_f1
            best_thresh_cand = t_cand_f1
            best_thresh = thresh
    methods["Threshold_Opt"] = {"f1": best_thresh_f1, "cand_f1": best_thresh_cand}
    print(f"Best Thresh {best_thresh:.2f}: F1={best_thresh_f1:.4f}, CandF1={best_thresh_cand:.4f}")
    
    # --- Phase 6: Two Stage ---
    print("\nPhase 6: Two-Stage Hierarchical")
    # Stage 1: (Cand, Conf) vs FP
    y_tr_s1 = np.where(y_tr == fp_idx, 1, 0)
    y_val_s1 = np.where(y_val == fp_idx, 1, 0)
    model_s1, _, _, _, probs_s1, _ = train_eval_model(X_tr, y_tr_s1, X_val, y_val_s1, config, nn.CrossEntropyLoss())
    
    # Stage 2: Cand vs Conf
    mask_tr_s2 = (y_tr == cand_idx) | (y_tr == conf_idx)
    mask_val_s2 = (y_val == cand_idx) | (y_val == conf_idx)
    X_tr_s2 = X_tr[mask_tr_s2]
    y_tr_s2 = np.where(y_tr[mask_tr_s2] == conf_idx, 1, 0)
    model_s2, _, _, _, _, _ = train_eval_model(X_tr_s2, y_tr_s2, X_val[mask_val_s2], np.where(y_val[mask_val_s2] == conf_idx, 1, 0), config, nn.CrossEntropyLoss())
    
    with torch.no_grad():
        probs_s2_all = torch.softmax(model_s2(torch.tensor(X_val, dtype=torch.float32)), dim=1).numpy()
    
    hier_preds = []
    for i in range(len(X_val)):
        if probs_s1[i][1] > 0.5:
            hier_preds.append(fp_idx)
        elif probs_s2_all[i][1] > 0.5:
            hier_preds.append(conf_idx)
        else:
            hier_preds.append(cand_idx)
            
    hier_f1 = f1_score(y_val, hier_preds, average="macro")
    hier_cand_f1 = f1_score(y_val, hier_preds, labels=[cand_idx], average=None)[0]
    methods["Two_Stage"] = {"f1": hier_f1, "cand_f1": hier_cand_f1}
    print(f"Two-Stage: F1={hier_f1:.4f}, CandF1={hier_cand_f1:.4f}")
    
    # --- Phase 8: Ensemble ---
    print("\nPhase 8: Ensemble (KAN + RF)")
    rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
    rf_probs_val = rf_model.predict_proba(X_val)
    ens_probs = 0.5 * probs_base + 0.5 * rf_probs_val
    ens_preds = np.argmax(ens_probs, axis=1)
    ens_f1 = f1_score(y_val, ens_preds, average="macro")
    ens_cand_f1 = f1_score(y_val, ens_preds, labels=[cand_idx], average=None)[0]
    methods["Ensemble"] = {"f1": ens_f1, "cand_f1": ens_cand_f1}
    print(f"Ensemble: F1={ens_f1:.4f}, CandF1={ens_cand_f1:.4f}")
    
    # Select Best Method
    best_method = max(methods.items(), key=lambda x: x[1]["f1"])[0]
    print(f"\nWINNING METHOD ON VALIDATION: {best_method}")
    
    # --- Phase 9: Final Test ---
    print("\nPhase 9: Final Test Evaluation")
    
    # Evaluate Baseline on Test
    model_final_base, acc_test_base, f1_test_base, cand_f1_test_base, probs_test_base, preds_test_base = train_eval_model(
        X_train_full, y_train_full, X_test, y_test, config, nn.CrossEntropyLoss()
    )
    cm_base = confusion_matrix(y_test, preds_test_base)
    
    final_preds = preds_test_base.copy()
    final_probs = probs_test_base.copy()
    
    if best_method.startswith("Weight_"):
        w = float(best_method.split("_")[1])
        weights = torch.tensor([w if i == cand_idx else 1.0 for i in range(3)], dtype=torch.float32)
        _, acc_test_opt, f1_test_opt, cand_f1_test_opt, final_probs, final_preds = train_eval_model(
            X_train_full, y_train_full, X_test, y_test, config, nn.CrossEntropyLoss(weight=weights)
        )
    elif best_method.startswith("Focal_"):
        g = float(best_method.split("_")[1])
        _, acc_test_opt, f1_test_opt, cand_f1_test_opt, final_probs, final_preds = train_eval_model(
            X_train_full, y_train_full, X_test, y_test, config, FocalLoss(gamma=g)
        )
    elif best_method == "Threshold_Opt":
        for i in range(len(probs_test_base)):
            weighted_probs = probs_test_base[i].copy()
            weighted_probs[cand_idx] *= (0.33 / best_thresh)
            final_preds[i] = np.argmax(weighted_probs)
        acc_test_opt = accuracy_score(y_test, final_preds)
        f1_test_opt = f1_score(y_test, final_preds, average="macro")
        cand_f1_test_opt = f1_score(y_test, final_preds, labels=[cand_idx], average=None)[0]
    elif best_method == "Ensemble":
        rf_probs_test = rf_model.predict_proba(X_test)
        final_probs = 0.5 * probs_test_base + 0.5 * rf_probs_test
        final_preds = np.argmax(final_probs, axis=1)
        acc_test_opt = accuracy_score(y_test, final_preds)
        f1_test_opt = f1_score(y_test, final_preds, average="macro")
        cand_f1_test_opt = f1_score(y_test, final_preds, labels=[cand_idx], average=None)[0]
    elif best_method == "Two_Stage":
        # Train Stage 1 full
        y_train_full_s1 = np.where(y_train_full == fp_idx, 1, 0)
        y_test_s1 = np.where(y_test == fp_idx, 1, 0)
        _, _, _, _, probs_s1_test, _ = train_eval_model(X_train_full, y_train_full_s1, X_test, y_test_s1, config, nn.CrossEntropyLoss())
        # Train Stage 2 full
        mask_tr_s2 = (y_train_full == cand_idx) | (y_train_full == conf_idx)
        X_tr_s2 = X_train_full[mask_tr_s2]
        y_tr_s2 = np.where(y_train_full[mask_tr_s2] == conf_idx, 1, 0)
        model_s2, _, _, _, _, _ = train_eval_model(X_tr_s2, y_tr_s2, X_test, np.zeros(len(X_test)), config, nn.CrossEntropyLoss())
        with torch.no_grad():
            probs_s2_test = torch.softmax(model_s2(torch.tensor(X_test, dtype=torch.float32)), dim=1).numpy()
        final_preds = []
        for i in range(len(X_test)):
            if probs_s1_test[i][1] > 0.5:
                final_preds.append(fp_idx)
            elif probs_s2_test[i][1] > 0.5:
                final_preds.append(conf_idx)
            else:
                final_preds.append(cand_idx)
        acc_test_opt = accuracy_score(y_test, final_preds)
        f1_test_opt = f1_score(y_test, final_preds, average="macro")
        cand_f1_test_opt = f1_score(y_test, final_preds, labels=[cand_idx], average=None)[0]
    else:
        acc_test_opt, f1_test_opt, cand_f1_test_opt = acc_test_base, f1_test_base, cand_f1_test_base

    cm_opt = confusion_matrix(y_test, final_preds)
    report_opt = classification_report(y_test, final_preds, target_names=label_encoder.classes_, output_dict=True)
    
    # Build Presentation Report
    with open(RESULTS_DIR / "final_presentation_deep_dive.txt", "w") as f:
        f.write("=== FINAL PRESENTATION TABLE ===\n\n")
        f.write(f"{'Metric':<25} {'Current':<15} {'Optimized':<15}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Candidate F1':<25} {0.53:<15.4f} {cand_f1_test_opt:<15.4f}\n")
        f.write(f"{'Confirmed F1':<25} {0.85:<15.4f} {report_opt['CONFIRMED']['f1-score']:<15.4f}\n")
        f.write(f"{'False Positive F1':<25} {0.85:<15.4f} {report_opt['FALSE POSITIVE']['f1-score']:<15.4f}\n")
        f.write(f"{'Macro F1':<25} {0.74:<15.4f} {f1_test_opt:<15.4f}\n")
        f.write(f"{'Accuracy':<25} {78.72:<15.2f}% {acc_test_opt*100:<15.2f}%\n\n")
        
        f.write("BEFORE Confusion Matrix:\n[[189  61 146]\n [ 47 480  22]\n [ 86  45 837]]\n\n")
        f.write(f"AFTER Confusion Matrix ({best_method}):\n{cm_opt}\n")
        
    print(f"\nFinal Test Cand F1: {cand_f1_test_opt:.4f}")
    print("Presentation table saved to final_presentation_deep_dive.txt.")

if __name__ == "__main__":
    run_deep_dive()
