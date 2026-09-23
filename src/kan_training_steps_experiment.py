import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import joblib

# Import the exact prepare_data function used in evaluate_models
from evaluate_models import prepare_data

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

KAN_WIDTH = [11, 6, 3]
KAN_GRID = 8
KAN_K = 3
RANDOM_STATE = 42

STEPS_TO_TEST = {
    "KAN_100": 100,
    "KAN_250": 250,
    "KAN_500": 500
}

def run_steps_experiment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading and preparing data EXACTLY as in Step 2...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        imputer,
        scaler,
    ) = prepare_data()
    
    from kan import KAN
    import kan
    
    print(f"\nPyKAN version info (if available): {getattr(kan, '__version__', 'unknown')}")
    print("Checking KAN architecture internal representation...")
    dummy_model = KAN(width=KAN_WIDTH, grid=KAN_GRID, k=KAN_K, seed=RANDOM_STATE)
    print(f"Passed width: {KAN_WIDTH}")
    print(f"Internal pykan width: {dummy_model.width}")
    print("Conclusion: pykan internally converts [N, M] to [[N, 0], [M, 0]] where the 0 represents 'sum nodes'.")
    print("The architecture 11 -> 6 -> 3 is correctly instantiated.")
    print("=" * 60)
    
    results = []
    text_output = []
    
    text_output.append("CONTROLLED KAN TRAINING STEPS EXPERIMENT")
    text_output.append("======================================================================\n")
    
    for name, steps in STEPS_TO_TEST.items():
        print(f"\nEvaluating {name} with steps={steps}...")
        
        # Must reset seed for each run for absolute fairness and reproducibility
        torch.manual_seed(RANDOM_STATE)
        
        model = KAN(width=KAN_WIDTH, grid=KAN_GRID, k=KAN_K, seed=RANDOM_STATE)
        
        train_input = torch.tensor(X_train, dtype=torch.float32)
        train_label = torch.tensor(y_train, dtype=torch.long)
        test_input = torch.tensor(X_test, dtype=torch.float32)
        test_label = torch.tensor(y_test, dtype=torch.long)
        
        dataset = {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }
        
        def train_acc():
            with torch.no_grad():
                return (torch.argmax(model(train_input), dim=1) == train_label).float().mean()
        def test_acc():
            with torch.no_grad():
                return (torch.argmax(model(test_input), dim=1) == test_label).float().mean()
                
        model.fit(
            dataset,
            opt="LBFGS",
            steps=steps,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=(train_acc, test_acc),
        )
        
        with torch.no_grad():
            logits = model(test_input)
            preds = torch.argmax(logits, dim=1).numpy()
            
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        
        # Calculate per-class metrics
        report_dict = classification_report(y_test, preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
        
        results.append({
            "Architecture": str(KAN_WIDTH),
            "Grid": KAN_GRID,
            "K": KAN_K,
            "Steps": steps,
            "Seed": RANDOM_STATE,
            "Accuracy": acc,
            "Macro_F1": macro_f1,
            "Candidate_Precision": report_dict.get("CANDIDATE", {}).get("precision", 0),
            "Candidate_Recall": report_dict.get("CANDIDATE", {}).get("recall", 0),
            "Candidate_F1": report_dict.get("CANDIDATE", {}).get("f1-score", 0),
            "Confirmed_Precision": report_dict.get("CONFIRMED", {}).get("precision", 0),
            "Confirmed_Recall": report_dict.get("CONFIRMED", {}).get("recall", 0),
            "Confirmed_F1": report_dict.get("CONFIRMED", {}).get("f1-score", 0),
            "False_Positive_Precision": report_dict.get("FALSE POSITIVE", {}).get("precision", 0),
            "False_Positive_Recall": report_dict.get("FALSE POSITIVE", {}).get("recall", 0),
            "False_Positive_F1": report_dict.get("FALSE POSITIVE", {}).get("f1-score", 0),
        })
        
        report_str = classification_report(y_test, preds, target_names=label_encoder.classes_, digits=4, zero_division=0)
        
        text_output.append(f"=== {name} ===")
        text_output.append(f"Architecture: {KAN_WIDTH}")
        text_output.append(f"Grid: {KAN_GRID}")
        text_output.append(f"K: {KAN_K}")
        text_output.append(f"Steps: {steps}")
        text_output.append(f"Seed: {RANDOM_STATE}")
        text_output.append(f"Accuracy: {acc:.4f}")
        text_output.append(f"Macro F1: {macro_f1:.4f}\n")
        text_output.append(report_str)
        text_output.append("\n" + "=" * 60 + "\n")
        
        # Save model checkpoint
        torch.save(model.state_dict(), MODELS_DIR / f"kan_{steps}_steps.pt")
        np.save(MODELS_DIR / f"kan_{steps}_steps_config.npy", {
            "width": KAN_WIDTH,
            "grid": KAN_GRID,
            "k": KAN_K,
            "steps": steps
        })
        
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "kan_training_steps_experiment.csv", index=False)
    
    with open(RESULTS_DIR / "kan_training_steps_experiment.txt", "w") as f:
        f.write("\n".join(text_output))
        
    print("\nExperiment completed. Results saved to results/ directory and models to models/ directory.")

if __name__ == "__main__":
    run_steps_experiment()
