"""
train_kan.py

Trains a Kolmogorov-Arnold Network (KAN) on the KOI dataset using pykan.

Notes for CPU training:
- Keep the network small (a couple of hidden nodes) - KANs have more
  learnable parameters per edge than a standard MLP, so they can be
  slower to train. Start small, increase width/grid only if needed.
- pykan works with torch tensors, and its `.fit()` method wraps the
  training loop (LBFGS optimizer by default, which converges fast on
  small tabular data - a good fit for CPU-only training).
"""

import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from kan import KAN

from preprocess import clean_and_split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def to_pykan_dataset(X_train, X_test, y_train, y_test):
    """pykan expects a dict of torch tensors with these exact keys."""
    return {
        "train_input": torch.tensor(X_train, dtype=torch.float32),
        "train_label": torch.tensor(y_train, dtype=torch.long),
        "test_input": torch.tensor(X_test, dtype=torch.float32),
        "test_label": torch.tensor(y_test, dtype=torch.long),
    }


def train_kan_model(steps: int = 100):
    X_train, X_test, y_train, y_test, label_encoder, features = clean_and_split()
    dataset = to_pykan_dataset(X_train, X_test, y_train, y_test)

    n_features = X_train.shape[1]
    n_classes = len(label_encoder.classes_)

    # width: [input_dim, hidden_dim, output_dim]. Start small on CPU.
    model = KAN(width=[n_features, 6, n_classes], grid=8, k=3, seed=42)

    def train_acc():
        preds = torch.argmax(model(dataset["train_input"]), dim=1)
        return (preds == dataset["train_label"]).float().mean()

    def test_acc():
        preds = torch.argmax(model(dataset["test_input"]), dim=1)
        return (preds == dataset["test_label"]).float().mean()

    print("Training KAN...")
    import torch.nn as nn
    model.fit(
    dataset,
    opt="LBFGS",
    steps=steps,
    loss_fn=nn.CrossEntropyLoss(),
    metrics=(train_acc, test_acc)
)

    with torch.no_grad():
        test_logits = model(dataset["test_input"])
        test_preds = torch.argmax(test_logits, dim=1).numpy()

    report = classification_report(y_test, test_preds, target_names=label_encoder.classes_)
    macro_f1 = f1_score(y_test, test_preds, average="macro")

    print(f"\nKAN Macro F1: {macro_f1:.4f}")
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "kan_results.txt", "w") as f:
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(report)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "kan_model.pt")

    # Save the model's own width/config too, since you need it to reload
    # the KAN class correctly later (plain state_dict isn't enough for KAN).
    np.save(MODELS_DIR / "kan_config.npy", {"width": [n_features, 6, n_classes], "grid": 8, "k": 3})

    return model, macro_f1


if __name__ == "__main__":
    train_kan_model()
