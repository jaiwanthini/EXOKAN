# Interpretable Exoplanet Candidate Classification using Kolmogorov-Arnold Networks (KAN)

## Problem Statement
NASA's Kepler and TESS missions flag thousands of Kepler Objects of Interest (KOIs) —
potential exoplanet detections based on the transit method. Most of these are false
positives (eclipsing binaries, instrumental noise, stellar variability), and separating
real candidates from noise currently relies on complex, largely manual vetting pipelines
that don't scale.

Black-box ML models can classify these quickly, but astronomers need to understand
*why* a detection was flagged a certain way before trusting or acting on it — especially
before committing expensive follow-up telescope time (e.g., JWST).

## Solution
This project trains a **Kolmogorov-Arnold Network (KAN)** — a 2024 neural architecture
that is interpretable *by design* (each edge learns an explicit, visualizable function,
rather than a fixed weight + activation) — to classify KOIs as `CONFIRMED`, `CANDIDATE`,
or `FALSE POSITIVE` based on stellar and transit measurements.

KAN performance and interpretability are benchmarked against Random Forest and XGBoost
baselines (explained post-hoc via SHAP), directly contrasting "interpretable by design"
vs. "explained after the fact."

## Dataset
[NASA Exoplanet Archive — Kepler Objects of Interest (KOI) cumulative table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
(public, ~9,500 rows, tabular). Download as CSV and place at `data/raw/koi_cumulative.csv`.

## Project Structure
```
exoplanet-kan-classifier/
├── data/
│   ├── raw/              # place koi_cumulative.csv here
│   └── processed/
├── src/
│   ├── data_loader.py    # loads raw CSV
│   ├── preprocess.py     # cleaning, feature selection, train/test split
│   ├── train_kan.py      # trains the KAN model
│   ├── train_baselines.py# trains Random Forest + XGBoost
│   └── interpret.py      # KAN activation plots + SHAP comparison
├── app/
│   └── streamlit_app.py  # interactive demo
├── models/                # saved trained models (generated)
├── results/                # metrics, plots (generated)
├── notebooks/                # exploratory analysis
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage
```bash
# 1. Download data (see Dataset section) into data/raw/koi_cumulative.csv

# 2. Train baselines
python src/train_baselines.py

# 3. Train KAN
python src/train_kan.py

# 4. Generate interpretability plots
python src/interpret.py

# 5. Run the demo app
streamlit run app/streamlit_app.py
```

## Results
_(Fill in after running the scripts)_

| Model | Macro F1 | Accuracy |
|---|---|---|
| Random Forest | | |
| XGBoost | | |
| KAN | | |

## Key Takeaway
_(Fill in your finding, e.g.: "KAN achieved competitive accuracy with RF/XGBoost while
providing direct visualization of learned decision functions — showing that features
like koi_model_snr and koi_depth have [describe the learned relationship] on
classification, consistent with known transit-detection physics.")_

## Tech Stack
Python, PyTorch, pykan, scikit-learn, XGBoost, SHAP, Streamlit, pandas, matplotlib
