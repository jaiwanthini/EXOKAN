"""
data_loader.py

Loads the NASA Exoplanet Archive - Kepler Objects of Interest (KOI) cumulative table.

How to get the raw data:
1. Go to: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
2. Click "Download Table" -> CSV
3. Save it as:
   data/raw/koi_cumulative.csv

This script only loads the raw data and performs a basic sanity check.
Cleaning, feature engineering, and train/test splitting are handled in preprocess.py.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "koi_cumulative.csv"
)


def load_raw_koi_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw KOI dataset.
    Skip NASA comment lines that start with '#'.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{path}\n\n"
            "Download the KOI cumulative table from the NASA Exoplanet Archive "
            "and place it inside data/raw/koi_cumulative.csv"
        )

    df = pd.read_csv(path, comment="#")
    return df


def basic_summary(df: pd.DataFrame) -> None:
    """Print a basic overview of the dataset."""

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nTarget Distribution (koi_disposition):")
    if "koi_disposition" in df.columns:
        print(df["koi_disposition"].value_counts())

    print("\nTop 10 Columns with Missing Values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nTotal Number of Columns:", len(df.columns))

    print("\nColumn Names:")
    for col in df.columns:
        print(col)


if __name__ == "__main__":
    df = load_raw_koi_data()
    basic_summary(df)