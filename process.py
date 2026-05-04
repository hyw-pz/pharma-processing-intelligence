"""
src/data/process.py

Data cleaning and feature engineering for the batch dataset.

Two steps combined intentionally — at this project scale it would be
over-engineering to split them into separate pipeline classes.

# NOTE: in production this would be part of a versioned pipeline with
# proper train/test imputation separation (fit on train, apply to test).
# Here we keep it simple: median imputation on the full dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

FEATURE_COLS = [
    "api_d50_um", "api_d90_um", "api_moisture_pct", "api_bulk_density_g_ml",
    "excipient_moisture_pct", "binder_conc_pct", "wet_mass_torque_nm",
    "granule_d50_um", "granule_moisture_pct", "comp_force_kn",
    "comp_force_dev_kn", "pre_comp_force_kn", "punch_speed_rpm",
    "tablet_weight_mg", "tablet_weight_rsd_pct", "tablet_hardness_n",
    "inlet_temp_c", "spray_rate_g_min", "pan_speed_rpm",
    "coating_weight_gain_pct", "coating_weight_gain_std_pct",
]

TARGET_COL = "dissolution_pass"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - Impute missing values with column median
      - Remove exact duplicate rows
      - Flag rows with suspicious values for review
    """
    df = df.copy()

    # Median imputation for sensor dropout columns
    n_missing = df[FEATURE_COLS].isnull().sum().sum()
    if n_missing > 0:
        medians = df[FEATURE_COLS].median()
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(medians)
        print(f"  Imputed {n_missing} missing values (median)")

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=FEATURE_COLS).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Removed {n_before - len(df)} duplicate rows")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that encode domain knowledge about tablet manufacturing.
    Each feature has a comment explaining why it might matter for dissolution.
    """
    df = df.copy()

    # Compression force deviation from 12 kN target
    # Over/under-compression both affect tablet porosity → dissolution rate
    df["comp_force_dev_kn"] = np.abs(df["comp_force_kn"] - 12.0)

    # Tablet hardness normalized by weight — proxy for compaction quality
    df["hardness_per_weight"] = df["tablet_hardness_n"] / df["tablet_weight_mg"]

    # Total moisture burden entering the process
    # High combined moisture can cause dense granules → slower dissolution
    df["total_moisture_index"] = (
        df["api_moisture_pct"] * 0.4
        + df["excipient_moisture_pct"] * 0.3
        + df["granule_moisture_pct"] * 0.3
    )

    # Coating weight gain coefficient of variation
    # High variability → uneven film coat → inconsistent dissolution barrier
    df["coating_cv"] = (
        df["coating_weight_gain_std_pct"]
        / df["coating_weight_gain_pct"].clip(lower=0.01)
    )

    # Inlet temperature deviation from 55°C optimum
    # Too hot → film defects; too cold → poor film formation
    df["inlet_temp_deviation"] = np.abs(df["inlet_temp_c"] - 55.0)

    # Binary alert: tablet weight RSD > 1% (typical manufacturing alert limit)
    df["weight_rsd_alert"] = (df["tablet_weight_rsd_pct"] > 1.0).astype(int)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (original + engineered) present in df."""
    engineered = [
        "hardness_per_weight", "total_moisture_index",
        "coating_cv", "inlet_temp_deviation", "weight_rsd_alert",
    ]
    return [c for c in FEATURE_COLS + engineered if c in df.columns]


def prepare_data(
    data_path: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full preparation: load → clean → add features → train/test split.
    Returns (X_train, X_test, y_train, y_test).
    """
    df = pd.read_csv(data_path, parse_dates=["batch_date"])
    print(f"Loaded {len(df)} batches")

    df = clean(df)
    df = add_features(df)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  OOS rate — train: {1-y_train.mean():.1%}, test: {1-y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test
