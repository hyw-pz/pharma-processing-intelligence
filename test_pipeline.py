"""
tests/test_pipeline.py

Basic tests for data cleaning and feature engineering.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_data import generate_batch_data
from src.data.process import clean, add_features, get_feature_cols


@pytest.fixture
def raw_df():
    return generate_batch_data(n_batches=100, seed=0)


# ── generate_data ─────────────────────────────────────────────────────────────

def test_generated_shape(raw_df):
    assert len(raw_df) == 100
    assert "dissolution_pass" in raw_df.columns


def test_label_is_binary(raw_df):
    assert set(raw_df["dissolution_pass"].unique()).issubset({0, 1})


def test_pass_rate_reasonable(raw_df):
    pass_rate = raw_df["dissolution_pass"].mean()
    assert 0.5 < pass_rate < 0.99, f"Unexpected pass rate: {pass_rate:.2%}"


# ── clean ─────────────────────────────────────────────────────────────────────

def test_clean_removes_missing(raw_df):
    df = clean(raw_df)
    from src.data.process import FEATURE_COLS
    assert df[FEATURE_COLS].isnull().sum().sum() == 0


def test_clean_removes_duplicates():
    df = generate_batch_data(50, seed=1)
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # add 5 dupes
    cleaned = clean(df)
    assert len(cleaned) == 50


# ── add_features ──────────────────────────────────────────────────────────────

def test_add_features_creates_expected_cols(raw_df):
    df = clean(raw_df)
    df = add_features(df)
    for col in ["hardness_per_weight", "total_moisture_index", "coating_cv",
                "inlet_temp_deviation", "weight_rsd_alert"]:
        assert col in df.columns, f"Missing: {col}"


def test_weight_rsd_alert_is_binary(raw_df):
    df = clean(raw_df)
    df = add_features(df)
    assert set(df["weight_rsd_alert"].unique()).issubset({0, 1})


def test_coating_cv_nonnegative(raw_df):
    df = clean(raw_df)
    df = add_features(df)
    assert (df["coating_cv"] >= 0).all()


def test_no_nan_after_full_prep(raw_df):
    df = clean(raw_df)
    df = add_features(df)
    cols = get_feature_cols(df)
    assert df[cols].isnull().sum().sum() == 0
