"""
src/data/eda.py

Exploratory data analysis for the batch manufacturing dataset.
Produces a data quality summary table and key distribution plots.

This was the first task I was given: "explore the dataset, flag any data
quality issues, and identify which parameters correlate with dissolution failures."

Outputs (saved to reports/figures/):
  - data_quality_summary.csv
  - eda_distributions.png    (feature distributions, PASS vs OOS)
  - eda_correlation.png      (correlation heatmap)
  - eda_missing.png          (missing value overview)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path("reports/figures")

FEATURE_COLS = [
    "api_moisture_pct", "api_d50_um", "api_bulk_density_g_ml",
    "granule_moisture_pct", "binder_conc_pct",
    "comp_force_kn", "comp_force_dev_kn", "tablet_hardness_n", "tablet_weight_rsd_pct",
    "inlet_temp_c", "coating_weight_gain_pct", "coating_weight_gain_std_pct",
]


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column summary: missing rate, mean, std, min, max.
    Saved as CSV for team review.
    """
    records = []
    for col in df.select_dtypes(include="number").columns:
        records.append({
            "column":      col,
            "n_total":     len(df),
            "n_missing":   int(df[col].isnull().sum()),
            "pct_missing": round(df[col].isnull().mean() * 100, 2),
            "mean":        round(df[col].mean(), 4),
            "std":         round(df[col].std(), 4),
            "min":         round(df[col].min(), 4),
            "max":         round(df[col].max(), 4),
        })
    return pd.DataFrame(records)


def plot_distributions(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Overlay distributions of key CPPs for PASS vs OOS batches.
    Helps visually identify which parameters separate the two groups.
    """
    cols_to_plot = [c for c in FEATURE_COLS if c in df.columns]
    n = len(cols_to_plot)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2))
    axes = axes.flatten()

    pass_df = df[df["dissolution_pass"] == 1]
    oos_df  = df[df["dissolution_pass"] == 0]

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        ax.hist(pass_df[col].dropna(), bins=25, alpha=0.6, color="#2ecc71", label="PASS", density=True)
        ax.hist(oos_df[col].dropna(),  bins=25, alpha=0.6, color="#e74c3c", label="OOS",  density=True)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Parameter Distributions: PASS vs OOS Batches", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "eda_distributions.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {FIGURES_DIR}/eda_distributions.png")

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Correlation heatmap of process parameters + dissolution_pass.
    Useful for quickly spotting which CPPs are most linearly associated with the outcome.

    NOTE: low correlation doesn't mean a feature is unimportant —
    tree models can capture non-linear relationships. This is just a starting point.
    """
    cols = [c for c in FEATURE_COLS if c in df.columns] + ["dissolution_pass"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle only
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 7},
    )
    ax.set_title("Parameter Correlation Matrix (incl. dissolution_pass)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "eda_correlation.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {FIGURES_DIR}/eda_correlation.png")

    return fig


def plot_missing_values(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bar chart of missing value rates per column."""
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", ax=ax, color="#e67e22", edgecolor="white")
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing Value Rate by Column", fontweight="bold")
    ax.axhline(5, color="red", linestyle="--", linewidth=1, label="5% threshold")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "eda_missing.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {FIGURES_DIR}/eda_missing.png")

    return fig


def run_eda(data_path: str = "data/raw/batches.csv"):
    """Run full EDA and save all outputs."""
    df = pd.read_csv(data_path, parse_dates=["batch_date"])
    print(f"Loaded {len(df)} batches, {len(df.columns)} columns")
    print(f"OOS rate: {1 - df['dissolution_pass'].mean():.1%}\n")

    # 1. Data quality summary
    summary = data_quality_summary(df)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(FIGURES_DIR / "data_quality_summary.csv", index=False)
    print("── Data Quality Summary ──────────────────────")
    print(summary[summary["n_missing"] > 0][["column", "n_missing", "pct_missing"]].to_string(index=False))
    print()

    # 2. Plots
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_missing_values(df)

    print("\nEDA complete. Outputs saved to reports/figures/")


if __name__ == "__main__":
    run_eda()
