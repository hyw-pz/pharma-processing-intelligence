"""
src/models/explain.py

SHAP feature importance for the trained XGBoost model.

Added this after the team asked "why is the model flagging these batches?" —
a fair question given that the results need to be interpretable by quality
scientists, not just data people.

Outputs:
  - reports/figures/shap_importance.png   (global feature importance bar chart)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FIGURES_DIR = Path("reports/figures")


def plot_shap_importance(
    model,
    X: pd.DataFrame,
    top_n: int = 12,
    save: bool = True,
) -> plt.Figure:
    """
    Bar chart of mean |SHAP| values — shows which features drive OOS predictions most.

    Using TreeExplainer (fast, exact for XGBoost).
    Taking class 0 (OOS) SHAP values so the chart reads as "contribution to OOS risk".
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # For binary classification XGBoost returns shape (n, features, 2)
    # Take class 0 = OOS
    if shap_values.values.ndim == 3:
        vals = shap_values.values[:, :, 0]
    else:
        vals = shap_values.values

    mean_abs_shap = pd.Series(
        np.abs(vals).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    mean_abs_shap.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (contribution to OOS prediction)")
    ax.set_title("Feature Importance — XGBoost (SHAP)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {FIGURES_DIR}/shap_importance.png")

    return fig


def get_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Return raw SHAP values array (n_samples x n_features) for class 0 (OOS)."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)
    if shap_values.values.ndim == 3:
        return shap_values.values[:, :, 0]
    return shap_values.values
