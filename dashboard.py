"""
src/visualization/dashboard.py

Batch quality monitoring dashboard.

Designed to be self-contained for demo purposes:
if no pre-trained model or data is found, it generates synthetic data
and trains a model on the fly. This means the live demo on Streamlit Cloud
works without any setup.

Run locally:
    streamlit run src/visualization/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root when running via streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pharma Process Intelligence",
    page_icon="💊",
    layout="wide",
)

st.title("💊 Pharma Process Intelligence")
st.caption(
    "Batch quality monitoring dashboard — predicts dissolution pass/fail "
    "from manufacturing process parameters."
)

# ── Load or generate data + model ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Setting up demo data and model...")
def get_demo_data_and_model():
    """
    Self-contained setup for the live demo:
    generate synthetic data, train XGBoost, return everything needed for the dashboard.
    """
    from scripts.generate_data import generate_batch_data
    from src.data.process import clean, add_features, get_feature_cols
    from src.models.train import train_xgboost
    from src.models.explain import get_shap_values

    df = generate_batch_data(n_batches=500, seed=42)
    df = clean(df)
    df = add_features(df)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df["dissolution_pass"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_xgboost(X_train, y_train)

    # Score full dataset for display
    df["oos_prob"]     = model.predict_proba(X)[:, 0]
    df["predicted_oos"] = (df["oos_prob"] >= 0.5).astype(int)

    # SHAP values
    shap_vals = get_shap_values(model, X_test)
    shap_df   = pd.DataFrame(
        np.abs(shap_vals).mean(axis=0),
        index=feature_cols,
        columns=["mean_abs_shap"],
    ).sort_values("mean_abs_shap", ascending=False)

    return df, model, feature_cols, shap_df


df, model, feature_cols, shap_df = get_demo_data_and_model()

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.header("Controls")
threshold = st.sidebar.slider("OOS alert threshold", 0.1, 0.9, 0.5, 0.05,
                               help="Batches with OOS probability above this are flagged.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\nThis dashboard is part of a project exploring whether "
    "process sensor data can predict tablet dissolution outcome before "
    "lab results are available (~24h lead time)."
)

# Apply threshold
df["flagged"] = (df["oos_prob"] >= threshold).astype(int)

# ── KPI row ───────────────────────────────────────────────────────────────────

n_total    = len(df)
n_flagged  = df["flagged"].sum()
n_oos_true = (df["dissolution_pass"] == 0).sum()
avg_risk   = df["oos_prob"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Batches",       n_total)
col2.metric("Flagged (predicted)", n_flagged,
            delta=f"{n_flagged/n_total:.1%}", delta_color="inverse")
col3.metric("Actual OOS (label)",  n_oos_true,
            delta=f"{n_oos_true/n_total:.1%}", delta_color="inverse")
col4.metric("Avg OOS Risk",        f"{avg_risk:.3f}", delta_color="inverse")

st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────

left, right = st.columns([2, 1])

with left:
    st.subheader("OOS Risk Score — All Batches")
    fig_timeline = px.scatter(
        df, x="batch_date", y="oos_prob",
        color=df["flagged"].map({0: "PASS", 1: "⚠️ Flagged"}),
        color_discrete_map={"PASS": "#2ecc71", "⚠️ Flagged": "#e74c3c"},
        hover_data=["batch_id"],
        opacity=0.65,
        labels={"oos_prob": "OOS Risk Score", "batch_date": "Batch Date"},
    )
    fig_timeline.add_hline(
        y=threshold, line_dash="dash", line_color="orange",
        annotation_text=f"threshold = {threshold}",
    )
    fig_timeline.update_layout(height=320, showlegend=True, margin=dict(t=20))
    st.plotly_chart(fig_timeline, use_container_width=True)

with right:
    st.subheader("Feature Importance (SHAP)")
    top_shap = shap_df.head(10).reset_index()
    top_shap.columns = ["feature", "mean_abs_shap"]
    fig_shap = px.bar(
        top_shap.sort_values("mean_abs_shap"),
        x="mean_abs_shap", y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Blues",
        labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
    )
    fig_shap.update_layout(
        height=320, showlegend=False,
        coloraxis_showscale=False, margin=dict(t=20)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# ── Parameter explorer ────────────────────────────────────────────────────────

st.subheader("Parameter Explorer")
st.caption("Select a process parameter to compare its distribution across predicted PASS vs. flagged batches.")

numeric_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in ["dissolution_pass", "flagged", "predicted_oos"]]

selected = st.selectbox(
    "Process parameter",
    numeric_cols,
    index=numeric_cols.index("comp_force_dev_kn") if "comp_force_dev_kn" in numeric_cols else 0,
)

c1, c2 = st.columns(2)

with c1:
    fig_hist = px.histogram(
        df, x=selected,
        color=df["flagged"].map({0: "PASS", 1: "Flagged"}),
        barmode="overlay", nbins=30, opacity=0.7,
        color_discrete_map={"PASS": "#2ecc71", "Flagged": "#e74c3c"},
        title=f"Distribution of {selected}",
    )
    fig_hist.update_layout(height=300, margin=dict(t=40))
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    fig_box = px.box(
        df, x=df["flagged"].map({0: "PASS", 1: "Flagged"}),
        y=selected,
        color=df["flagged"].map({0: "PASS", 1: "Flagged"}),
        color_discrete_map={"PASS": "#2ecc71", "Flagged": "#e74c3c"},
        title=f"{selected} — PASS vs Flagged",
        points="outliers",
    )
    fig_box.update_layout(height=300, showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Flagged batches table ─────────────────────────────────────────────────────

st.subheader(f"Flagged Batches ({n_flagged} total)")

display_cols = ["batch_id", "batch_date", "oos_prob", "dissolution_pass",
                "comp_force_dev_kn", "coating_weight_gain_std_pct", "api_moisture_pct"]
display_cols = [c for c in display_cols if c in df.columns]

flagged_df = (
    df[df["flagged"] == 1][display_cols]
    .sort_values("oos_prob", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

st.dataframe(
    flagged_df.style.format({
        "oos_prob":                    "{:.3f}",
        "comp_force_dev_kn":           "{:.2f}",
        "coating_weight_gain_std_pct": "{:.3f}",
        "api_moisture_pct":            "{:.2f}",
    }).background_gradient(subset=["oos_prob"], cmap="Reds"),
    use_container_width=True,
)

st.caption("dissolution_pass = 1 means the batch actually passed lab testing (ground truth label).")
