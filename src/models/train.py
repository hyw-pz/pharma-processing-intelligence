"""
src/models/train.py

Train an XGBoost classifier to predict dissolution pass/fail,
with a logistic regression baseline for comparison.

Task framing: the team wanted to explore whether process sensor data
collected during manufacturing could predict dissolution outcome before
the lab test comes back (which takes ~24h). This script is the first
working version — results to be reviewed with the team.

# NOTE: threshold tuning and model calibration are left as next steps.
# For now, default threshold (0.5) is used and results reviewed qualitatively.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_SEED = 42


def train_baseline(X_train, y_train) -> Pipeline:
    """Logistic regression baseline — fast, interpretable, good sanity check."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    XGBoost classifier. Parameters chosen based on dataset size (~400 train rows)
    and class imbalance (~18% OOS). scale_pos_weight compensates for imbalance.

    # NOTE: these hyperparameters were set manually based on general guidance
    # for small-to-medium tabular datasets. A proper grid search would be the
    # next step if the team decides to productionize this model.
    """
    neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        base_score=0.5,
        scale_pos_weight=neg_pos_ratio,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Print classification report and return key metrics as dict."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n── {model_name} ──────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["OOS (0)", "PASS (1)"]))
    print(f"ROC-AUC: {auc:.4f}")

    return {
        "model_name": model_name,
        "roc_auc":    round(auc, 4),
    }


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "models/",
) -> dict:
    """
    Train both models, evaluate, save XGBoost as the primary model.
    Returns dict with both models and their metrics.
    """
    print("\nTraining logistic regression baseline...")
    lr    = train_baseline(X_train, y_train)
    m_lr  = evaluate(lr, X_test, y_test, "Logistic Regression (baseline)")

    print("\nTraining XGBoost...")
    xgb   = train_xgboost(X_train, y_train)
    m_xgb = evaluate(xgb, X_test, y_test, "XGBoost")

    save_model(xgb, f"{output_dir}/xgboost.pkl")

    print(f"\nSummary: LR AUC = {m_lr['roc_auc']:.4f}  |  XGB AUC = {m_xgb['roc_auc']:.4f}")

    return {"logistic_regression": lr, "xgboost": xgb}
