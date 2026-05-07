"""
scripts/run_pipeline.py

Run the full pipeline: generate data → EDA → clean → train → explain.

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_data import generate_batch_data
from src.data.eda import run_eda
from src.data.process import prepare_data
from src.models.train import run_training
from src.models.explain import plot_shap_importance

DATA_PATH  = "data/raw/batches.csv"
MODEL_DIR  = "models/"


def main():
    # 1. Generate data (skip if already exists)
    if not Path(DATA_PATH).exists():
        print("── Step 1: Generating synthetic batch data ──────────────")
        df = generate_batch_data(n_batches=500)
        Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved → {DATA_PATH}\n")
    else:
        print(f"── Step 1: Data already exists at {DATA_PATH}, skipping generation\n")

    # 2. EDA
    print("── Step 2: Exploratory Data Analysis ───────────────────")
    run_eda(DATA_PATH)
    print()

    # 3. Clean + feature engineering + split
    print("── Step 3: Data preparation ─────────────────────────────")
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    print()

    # 4. Train
    print("── Step 4: Model training ───────────────────────────────")
    models = run_training(X_train, y_train, X_test, y_test, output_dir=MODEL_DIR)
    print()

    # 5. SHAP feature importance
    print("── Step 5: SHAP explainability ──────────────────────────")
    plot_shap_importance(models["xgboost"], X_test)

    print("\nPipeline complete.")
    print("  → EDA plots and SHAP chart saved to reports/figures/")
    print("  → Model saved to models/xgboost.pkl")
    print("  → Launch dashboard: streamlit run src/visualization/dashboard.py")


if __name__ == "__main__":
    main()
