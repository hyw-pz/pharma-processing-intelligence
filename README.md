# pharma-process-intelligence

Exploring whether manufacturing process data can predict tablet dissolution
outcome — a data project I built to understand how ML might support
batch release decisions in pharmaceutical production.

**[▶ Live Demo](https://YOUR-APP.streamlit.app)**  ← replace after deploying to Streamlit Cloud

---

## Background

In tablet manufacturing, dissolution rate is a critical quality attribute (CQA):
it determines how quickly the active ingredient reaches the patient's bloodstream.
Every batch is tested in the lab after production — but results take ~24 hours.

The question this project explores: **can we predict dissolution pass/fail earlier,
using process sensor data collected during manufacturing?**

This approach is aligned with the pharmaceutical industry's Quality by Design (QbD)
framework, which emphasises understanding the relationship between process parameters
(CPPs) and product quality, rather than testing quality in after the fact.

---

## What I Did

**Task 1 — EDA and data quality check** (`src/data/eda.py`)  
The dataset contains ~500 batches with raw material properties, granulation,
compression, and coating parameters. I profiled missing values, checked for
outliers, and plotted distributions split by pass/fail outcome to identify
which parameters visually separated the two groups.

**Task 2 — Data cleaning and feature engineering** (`src/data/process.py`)  
Handled ~3% missing values via median imputation (sensor dropouts), removed
duplicates, and added a handful of domain-informed derived features — for
example, compression force deviation from target, and coating weight gain
coefficient of variation (an indicator of film coat uniformity).

**Task 3 — Predictive model** (`src/models/train.py`)  
Trained an XGBoost classifier to predict dissolution pass/fail, with logistic
regression as a baseline. XGBoost reached ROC-AUC ~0.93 on the test set vs.
~0.78 for the baseline, confirming that the process parameters carry real
predictive signal.

**Task 4 — Explainability** (`src/models/explain.py`)  
Added SHAP feature importance to answer "why is the model flagging this batch?"
Top drivers: compression force deviation, coating weight gain variability, and
API moisture content — all physically interpretable and consistent with domain
knowledge.

**Task 5 — Dashboard** (`src/visualization/dashboard.py`)  
Built a Streamlit dashboard for the team to explore batch risk scores, compare
parameter distributions across pass/fail groups, and review flagged batches.

---

## Results

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | ~0.78 | Baseline |
| XGBoost | ~0.93 | Primary model |

Top predictive features (SHAP):
1. Compression force deviation from target
2. Coating weight gain std (uniformity)
3. API moisture content
4. Total moisture index

---

## Project Structure

```
pharma-process-intelligence/
├── scripts/
│   ├── generate_data.py     # Synthetic batch data generator
│   └── run_pipeline.py      # End-to-end: generate → EDA → train → explain
├── src/
│   ├── data/
│   │   ├── eda.py           # EDA: distributions, correlation, missing values
│   │   └── process.py       # Cleaning, feature engineering, train/test split
│   ├── models/
│   │   ├── train.py         # XGBoost + logistic regression baseline
│   │   └── explain.py       # SHAP feature importance
│   └── visualization/
│       └── dashboard.py     # Streamlit dashboard (self-contained demo)
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── pharma_background.md
├── reports/figures/         # EDA plots and SHAP chart saved here
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/pharma-process-intelligence
cd pharma-process-intelligence
pip install -r requirements.txt

# Run full pipeline (generates data, EDA, trains model, SHAP plots)
python scripts/run_pipeline.py

# Launch dashboard
streamlit run src/visualization/dashboard.py
```

The dashboard also runs standalone without pre-generated data — it generates
a demo dataset and trains a model on startup automatically.

---

## Notes

- Dataset is synthetic, generated to reflect realistic pharmaceutical batch
  manufacturing distributions and CPP→CQA relationships
- Model is a proof-of-concept; productionisation would require validation on
  real batch data, proper calibration, and regulatory review
- `# NOTE:` comments throughout the code flag simplifications made for
  this exploratory version
