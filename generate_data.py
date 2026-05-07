"""
scripts/generate_data.py

Generates a synthetic pharmaceutical batch manufacturing dataset.
Each row = one production batch.

Run:
    python scripts/generate_data.py --n_batches 500 --output data/raw/batches.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42


def generate_batch_data(n_batches: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Raw Material Properties ──────────────────────────────────────────────
    api_d50             = rng.normal(50,  8,   n_batches).clip(20, 90)
    api_d90             = api_d50 * rng.uniform(1.5, 2.2, n_batches)
    api_moisture        = rng.normal(2.5, 0.6, n_batches).clip(0.5, 5.0)
    api_bulk_density    = rng.normal(0.45, 0.06, n_batches).clip(0.25, 0.70)
    excipient_moisture  = rng.normal(3.0, 0.5, n_batches).clip(1.0, 6.0)

    # ── Granulation ──────────────────────────────────────────────────────────
    binder_conc         = rng.normal(4.0, 0.5, n_batches).clip(2.0, 6.5)
    wet_mass_torque     = rng.normal(1.8, 0.3, n_batches).clip(0.8, 3.5)
    granule_d50         = rng.normal(200, 30,  n_batches).clip(100, 350)
    granule_moisture    = rng.normal(2.0, 0.4, n_batches).clip(0.5, 4.5)

    # ── Compression ──────────────────────────────────────────────────────────
    comp_force          = rng.normal(12.0, 1.2, n_batches).clip(6, 20)
    comp_force_dev      = np.abs(comp_force - 12.0)
    pre_comp_force      = rng.normal(2.0, 0.3, n_batches).clip(0.5, 5.0)
    punch_speed         = rng.normal(35,  5,   n_batches).clip(15, 60)
    tablet_weight       = rng.normal(250, 5,   n_batches).clip(230, 270)
    tablet_weight_rsd   = rng.exponential(0.5, n_batches).clip(0.1, 3.0)
    tablet_hardness     = rng.normal(80,  10,  n_batches).clip(40, 130)

    # ── Coating ──────────────────────────────────────────────────────────────
    inlet_temp              = rng.normal(55,  4,   n_batches).clip(40, 75)
    spray_rate              = rng.normal(15,  2,   n_batches).clip(8, 25)
    pan_speed               = rng.normal(12,  1.5, n_batches).clip(6, 20)
    coating_weight_gain     = rng.normal(3.0, 0.4, n_batches).clip(1.5, 5.0)
    coating_weight_gain_std = rng.exponential(0.15, n_batches).clip(0.02, 0.8)

    # ── Label: dissolution_pass ───────────────────────────────────────────────
    log_odds = (
        2.5
        - 0.8  * comp_force_dev
        - 0.6  * (api_moisture - 2.5)
        - 2.0  * coating_weight_gain_std
        + 0.04 * (tablet_hardness - 80)
        - 0.5  * (granule_moisture - 2.0)
        + rng.normal(0, 0.4, n_batches)
    )
    prob_pass        = 1 / (1 + np.exp(-log_odds))
    dissolution_pass = (rng.uniform(size=n_batches) < prob_pass).astype(int)

    df = pd.DataFrame({
        "batch_id":                    [f"BATCH-{i:04d}" for i in range(n_batches)],
        "batch_date":                  pd.date_range("2023-01-01", periods=n_batches, freq="8h"),
        "api_d50_um":                  api_d50.round(2),
        "api_d90_um":                  api_d90.round(2),
        "api_moisture_pct":            api_moisture.round(3),
        "api_bulk_density_g_ml":       api_bulk_density.round(4),
        "excipient_moisture_pct":      excipient_moisture.round(3),
        "binder_conc_pct":             binder_conc.round(3),
        "wet_mass_torque_nm":          wet_mass_torque.round(3),
        "granule_d50_um":              granule_d50.round(2),
        "granule_moisture_pct":        granule_moisture.round(3),
        "comp_force_kn":               comp_force.round(3),
        "comp_force_dev_kn":           comp_force_dev.round(3),
        "pre_comp_force_kn":           pre_comp_force.round(3),
        "punch_speed_rpm":             punch_speed.round(1),
        "tablet_weight_mg":            tablet_weight.round(2),
        "tablet_weight_rsd_pct":       tablet_weight_rsd.round(3),
        "tablet_hardness_n":           tablet_hardness.round(2),
        "inlet_temp_c":                inlet_temp.round(2),
        "spray_rate_g_min":            spray_rate.round(3),
        "pan_speed_rpm":               pan_speed.round(2),
        "coating_weight_gain_pct":     coating_weight_gain.round(3),
        "coating_weight_gain_std_pct": coating_weight_gain_std.round(4),
        "dissolution_pass":            dissolution_pass,
    })

    # Realistic sensor dropouts (~3% missing in a few columns)
    for col in ["wet_mass_torque_nm", "inlet_temp_c", "coating_weight_gain_std_pct"]:
        df.loc[rng.uniform(size=n_batches) < 0.03, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches", type=int, default=500)
    parser.add_argument("--output",    type=str, default="data/raw/batches.csv")
    parser.add_argument("--seed",      type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = generate_batch_data(args.n_batches, args.seed)
    df.to_csv(args.output, index=False)

    n_pass = df["dissolution_pass"].sum()
    print(f"Generated {len(df)} batches → {args.output}")
    print(f"  Pass: {n_pass} ({n_pass/len(df):.1%})  |  OOS: {len(df)-n_pass} ({1-n_pass/len(df):.1%})")
    print(f"  Missing values: {df.isnull().sum().sum()} cells")


if __name__ == "__main__":
    main()
