"""Collect every artifact into the numbers quoted in the README.

Single source of truth: run this after ingest -> features -> degradation ->
models -> leakage_check, and it prints (and saves) exactly the figures the
README relies on. Nothing here recomputes a model; it only
reads what the pipeline already wrote, so the README can never drift from the
artifacts.
"""
from __future__ import annotations

import json

import pandas as pd

from config import DATA_DIR

pd.set_option("display.width", 200)


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    print("=" * 78)
    print("DATASET")
    print("=" * 78)
    n_races = df.groupby(["year", "round"]).ngroups
    print(f"seasons              : {sorted(df['year'].unique().tolist())}")
    print(f"races (dry, modelled): {n_races}")
    print(f"driver-laps          : {len(df):,}")
    print(f"drivers              : {df['driver'].nunique()}")
    print(f"circuits             : {df['circuit'].nunique()}")
    print(f"telemetry coverage   : {df['max_speed'].notna().mean()*100:.1f}%")
    print(f"pit-window base rate : {df['pits_within_horizon'].mean()*100:.2f}%")
    print("\nper season:")
    print(df.groupby("year").agg(races=("round", "nunique"),
                                 laps=("lap_number", "size"),
                                 base_rate=("pits_within_horizon", "mean")).round(4))

    for name, f in [("LAP ATTRITION", "lap_attrition.csv"),
                    ("DEGRADATION SLOPES", "degradation_slopes.csv"),
                    ("EXPANDING-WINDOW CV", "expanding_window_cv.csv")]:
        p = DATA_DIR / f
        if p.exists():
            print("\n" + "=" * 78 + f"\n{name}\n" + "=" * 78)
            print(pd.read_csv(p).to_string(index=False))

    p = DATA_DIR / "model_metrics.json"
    if p.exists():
        m = json.load(open(p))
        print("\n" + "=" * 78 + "\nMODEL METRICS\n" + "=" * 78)
        print(json.dumps(m, indent=2))

    p = DATA_DIR / "leakage_checklist.csv"
    if p.exists():
        print("\n" + "=" * 78 + "\nLEAKAGE CHECKLIST\n" + "=" * 78)
        lc = pd.read_csv(p)
        print(lc["status"].value_counts().to_string())
        for _, r in lc.iterrows():
            print(f"  [{r['status']}] {r['item']}")


if __name__ == "__main__":
    main()
