"""Two gradient-boosting models, both validated on time-based season splits.

  1. Pit-window classifier  - given the state at lap L, does this driver pit
     within the next 3 laps?  HistGradientBoostingClassifier.
  2. Finishing-position regressor - final classified position per driver-race.
     HistGradientBoostingRegressor, benchmarked against "predict grid position".

There is no random train_test_split anywhere in this file. Splits are by season:
train on earlier seasons, test on later ones, never the reverse.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             precision_recall_fscore_support, r2_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from config import (DATA_DIR, MODEL_DIR, PIT_HORIZON, RANDOM_STATE,
                    TEST_SEASONS, TRAIN_SEASONS, VALID_SEASONS)

# ---------------------------------------------------------------- pit window
# Every feature here is knowable by a strategist standing on the pit wall at
# lap L. Total stint length is absent on purpose: it is what the model predicts.
PIT_NUM = [
    "tyre_life", "lap_number", "laps_remaining", "race_progress",
    "lap_time_vs_field_baseline", "pace_last3", "pace_trend_3",
    "lap_time_delta_prev", "position", "grid_position", "position_vs_grid",
    "stops_so_far", "stint_number", "track_temp", "air_temp", "humidity",
    "wind_speed", "track_evolution_gain_s", "fuel_remaining_kg",
    "max_speed", "mean_throttle", "brake_pct", "drs_pct", "full_throttle_pct",
]
PIT_CAT = ["compound", "circuit", "team_name"]
PIT_TARGET = "pits_within_horizon"

# ---------------------------------------------------------------- finishing
FIN_NUM = [
    "grid_position", "n_stops", "n_green_laps", "mean_pace_vs_field",
    "pace_std", "mean_tyre_life", "max_tyre_life", "mean_track_temp",
    "total_laps", "n_compounds_used", "mean_max_speed", "mean_throttle_race",
    "soft_share", "medium_share", "hard_share", "deg_exposure",
]
FIN_CAT = ["circuit", "team_name"]
FIN_TARGET = "final_position"

# Strict pre-race variant: only what is known before lights out (grid slot,
# team, circuit). Reported next to the post-race summary model so the two are
# not confused.
FIN_PRERACE_NUM = ["grid_position", "total_laps"]
FIN_PRERACE_CAT = ["circuit", "team_name"]


def make_pipeline(kind: str, num, cat, **kw) -> Pipeline:
    """Preprocessing + model in one Pipeline.

    The Pipeline matters for more than tidiness: imputers and the ordinal
    encoder are fit inside whatever data they are given, so when we call
    .fit(train) they never see the test seasons.
    """
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
            ]), cat),
        ],
        remainder="drop",
    )
    cat_idx = list(range(len(num), len(num) + len(cat)))
    if kind == "clf":
        est = HistGradientBoostingClassifier(
            categorical_features=cat_idx, random_state=RANDOM_STATE, **kw)
    else:
        est = HistGradientBoostingRegressor(
            categorical_features=cat_idx, random_state=RANDOM_STATE, **kw)
    return Pipeline([("pre", pre), ("model", est)])


# ============================================================ PIT WINDOW
def pit_window_model(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=[PIT_TARGET]).copy()
    cols = PIT_NUM + PIT_CAT

    tr = d[d["year"].isin(TRAIN_SEASONS)]
    va = d[d["year"].isin(VALID_SEASONS)]
    te = d[d["year"].isin(TEST_SEASONS)]

    print(f"\n{'='*72}\nPIT-WINDOW CLASSIFIER  (pit within next {PIT_HORIZON} laps?)")
    print(f"{'='*72}")
    print(f"train {TRAIN_SEASONS}: {len(tr):,} laps | "
          f"valid {VALID_SEASONS}: {len(va):,} | test {TEST_SEASONS}: {len(te):,}")

    base_rate = te[PIT_TARGET].mean()
    print(f"\n*** BASE RATE (test seasons): {base_rate*100:.2f}% of laps are "
          f"positive. This is a heavily imbalanced problem - accuracy is\n"
          f"    meaningless here, so ROC-AUC, PR-AUC and precision/recall are "
          f"reported instead. A no-skill PR-AUC equals the base rate "
          f"({base_rate:.4f}).")

    def _new():
        return make_pipeline("clf", PIT_NUM, PIT_CAT,
                             max_iter=400, learning_rate=0.06,
                             max_leaf_nodes=31, l2_regularization=1.0,
                             early_stopping=False)

    # Stage 1: fit on TRAIN only and score the VALIDATION season. The decision
    # threshold is chosen here and nowhere else.
    pipe = _new()
    pipe.fit(tr[cols], tr[PIT_TARGET])

    out = {"base_rate_test": float(base_rate),
           "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
           "train_seasons": TRAIN_SEASONS, "valid_seasons": VALID_SEASONS,
           "test_seasons": TEST_SEASONS, "horizon_laps": PIT_HORIZON}

    for name, part in [("valid", va), ("test", te)]:
        if part.empty:
            continue
        if name == "test":
            # Stage 2: refit on TRAIN + VALID (all seasons strictly earlier than
            # the test season) and evaluate once on the held-out final season.
            # The threshold is carried over from the validation season and is
            # NOT re-tuned on test.
            trv = d[d["year"].isin(TRAIN_SEASONS + VALID_SEASONS)]
            pipe = _new()
            pipe.fit(trv[cols], trv[PIT_TARGET])
            out["refit_seasons_for_test"] = sorted(trv["year"].unique().tolist())
        y = part[PIT_TARGET].to_numpy()
        p = pipe.predict_proba(part[cols])[:, 1]
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        # Threshold chosen on the VALIDATION season only, then applied to test.
        if name == "valid":
            best_t, best_f1 = 0.5, -1.0
            for t in np.arange(0.05, 0.95, 0.01):
                pr, rc, f1, _ = precision_recall_fscore_support(
                    y, (p >= t).astype(int), average="binary", zero_division=0)
                if f1 > best_f1:
                    best_t, best_f1 = float(t), float(f1)
            out["threshold_from_valid"] = best_t
        t = out.get("threshold_from_valid", 0.5)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, (p >= t).astype(int), average="binary", zero_division=0)
        lift = ap / part[PIT_TARGET].mean() if part[PIT_TARGET].mean() > 0 else np.nan
        out[name] = {"roc_auc": round(float(auc), 4), "pr_auc": round(float(ap), 4),
                     "base_rate": round(float(part[PIT_TARGET].mean()), 4),
                     "pr_auc_lift_over_base": round(float(lift), 2),
                     "precision": round(float(pr), 4), "recall": round(float(rc), 4),
                     "f1": round(float(f1), 4), "threshold": round(float(t), 3),
                     "n": len(part)}
        print(f"\n  {name.upper():5s} ({part['year'].unique().tolist()}): "
              f"ROC-AUC={auc:.4f}  PR-AUC={ap:.4f} "
              f"(base rate {part[PIT_TARGET].mean():.4f}, "
              f"lift x{lift:.1f})")
        print(f"         @thr={t:.2f}  precision={pr:.3f}  recall={rc:.3f}  f1={f1:.3f}")

    # Baseline: tyre age alone, as a single-feature ranker. If the GBM cannot
    # beat "the tyres are old", the model is not adding anything.
    if not te.empty:
        b = roc_auc_score(te[PIT_TARGET], te["tyre_life"].fillna(0))
        out["baseline_tyre_life_only_roc_auc"] = round(float(b), 4)
        print(f"\n  BASELINE (rank by tyre_life alone), test: ROC-AUC={b:.4f}")
        print(f"  GBM improvement over baseline: "
              f"{out['test']['roc_auc'] - b:+.4f} ROC-AUC")

    with open(MODEL_DIR / "pit_window.pkl", "wb") as fh:
        pickle.dump({"pipeline": pipe, "features": cols,
                     "num": PIT_NUM, "cat": PIT_CAT, "metrics": out}, fh)
    return out


# ============================================================ FINISHING POS
def build_race_level(df: pd.DataFrame, slopes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate driver-laps to one row per driver-race.

    Every aggregate is over that driver's own laps in that race. Nothing here
    touches the finishing order except the target itself. `status` (DNF reason)
    is explicitly NOT carried into the features - it is only known afterwards.
    """
    d = df.copy()
    d["is_soft"] = (d["compound"] == "SOFT").astype(float)
    d["is_medium"] = (d["compound"] == "MEDIUM").astype(float)
    d["is_hard"] = (d["compound"] == "HARD").astype(float)

    if slopes is not None and not slopes.empty:
        s = slopes.set_index("compound")["deg_s_per_lap"].to_dict()
        d["lap_deg_exposure"] = d["compound"].map(s).astype(float) * d["tyre_life"]
    else:
        d["lap_deg_exposure"] = np.nan

    g = d.groupby(["year", "round", "driver"], as_index=False)
    agg = g.agg(
        circuit=("circuit", "first"),
        team_name=("team_name", "first"),
        grid_position=("grid_position", "first"),
        total_laps=("total_laps", "first"),
        final_position=("final_position", "first"),
        final_classified=("final_classified", "first"),
        status=("status", "first"),
        n_stops=("race_total_stops", "first"),
        n_green_laps=("lap_number", "count"),
        mean_pace_vs_field=("lap_time_vs_field_baseline", "mean"),
        pace_std=("lap_time_vs_field_baseline", "std"),
        mean_tyre_life=("tyre_life", "mean"),
        max_tyre_life=("tyre_life", "max"),
        mean_track_temp=("track_temp", "mean"),
        n_compounds_used=("compound", "nunique"),
        mean_max_speed=("max_speed", "mean"),
        mean_throttle_race=("mean_throttle", "mean"),
        soft_share=("is_soft", "mean"),
        medium_share=("is_medium", "mean"),
        hard_share=("is_hard", "mean"),
        deg_exposure=("lap_deg_exposure", "mean"),
    )
    agg = agg[agg["grid_position"].notna() & (agg["grid_position"] > 0)]
    agg = agg[agg["final_position"].notna()]
    return agg


def finishing_position_model(race_df: pd.DataFrame, target_mode: str = "delta") -> dict:
    """Finishing-position regressor.

    target_mode="delta" predicts (final_position - grid_position) and adds the
    grid back, instead of predicting the position outright. Grid position is by
    far the strongest single predictor in F1, so this gives the model the right
    inductive bias: it starts from "the car finishes where it qualified" and
    only has to learn the deviation. Predicting the absolute position forces it
    to relearn that relationship from a few hundred rows.
    """
    cols = FIN_NUM + FIN_CAT
    tr = race_df[race_df["year"].isin(TRAIN_SEASONS)]
    va = race_df[race_df["year"].isin(VALID_SEASONS)]
    te = race_df[race_df["year"].isin(TEST_SEASONS)]

    print(f"\n{'='*72}\nFINISHING-POSITION REGRESSOR\n{'='*72}")
    print(f"train {TRAIN_SEASONS}: {len(tr):,} driver-races | "
          f"valid {VALID_SEASONS}: {len(va):,} | test {TEST_SEASONS}: {len(te):,}")

    # Hyperparameters selected on the VALIDATION season only (see
    # docs/model_selection.md). The test season was not consulted.
    def _new():
        return make_pipeline("reg", FIN_NUM, FIN_CAT,
                             max_iter=200, learning_rate=0.03,
                             max_leaf_nodes=7, l2_regularization=5.0,
                             early_stopping=False)

    def _y(part):
        if target_mode == "delta":
            return (part[FIN_TARGET].astype(float)
                    - part["grid_position"].astype(float))
        return part[FIN_TARGET].astype(float)

    def _pred(pipe, part):
        raw = pipe.predict(part[cols])
        if target_mode == "delta":
            return raw + part["grid_position"].to_numpy(dtype=float)
        return raw

    pipe = _new()
    pipe.fit(tr[cols], _y(tr))

    out = {"n_train": len(tr), "n_valid": len(va), "n_test": len(te),
           "target_mode": target_mode,
           "train_seasons": TRAIN_SEASONS, "valid_seasons": VALID_SEASONS,
           "test_seasons": TEST_SEASONS}
    for name, part in [("valid", va), ("test", te)]:
        if part.empty:
            continue
        if name == "test":
            trv = race_df[race_df["year"].isin(TRAIN_SEASONS + VALID_SEASONS)]
            pipe = _new()
            pipe.fit(trv[cols], _y(trv))
            out["refit_seasons_for_test"] = sorted(trv["year"].unique().tolist())
        y = part[FIN_TARGET].to_numpy(dtype=float)
        pred = _pred(pipe, part)
        mae = mean_absolute_error(y, pred)
        # Grid position is the baseline that matters: it is very predictive on
        # its own because overtaking is hard.
        base_mae = mean_absolute_error(y, part["grid_position"].to_numpy(dtype=float))
        mean_mae = mean_absolute_error(y, np.full_like(y, tr[FIN_TARGET].mean()))
        out[name] = {
            "mae": round(float(mae), 4),
            "baseline_grid_mae": round(float(base_mae), 4),
            "baseline_mean_mae": round(float(mean_mae), 4),
            "improvement_vs_grid": round(float(base_mae - mae), 4),
            "improvement_pct": round(float((base_mae - mae) / base_mae * 100), 2),
            "beats_grid_baseline": bool(mae < base_mae),
            "r2": round(float(r2_score(y, pred)), 4),
            "n": len(part),
        }
        verdict = "BEATS" if mae < base_mae else "DOES NOT BEAT"
        print(f"\n  {name.upper():5s} ({part['year'].unique().tolist()}): "
              f"MAE={mae:.3f} positions   R^2={r2_score(y, pred):.3f}")
        print(f"         baseline 'predict grid position' MAE={base_mae:.3f}  "
              f"-> model {verdict} it by {base_mae - mae:+.3f} positions "
              f"({(base_mae-mae)/base_mae*100:+.1f}%)")
        print(f"         baseline 'predict train mean'    MAE={mean_mae:.3f}")

    with open(MODEL_DIR / "finishing_position.pkl", "wb") as fh:
        pickle.dump({"pipeline": pipe, "features": cols,
                     "num": FIN_NUM, "cat": FIN_CAT, "metrics": out}, fh)
    return out


def finishing_position_prerace(race_df: pd.DataFrame) -> dict:
    """Strict pre-race finishing-position model: grid + team + circuit only."""
    cols = FIN_PRERACE_NUM + FIN_PRERACE_CAT
    te = race_df[race_df["year"].isin(TEST_SEASONS)]
    trv = race_df[race_df["year"].isin(TRAIN_SEASONS + VALID_SEASONS)]

    print(f"\n{'-'*72}\nFINISHING POSITION - STRICT PRE-RACE VARIANT (no in-race info)")
    print(f"{'-'*72}")
    if te.empty:
        print("  no test season available")
        return {}
    pipe = make_pipeline("reg", FIN_PRERACE_NUM, FIN_PRERACE_CAT,
                         max_iter=200, learning_rate=0.03, max_leaf_nodes=7,
                         l2_regularization=5.0, early_stopping=False)
    # Same delta-from-grid formulation as the main model.
    pipe.fit(trv[cols], trv[FIN_TARGET].astype(float)
             - trv["grid_position"].astype(float))
    y = te[FIN_TARGET].to_numpy(dtype=float)
    pred = pipe.predict(te[cols]) + te["grid_position"].to_numpy(dtype=float)
    mae = mean_absolute_error(y, pred)
    base = mean_absolute_error(y, te["grid_position"].to_numpy(dtype=float))
    out = {"mae": round(float(mae), 4), "baseline_grid_mae": round(float(base), 4),
           "improvement_vs_grid": round(float(base - mae), 4),
           "beats_grid_baseline": bool(mae < base),
           "r2": round(float(r2_score(y, pred)), 4),
           "features": cols, "n_test": len(te), "target_mode": "delta",
           "refit_seasons": sorted(trv["year"].unique().tolist())}
    verdict = "BEATS" if mae < base else "DOES NOT BEAT"
    print(f"  TEST {TEST_SEASONS}: MAE={mae:.3f} positions   R^2={r2_score(y, pred):.3f}")
    print(f"     baseline 'predict grid position' MAE={base:.3f} -> {verdict} "
          f"by {base - mae:+.3f} positions")
    with open(MODEL_DIR / "finishing_position_prerace.pkl", "wb") as fh:
        pickle.dump({"pipeline": pipe, "features": cols, "metrics": out}, fh)
    return out


def expanding_window_cv(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Expanding-window validation: train on <= Y, test on Y+1. Never reversed."""
    cols = PIT_NUM + PIT_CAT
    rows = []
    for i in range(1, len(seasons)):
        tr_y, te_y = seasons[:i], seasons[i]
        tr = df[df["year"].isin(tr_y)]
        te = df[df["year"] == te_y]
        if tr.empty or te.empty:
            continue
        p = make_pipeline("clf", PIT_NUM, PIT_CAT, max_iter=250,
                          learning_rate=0.06, early_stopping=False)
        p.fit(tr[cols], tr[PIT_TARGET])
        prob = p.predict_proba(te[cols])[:, 1]
        rows.append({
            "train_seasons": str(tr_y), "test_season": te_y,
            "n_train": len(tr), "n_test": len(te),
            "base_rate": round(float(te[PIT_TARGET].mean()), 4),
            "roc_auc": round(float(roc_auc_score(te[PIT_TARGET], prob)), 4),
            "pr_auc": round(float(average_precision_score(te[PIT_TARGET], prob)), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    slopes_p = DATA_DIR / "degradation_slopes.csv"
    slopes = pd.read_csv(slopes_p) if slopes_p.exists() else None
    if slopes is None:
        print("WARNING: degradation_slopes.csv missing; deg_exposure will be NaN. "
              "Run src/degradation.py first.")

    pit = pit_window_model(df)

    # The race-level model is built from the ALL-WEATHER table. Dropping wet
    # races is a tyre-degradation decision; for "where did this car finish" a
    # wet race is a perfectly valid observation, and excluding them costs the
    # model roughly a third of its rows.
    aw = DATA_DIR / "features_all_weather.parquet"
    race_src = pd.read_parquet(aw) if aw.exists() else df
    print(f"\nrace-level table built from "
          f"{'features_all_weather.parquet' if aw.exists() else 'features.parquet'}: "
          f"{race_src.groupby(['year','round']).ngroups} races")
    race_df = build_race_level(race_src, slopes)
    race_df.to_parquet(DATA_DIR / "race_level.parquet", index=False)
    fin = finishing_position_model(race_df)
    fin_pre = finishing_position_prerace(race_df)

    seasons = sorted(df["year"].unique().tolist())
    print(f"\n{'='*72}\nEXPANDING-WINDOW VALIDATION (pit model)\n{'='*72}")
    cv = expanding_window_cv(df, seasons)
    print(cv.to_string(index=False))
    cv.to_csv(DATA_DIR / "expanding_window_cv.csv", index=False)

    with open(DATA_DIR / "model_metrics.json", "w") as fh:
        json.dump({"pit_window": pit, "finishing_position": fin,
                   "finishing_position_prerace": fin_pre,
                   "expanding_window_cv": cv.to_dict("records")}, fh, indent=2)
    print(f"\nmetrics -> {DATA_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
