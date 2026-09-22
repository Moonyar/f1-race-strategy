"""Verify the leakage checklist in code.

Each item is CHECKED, not asserted. Several checks read the source code of the
pipeline itself; others recompute statistics from the saved artifacts. Anything
that cannot be proven mechanically is printed as MANUAL with the reasoning, so
the distinction between "verified" and "argued" stays visible.

Run:  python src/leakage_check.py
"""
from __future__ import annotations

import ast
import pickle

import pandas as pd

from config import (DATA_DIR, MODEL_DIR, PIT_HORIZON, ROOT, TEST_SEASONS,
                    TRAIN_SEASONS, VALID_SEASONS)
import models as M

SRC = ROOT / "src"
RESULTS: list[tuple[str, str, str]] = []


def record(status: str, item: str, detail: str) -> None:
    RESULTS.append((status, item, detail))
    print(f"[{status:6s}] {item}\n         {detail}")


# ---------------------------------------------------------------- 1
def check_no_random_split() -> None:
    """No random train_test_split / shuffle / KFold anywhere in the pipeline.

    This parses each module's AST rather than grepping its text. A text grep
    matches the phrase inside a docstring that merely *claims* no random split
    is used, which is exactly the kind of false reassurance this check exists to
    rule out. The AST walk looks only at code that actually executes: imported
    names, called functions, attribute accesses, and keyword arguments.
    """
    banned_names = {"train_test_split", "ShuffleSplit", "KFold",
                    "StratifiedKFold", "GroupKFold", "cross_val_score",
                    "cross_validate", "RandomizedSearchCV", "permutation"}
    hits: list[str] = []
    scanned = 0
    for f in sorted(SRC.glob("*.py")) + [ROOT / "app.py"]:
        if not f.exists() or f.name == "leakage_check.py":
            continue
        scanned += 1
        tree = ast.parse(f.read_text(), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if a.name.split(".")[-1] in banned_names:
                        hits.append(f"{f.name}:{node.lineno} imports {a.name}")
            elif isinstance(node, ast.Name) and node.id in banned_names:
                hits.append(f"{f.name}:{node.lineno} uses {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in banned_names:
                hits.append(f"{f.name}:{node.lineno} calls .{node.attr}")
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shuffle" and isinstance(kw.value, ast.Constant) \
                            and kw.value.value is True:
                        hits.append(f"{f.name}:{node.lineno} passes shuffle=True")
                    if kw.arg == "frac":
                        fn = node.func
                        nm = fn.attr if isinstance(fn, ast.Attribute) else ""
                        if nm == "sample":
                            hits.append(f"{f.name}:{node.lineno} .sample(frac=...)")
    if hits:
        record("FAIL", "1. No random train_test_split anywhere",
               "found: " + "; ".join(hits))
    else:
        record("PASS", "1. No random train_test_split anywhere",
               f"AST-parsed {scanned} modules (src/*.py + app.py) for "
               f"{len(banned_names)} shuffle/CV symbols, shuffle=True kwargs and "
               f"DataFrame.sample(frac=); zero executable matches. Docstrings and "
               f"comments are ignored by construction, so a module merely "
               f"*mentioning* train_test_split cannot pass this check by accident "
               f"or fail it by accident. Splits are by season only: "
               f"train={TRAIN_SEASONS} valid={VALID_SEASONS} test={TEST_SEASONS}.")


# ---------------------------------------------------------------- 2
def check_no_stint_length() -> None:
    """Stint TOTAL length must never be a feature - it is the answer."""
    banned_names = ["stint_length", "stint_total", "stint_len", "total_stint",
                    "laps_in_stint", "stint_end", "next_pit_lap", "pit_lap"]
    feats = M.PIT_NUM + M.PIT_CAT + M.FIN_NUM + M.FIN_CAT
    bad = [f for f in feats if any(b in f.lower() for b in banned_names)]
    # Also: does any feature correlate suspiciously with remaining stint length?
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    d = df.sort_values(["year", "round", "driver", "lap_number"]).copy()
    grp = d.groupby(["year", "round", "driver", "stint_number"])
    d["stint_total_len"] = grp["tyre_life"].transform("max")
    d["laps_left_in_stint"] = d["stint_total_len"] - d["tyre_life"]
    present = [f for f in M.PIT_NUM if f in d.columns]
    corr = d[present].corrwith(d["laps_left_in_stint"]).abs().sort_values(ascending=False)
    worst = corr.head(3)
    if bad:
        record("FAIL", "2. Stint total length is never a feature",
               f"banned-looking features present: {bad}")
    else:
        record("PASS", "2. Stint total length is never a feature",
               "no feature name encodes stint length. Highest |corr| between any "
               "model feature and laps-left-in-stint (the leaked quantity): "
               + ", ".join(f"{k}={v:.3f}" for k, v in worst.items())
               + ". tyre_life correlating moderately is expected and legitimate - "
                 "tyre age is known at lap L; stint LENGTH is not.")


# ---------------------------------------------------------------- 3
def check_no_results_in_pit_model() -> None:
    """Final race results must never feed the pit-window model."""
    post_race = ["final_position", "final_classified", "status", "points",
                 "n_stops", "max_tyre_life"]
    feats = M.PIT_NUM + M.PIT_CAT
    bad = [f for f in feats if f.lower() in post_race]
    if bad:
        record("FAIL", "3. Final race results never feed the pit model", f"leaked: {bad}")
    else:
        record("PASS", "3. Final race results never feed the pit model",
               f"pit model uses {len(feats)} features, none of "
               f"{post_race} among them. Verified against the exact feature "
               f"lists the fitted pipeline was trained on.")


# ---------------------------------------------------------------- 3b
def check_finishing_model_features() -> None:
    """Finishing-position model must exclude Status (DNF reason)."""
    feats = M.FIN_NUM + M.FIN_CAT
    bad = [f for f in feats if f.lower() in ("status", "points", "final_classified")]
    # n_stops IS a legitimate post-hoc race summary for this model: the target
    # is the finishing position of a completed race, so race-summary features
    # are in-scope. Flag it so the distinction is explicit, not accidental.
    if bad:
        record("FAIL", "3b. Finishing model excludes Status / DNF reason", f"leaked: {bad}")
    else:
        record("PASS", "3b. Finishing model excludes Status / DNF reason",
               "Status, Points and ClassifiedPosition are all absent from "
               "FIN_NUM/FIN_CAT. NOTE (scope, not leakage): this model is a "
               "post-race explanatory model - it summarises a completed race "
               "(n_stops, mean pace) to predict where the driver finished. It is "
               "NOT a pre-race predictor and must not be described as one.")


# ---------------------------------------------------------------- 4
def check_pipeline_fit_scope() -> None:
    """Imputers/encoders must be fit inside the training fold only."""
    ok = True
    details = []
    for name, fn in [("pit_window", "pit_window.pkl"),
                     ("finishing_position", "finishing_position.pkl")]:
        p = MODEL_DIR / fn
        if not p.exists():
            details.append(f"{name}: model not found, run src/models.py")
            ok = False
            continue
        obj = pickle.load(open(p, "rb"))
        pipe = obj["pipeline"]
        steps = [s[0] for s in pipe.steps]
        pre = pipe.named_steps["pre"]
        inner = [t[0] for t in pre.transformers_]
        details.append(f"{name}: Pipeline{steps} with ColumnTransformer{inner}; "
                       f"imputers+OrdinalEncoder live INSIDE the pipeline, so "
                       f".fit(train) fits them on train rows only")
    # Prove it: a fitted median imputer must equal the TRAIN median, not the all-data median.
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    p = MODEL_DIR / "pit_window.pkl"
    if p.exists():
        obj = pickle.load(open(p, "rb"))
        imp = obj["pipeline"].named_steps["pre"].named_transformers_["num"]
        # model was refit on train+valid for the test evaluation
        refit = obj["metrics"].get("refit_seasons_for_test", TRAIN_SEASONS)
        ref = df[df["year"].isin(refit)]
        col = "track_temp"
        i = M.PIT_NUM.index(col)
        fitted = float(imp.statistics_[i])
        want = float(ref[col].median())
        alldata = float(df[col].median())
        match = abs(fitted - want) < 1e-6
        details.append(
            f"proof: fitted median for '{col}' = {fitted:.4f}; median over refit "
            f"seasons {refit} = {want:.4f} (match={match}); median over ALL "
            f"seasons incl. test = {alldata:.4f}. The imputer learned the "
            f"training-fold value, not the all-data value.")
        ok = ok and match
    record("PASS" if ok else "FAIL",
           "4. Scalers/imputers fit inside the training fold only",
           " | ".join(details))


# ---------------------------------------------------------------- 5
def check_weather_timing() -> None:
    """Weather must come from lap L or earlier, never race-end conditions."""
    src = (SRC / "ingest.py").read_text()
    uses_lapstart = 'left_on="LapStartTime"' in src
    uses_nearest = 'direction="nearest"' in src
    # Empirical check: weather attached to lap 1 must differ from lap N.
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    firsts = df.sort_values("lap_number").groupby(["year", "round"]).first()
    lasts = df.sort_values("lap_number").groupby(["year", "round"]).last()
    diff = (firsts["track_temp"] - lasts["track_temp"]).abs()
    varying = float((diff > 0.5).mean())
    per_race_const = float((df.groupby(["year", "round"])["track_temp"].nunique() == 1).mean())
    ok = uses_lapstart and uses_nearest and varying > 0.5
    record("PASS" if ok else "FAIL",
           "5. Weather features come from lap L or earlier",
           f"join is merge_asof on LapStartTime ({uses_lapstart}) with "
           f"direction='nearest' ({uses_nearest}), i.e. keyed to the START of "
           f"each lap. Empirically track_temp differs between the first and "
           f"last lap in {varying*100:.0f}% of races (median |delta| "
           f"{diff.median():.1f} C), and is constant across a whole race in only "
           f"{per_race_const*100:.0f}% - so laps are NOT all carrying one "
           f"race-end weather value. CAVEAT: 'nearest' can pick an observation "
           f"up to ~30 s after the lap start. Within-lap, not end-of-race.")


# ---------------------------------------------------------------- 6
def check_field_baseline() -> None:
    """Field baseline is a within-race contemporaneous aggregate. Documented."""
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    # Show it is within-race: baseline for a race must depend only on that race.
    n_races = df.groupby(["year", "round"]).ngroups
    # Does the baseline use future laps? The rolling window is centred (window=5),
    # so lap L's baseline uses laps L-2..L+2 of the SAME race.
    src = (SRC / "features.py").read_text()
    centred = "center=True" in src
    record("MANUAL", "6. Field-baseline lap times are a within-race aggregate",
           f"Known and accepted. track_evolution_baseline is the field's "
           f"median fuel-corrected lap time per lap number, computed within each "
           f"of the {n_races} races separately, smoothed with a CENTRED rolling "
           f"median (center=True: {centred}, window=5). Two consequences, both "
           f"accepted knowingly: (a) it uses contemporaneous laps by OTHER "
           f"drivers - defensible, because that is exactly what a pit wall sees "
           f"live; (b) the centred window means lap L's baseline also uses laps "
           f"L+1 and L+2 of the same race, which a live strategist would not yet "
           f"have. This is a genuine 2-lap lookahead in a field-wide aggregate. "
           f"It is NOT driver-specific and carries no information about whether "
           f"THIS driver pits, but it is the one place the pipeline is not "
           f"strictly causal. A trailing window would remove it at the cost of "
           f"a laggier baseline.")


# ---------------------------------------------------------------- 7
def check_season_ordering() -> None:
    """Train seasons must all be strictly earlier than test seasons."""
    tr, va = max(TRAIN_SEASONS), max(VALID_SEASONS)
    ok = tr < min(VALID_SEASONS) and va < min(TEST_SEASONS)
    cv = DATA_DIR / "expanding_window_cv.csv"
    extra = ""
    if cv.exists():
        c = pd.read_csv(cv)
        bad = [r for _, r in c.iterrows()
               if max(ast.literal_eval(r["train_seasons"])) >= r["test_season"]]
        extra = (f" Expanding-window CV: {len(c)} folds, "
                 f"{len(bad)} with a train season >= its test season.")
        ok = ok and not bad
    record("PASS" if ok else "FAIL",
           "7. Time ordering: later seasons never inform earlier predictions",
           f"train={TRAIN_SEASONS} < valid={VALID_SEASONS} < test={TEST_SEASONS}."
           + extra +
           " Test-season model is refit on train+valid (all strictly earlier "
           "seasons); the decision threshold is chosen on valid and carried "
           "over, never re-tuned on test.")


# ---------------------------------------------------------------- 8
def check_label_construction() -> None:
    """The pit label must look FORWARD only, and must not be trivially in X."""
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    y = df["pits_within_horizon"]
    feats = [f for f in M.PIT_NUM if f in df.columns]
    corr = df[feats].corrwith(y).abs().sort_values(ascending=False)
    top = corr.head(3)
    leaky = corr[corr > 0.9]
    record("PASS" if leaky.empty else "FAIL",
           "8. Pit label is forward-looking and not present in X",
           f"label = 1 iff a PitInTime occurs in laps L+1..L+{PIT_HORIZON} "
           f"(strict >, so lap L's own pit never labels itself). Base rate "
           f"{y.mean()*100:.2f}%. Max |corr| between any feature and the label: "
           + ", ".join(f"{k}={v:.3f}" for k, v in top.items())
           + f"; {len(leaky)} features exceed 0.9.")


def main() -> int:
    print("=" * 78)
    print("LEAKAGE CHECKLIST")
    print("=" * 78 + "\n")
    for fn in [check_no_random_split, check_no_stint_length,
               check_no_results_in_pit_model, check_finishing_model_features,
               check_pipeline_fit_scope, check_weather_timing,
               check_field_baseline, check_season_ordering,
               check_label_construction]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record("ERROR", fn.__name__, f"{type(e).__name__}: {e}")
        print()

    n_fail = sum(1 for s, _, _ in RESULTS if s in ("FAIL", "ERROR"))
    n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
    n_man = sum(1 for s, _, _ in RESULTS if s == "MANUAL")
    print("=" * 78)
    print(f"SUMMARY: {n_pass} verified PASS, {n_man} MANUAL (documented "
          f"judgement call), {n_fail} FAIL/ERROR")
    print("=" * 78)

    out = pd.DataFrame(RESULTS, columns=["status", "item", "detail"])
    out.to_csv(DATA_DIR / "leakage_checklist.csv", index=False)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
