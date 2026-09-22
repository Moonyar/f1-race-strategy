"""Per-compound tyre-degradation curves via statsmodels.

The output is a table of degradation slopes: seconds lost per lap of tyre age,
per compound.

Specification
-------------
The obvious model is a pooled OLS of fuel-corrected lap time on
`tyre_life * C(compound)` with circuit and driver fixed effects. It is biased
here: it put HARD above MEDIUM and gave the track-evolution term a coefficient
of +0.40, when the term is built in seconds of field-wide pace and should come
out near -1.

The reason is between-stint selection. Stint length is chosen by the strategist
and differs systematically by compound (HARD stints average 27 laps, MEDIUM 19,
SOFT 16), and stints differ in traffic, car damage and race phase. Pooled OLS
identifies the tyre-age slope partly from that between-stint variation, which is
confounded with everything the strategist was reacting to.

The fix is a stint fixed effect: every stint gets its own intercept, so the
slope is identified only from how a car's pace changes within one stint. With
stint effects absorbed the track-evolution coefficient is -1.02, as expected.
Both models are fitted so the difference can be seen.

Sanity check
------------
SOFT should degrade faster than MEDIUM and HARD. If it does not, the likely cause
is upstream (surviving in/out laps, or a broken fuel correction), so the script
exits non-zero. MEDIUM vs HARD is not checked, because in this data the two do
not separate (see `compound_contrasts.csv`).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from config import DATA_DIR, FIG_DIR, DRY_COMPOUNDS

# Stints need enough laps and enough spread in tyre age to identify a slope.
MIN_STINT_LAPS = 5
MIN_STINT_AGE_SPAN = 3

CONTROLS = ["track_temp", "track_evolution_gain_s"]

# Pooled specification, fitted only for comparison with the within-stint model.
NAIVE_FORMULA = (
    "lap_time_fuel_corrected ~ tyre_life * C(compound) "
    "+ C(circuit) + C(driver) + C(year) + track_temp + track_evolution_gain_s"
)


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    d = df[df["compound"].isin(DRY_COMPOUNDS)].copy()
    d["compound"] = d["compound"].astype(str)
    d = d.dropna(subset=["lap_time_fuel_corrected", "tyre_life", "compound",
                         "circuit", "driver", "year"] + CONTROLS)
    d["race_id"] = d["year"].astype(str) + "_" + d["round"].astype(str)
    d["stint_uid"] = (d["race_id"] + "_" + d["driver"] + "_"
                      + d["stint_number"].astype(str))
    return d


def _usable_stints(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("stint_uid")["tyre_life"]
    n = g.transform("size")
    span = g.transform("max") - g.transform("min")
    return d[(n >= MIN_STINT_LAPS) & (span >= MIN_STINT_AGE_SPAN)].copy()


def fit_within_stint(d: pd.DataFrame, max_tyre_life: float | None = None,
                     quadratic: bool = False):
    """Within-stint fixed-effects estimate of per-compound degradation.

    Stint effects are absorbed by demeaning within stint (the standard within
    transform) rather than by 2,500+ dummy columns. Standard errors are
    clustered by race and then scaled by sqrt((N-K)/(N-K-n_stints)) to account
    for the degrees of freedom used up by the absorbed effects, which the OLS
    object cannot know about.
    """
    x = d if max_tyre_life is None else d[d["tyre_life"] <= max_tyre_life]
    x = _usable_stints(x)
    if x.empty:
        return None

    terms: list[str] = []
    for c in DRY_COMPOUNDS:
        if (x["compound"] == c).sum() < 100:
            continue
        x[f"tl_{c}"] = x["tyre_life"] * (x["compound"] == c)
        terms.append(f"tl_{c}")
        if quadratic:
            x[f"tl2_{c}"] = (x["tyre_life"] ** 2) * (x["compound"] == c)
            terms.append(f"tl2_{c}")
    cols = terms + CONTROLS

    g = x.groupby("stint_uid")
    y = (x["lap_time_fuel_corrected"]
         - g["lap_time_fuel_corrected"].transform("mean"))
    X = pd.DataFrame({c: x[c] - g[c].transform("mean") for c in cols},
                     index=x.index)
    X = X.loc[:, X.std() > 1e-9]

    # Cluster-robust SEs need a decent number of clusters; with only a handful
    # of races (per-circuit fits) the estimator is unreliable and statsmodels
    # can fail outright, so fall back to HC1 and record which was used.
    n_clusters = x["race_id"].nunique()
    try:
        if n_clusters >= 5:
            res = sm.OLS(y, X).fit(cov_type="cluster",
                                   cov_kwds={"groups": x["race_id"]})
            se_kind = f"cluster({n_clusters} races)"
        else:
            res = sm.OLS(y, X).fit(cov_type="HC1")
            se_kind = f"HC1 ({n_clusters} races)"
    except Exception:  # noqa: BLE001
        res = sm.OLS(y, X).fit(cov_type="HC1")
        se_kind = f"HC1 fallback ({n_clusters} races)"
    res._se_kind = se_kind
    n_stints = x["stint_uid"].nunique()
    N, K = len(x), X.shape[1]
    res._df_scale = float(np.sqrt((N - K) / max(N - K - n_stints, 1)))
    res._n_stints = n_stints
    res._n_races = x["race_id"].nunique()
    res._cols = list(X.columns)
    res._sample = x
    return res


def slope_table(res) -> pd.DataFrame:
    x = res._sample
    rows = []
    for c in DRY_COMPOUNDS:
        key = f"tl_{c}"
        if key not in res.params.index:
            continue
        est = float(res.params[key])
        se = float(res.bse[key]) * res._df_scale
        rows.append({
            "compound": c,
            "deg_s_per_lap": round(est, 4),
            "std_err": round(se, 4),
            "t_stat": round(est / se, 2) if se else np.nan,
            "ci_low": round(est - 1.96 * se, 4),
            "ci_high": round(est + 1.96 * se, 4),
            "n_laps": int((x["compound"] == c).sum()),
            "n_stints": int(x.loc[x["compound"] == c, "stint_uid"].nunique()),
        })
    order = {c: i for i, c in enumerate(DRY_COMPOUNDS)}
    return (pd.DataFrame(rows).sort_values("compound", key=lambda s: s.map(order))
            .reset_index(drop=True))


def contrast_table(res) -> pd.DataFrame:
    """Pairwise differences in degradation slope, with p-values.

    Whether the compounds can be ranked is a question about the differences, so
    they are tested directly rather than read off overlapping intervals.
    """
    cols = res._cols
    cov = res.cov_params().to_numpy()
    rows = []
    for a, b in [("SOFT", "MEDIUM"), ("SOFT", "HARD"), ("MEDIUM", "HARD")]:
        ka, kb = f"tl_{a}", f"tl_{b}"
        if ka not in cols or kb not in cols:
            continue
        v = np.zeros(len(cols))
        v[cols.index(ka)] = 1.0
        v[cols.index(kb)] = -1.0
        est = float(v @ res.params.to_numpy())
        se = float(np.sqrt(v @ cov @ v)) * res._df_scale
        t = est / se if se else np.nan
        p = float(2 * (1 - stats.norm.cdf(abs(t))))
        rows.append({"contrast": f"{a} - {b}", "diff_s_per_lap": round(est, 4),
                     "std_err": round(se, 4), "t_stat": round(t, 2),
                     "p_value": round(p, 4),
                     "significant_at_05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def window_sensitivity(d: pd.DataFrame) -> pd.DataFrame:
    """How the slopes move as the tyre-age window changes.

    Compounds are run to very different ages (SOFT rarely past 20 laps, HARD
    routinely past 30). A single linear slope over different age ranges is not
    an apples-to-apples comparison if degradation is at all non-linear, so the
    slopes are refitted over several windows.
    """
    rows = []
    for m in [15, 20, 25, 30, None]:
        r = fit_within_stint(d, max_tyre_life=m)
        if r is None:
            continue
        t = slope_table(r).set_index("compound")["deg_s_per_lap"]
        rec = {"max_tyre_life": m if m else "all"}
        rec.update({c: t.get(c, np.nan) for c in DRY_COMPOUNDS})
        rec["n_laps"] = int(r.nobs)
        rec["soft_is_highest"] = bool(t.get("SOFT", -9) == t.max())
        rows.append(rec)
    return pd.DataFrame(rows)


def check_soft_is_worst(tab: pd.DataFrame) -> tuple[bool, str]:
    s = tab.set_index("compound")["deg_s_per_lap"]
    if "SOFT" not in s.index:
        return False, "SOFT not estimated"
    ok = bool(s["SOFT"] == s.max() and s["SOFT"] > 0)
    msg = ", ".join(f"{c} {s[c]:+.4f}" for c in s.index)
    return ok, msg


def per_circuit_slopes(d: pd.DataFrame, min_laps: int = 300) -> pd.DataFrame:
    rows = []
    for circuit, g in d.groupby("circuit"):
        if len(g) < min_laps or g["compound"].nunique() < 2:
            continue
        r = fit_within_stint(g)
        if r is None:
            continue
        try:
            t = slope_table(r)
        except Exception:  # noqa: BLE001
            continue
        if t.empty:
            continue
        t["circuit"] = circuit
        t["n_races"] = g["race_id"].nunique()
        rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_curves(d: pd.DataFrame, tab: pd.DataFrame, path=None):
    """Observed within-stint pace evolution by tyre age, per compound."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or (FIG_DIR / "degradation_curves.png")
    colors = {"SOFT": "#d32f2f", "MEDIUM": "#f9a825", "HARD": "#455a64"}
    x = _usable_stints(d).copy()
    # Centre each stint on its own mean: this is the same within transform the
    # model uses, so the picture shows what the model actually fits.
    x["centred"] = (x["lap_time_fuel_corrected"]
                    - x.groupby("stint_uid")["lap_time_fuel_corrected"].transform("mean"))
    x["age_c"] = x["tyre_life"] - x.groupby("stint_uid")["tyre_life"].transform("mean")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for comp in DRY_COMPOUNDS:
        g = x[x["compound"] == comp]
        if g.empty:
            continue
        m = g.groupby("age_c")["centred"].agg(["mean", "count"])
        # Require a dense bin, scaled to how many stints this compound has:
        # SOFT is run far less often than HARD, so a fixed count would truncate
        # the SOFT curve to nothing while leaving HARD's noisy tails in.
        thresh = max(25, 0.12 * g["stint_uid"].nunique())
        m = m[(m["count"] >= thresh)
              & (np.abs(np.asarray(m.index, dtype=float)) <= 12)]
        if m.empty:
            continue
        ax.plot(m.index, m["mean"], marker="o", ms=3, lw=1.8, color=colors[comp],
                label=f"{comp} ({int(g['stint_uid'].nunique()):,} stints)")
        row = tab[tab["compound"] == comp]
        if not row.empty:
            sl = float(row["deg_s_per_lap"].iloc[0])
            xs = np.array(m.index, dtype=float)
            ax.plot(xs, sl * xs, ls="--", lw=1.1, color=colors[comp], alpha=.65)
    ax.axhline(0, color="k", lw=.6, alpha=.4)
    ax.set_xlabel("Tyre age relative to stint mean (laps)")
    ax.set_ylabel("Fuel-corrected pace relative to stint mean (s)")
    ax.set_title("Within-stint tyre degradation by compound, 2022-2024 dry races\n"
                 "solid = observed mean, dashed = fitted within-stint slope")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    d = load_features()
    print(f"degradation sample: {len(d):,} laps, {d['race_id'].nunique()} dry "
          f"races, {d['stint_uid'].nunique():,} stints, "
          f"seasons {sorted(d['year'].unique().tolist())}")

    # ---- pooled model, for comparison -----------------------------------------
    naive = smf.ols(NAIVE_FORMULA, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["race_id"]})
    naive_te = float(naive.params.get("track_evolution_gain_s", np.nan))

    # ---- primary model ------------------------------------------------------
    res = fit_within_stint(d)
    tab = slope_table(res)
    con = contrast_table(res)

    print("\n=== PER-COMPOUND DEGRADATION SLOPES ===")
    print("    within-stint fixed effects, SEs clustered by race, s per lap of tyre age")
    print(tab.to_string(index=False))
    print(f"\nstints={res._n_stints:,}  races={res._n_races}  n={int(res.nobs):,}")
    te = float(res.params.get("track_evolution_gain_s", np.nan))
    print(f"track_evolution_gain_s = {te:+.4f}   track_temp = "
          f"{float(res.params.get('track_temp', np.nan)):+.4f}")
    print(f"\nSPECIFICATION CHECK: the track-evolution term is built in seconds of "
          f"field-wide pace,\n  so its coefficient must land near -1. "
          f"within-stint FE: {te:+.3f}   naive pooled OLS: {naive_te:+.3f}"
          f"  <- the naive model is misspecified.")

    ok, msg = check_soft_is_worst(tab)
    print(f"\nSANITY CHECK  SOFT degrades fastest:  {msg}")
    if not ok:
        print("*** FAILED. Do NOT trust these numbers. Check for surviving "
              "in/out laps or a broken fuel correction. ***")
        return 1
    print("*** PASSED ***")

    print("\n=== PAIRWISE CONTRASTS (is the ranking real?) ===")
    print(con.to_string(index=False))
    mh = con[con["contrast"] == "MEDIUM - HARD"]
    mh_sig = bool(mh["significant_at_05"].iloc[0]) if not mh.empty else False
    if not mh_sig:
        print("\nMEDIUM and HARD do not separate in this data; only the "
              "SOFT-vs-rest gap is a result.")

    print("\n=== TYRE-AGE WINDOW SENSITIVITY ===")
    win = window_sensitivity(d)
    print(win.to_string(index=False))

    # ---- quadratic ----------------------------------------------------------
    resq = fit_within_stint(d, quadratic=True)
    print("\n=== QUADRATIC TERM (is there a measurable tyre 'cliff'?) ===")
    qterms = {}
    for c in DRY_COMPOUNDS:
        k = f"tl2_{c}"
        if k in resq.params.index:
            se = float(resq.bse[k]) * resq._df_scale
            t = float(resq.params[k]) / se if se else np.nan
            p = float(2 * (1 - stats.norm.cdf(abs(t))))
            qterms[c] = (float(resq.params[k]), p)
            print(f"  {c:7s} quadratic coef={resq.params[k]:+.6f}  p={p:.4f}")

    # ---- artifacts ----------------------------------------------------------
    raw_files = sorted((DATA_DIR / "raw").glob("*.parquet"))
    n_wet = 0
    for f in raw_files:
        try:
            if bool(pd.read_parquet(f, columns=["Rainfall"])["Rainfall"]
                    .fillna(False).any()):
                n_wet += 1
        except Exception:  # noqa: BLE001
            pass

    s = tab.set_index("compound")["deg_s_per_lap"]
    ratio = s["SOFT"] / s.drop("SOFT").mean() if len(s) > 1 else float("nan")
    meta = {
        "n_races_ingested": len(raw_files),
        "n_wet_races_dropped": n_wet,
        "n_races_modelled": int(d["race_id"].nunique()),
        "n_stints": int(res._n_stints),
        "n_laps": int(res.nobs),
        "track_evolution_coef": te,
        "track_evolution_coef_naive": naive_te,
        "track_temp_coef": float(res.params.get("track_temp", np.nan)),
        "soft_vs_others_ratio": float(ratio),
        "sanity_check": msg,
        "medium_hard_separate": mh_sig,
        "quadratic_terms": {c: {"coef": q, "p": p} for c, (q, p) in qterms.items()},
    }
    with open(DATA_DIR / "degradation_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    tab.to_csv(DATA_DIR / "degradation_slopes.csv", index=False)
    con.to_csv(DATA_DIR / "compound_contrasts.csv", index=False)
    win.to_csv(DATA_DIR / "degradation_window_sensitivity.csv", index=False)

    circ = per_circuit_slopes(d)
    if not circ.empty:
        circ.to_csv(DATA_DIR / "degradation_slopes_by_circuit.csv", index=False)
        print(f"\nper-circuit slopes written for {circ['circuit'].nunique()} circuits")
    print(f"figure: {plot_curves(d, tab)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
