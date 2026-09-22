"""Streamlit front end for the F1 race-strategy models.

Reads data/features.parquet and the pickled models. Nothing is retrained here -
the app is a viewer over artifacts produced by src/.

    streamlit run app.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_DIR, MODEL_DIR  # noqa: E402

st.set_page_config(page_title="F1 Race Strategy Modeling", layout="wide")

COMPOUND_COLORS = {"SOFT": "#d32f2f", "MEDIUM": "#f9a825", "HARD": "#90a4ae"}


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "features.parquet")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_model(name: str):
    p = MODEL_DIR / name
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


try:
    df = load_features()
except FileNotFoundError:
    st.error("data/features.parquet not found. Run `python src/ingest.py` then "
             "`python src/features.py` first.")
    st.stop()

slopes = load_csv("degradation_slopes.csv")
slopes_circ = load_csv("degradation_slopes_by_circuit.csv")
attrition = load_csv("lap_attrition.csv")
leak = load_csv("leakage_checklist.csv")
pit_model = load_model("pit_window.pkl")

st.title("Formula 1 Race Strategy Modeling")
st.caption(
    f"{len(df):,} green-flag dry driver-laps | "
    f"{df.groupby(['year','round']).ngroups} races | "
    f"seasons {', '.join(str(y) for y in sorted(df['year'].unique()))} "
    f"(ground-effect regulation era) | source: FastF1 API"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Degradation", "Race explorer", "Pit-window model", "Data & validation"]
)

# ============================================================== TAB 1
with tab1:
    st.subheader("Per-compound tyre-degradation curves")
    st.markdown(
        "Lap times are **fuel-corrected** and expressed relative to a per-race "
        "**track-evolution baseline**, so what is left is tyre age. Slopes come "
        "from an OLS fit with circuit, driver and season fixed effects."
    )
    circuits = ["All circuits (pooled fit)"] + sorted(df["circuit"].dropna().unique())
    c = st.selectbox("Circuit", circuits)

    if c == "All circuits (pooled fit)" and slopes is not None:
        show = slopes.copy()
    elif slopes_circ is not None and c in set(slopes_circ.get("circuit", [])):
        show = slopes_circ[slopes_circ["circuit"] == c].copy()
    else:
        show = None

    if show is not None and not show.empty:
        st.dataframe(
            show[[col for col in ["compound", "deg_s_per_lap", "std_err",
                                  "ci_low", "ci_high", "t_stat", "n_laps"]
                  if col in show.columns]],
            hide_index=True, width='stretch',
        )
        cols = st.columns(len(show))
        for col, (_, r) in zip(cols, show.iterrows()):
            col.metric(f"{r['compound']} degradation",
                       f"{r['deg_s_per_lap']:+.3f} s/lap")
    else:
        st.info(f"No fitted slope table for {c} (too few laps or compounds). "
                "Observed curve below still reflects the raw data.")

    sub = df if c.startswith("All") else df[df["circuit"] == c]
    plot = sub.copy()
    plot["centred"] = plot["lap_time_vs_field_baseline"] - plot.groupby(
        ["year", "round", "driver"])["lap_time_vs_field_baseline"].transform("mean")
    agg = (plot.groupby(["tyre_life", "compound"])["centred"]
           .agg(["mean", "count"]).reset_index())
    agg = agg[(agg["count"] >= 10) & (agg["tyre_life"] <= 40)]
    if not agg.empty:
        wide = agg.pivot(index="tyre_life", columns="compound", values="mean")
        st.line_chart(wide, x_label="Tyre age (laps)",
                      y_label="Pace vs field baseline (s, centred)",
                      color=[COMPOUND_COLORS.get(x, "#888") for x in wide.columns])
    st.caption("Observed mean pace by tyre age, centred within driver-race so "
               "fast drivers and fast circuits do not shift the curves.")

# ============================================================== TAB 2
with tab2:
    st.subheader("Race explorer")
    yr = st.selectbox("Season", sorted(df["year"].unique()), key="y2")
    races = df[df["year"] == yr][["round", "event_name"]].drop_duplicates().sort_values("round")
    ev = st.selectbox("Race", races["event_name"].tolist(), key="r2")
    r = df[(df["year"] == yr) & (df["event_name"] == ev)]

    drivers = sorted(r["driver"].unique())
    default = drivers[: min(4, len(drivers))]
    picked = st.multiselect("Drivers", drivers, default=default)
    rr = r[r["driver"].isin(picked)]

    if rr.empty:
        st.info("Pick at least one driver.")
    else:
        metric = st.radio("Pace metric", ["lap_time_fuel_corrected", "lap_time",
                                          "lap_time_vs_field_baseline"],
                          horizontal=True)
        wide = rr.pivot_table(index="lap_number", columns="driver", values=metric)
        st.line_chart(wide, x_label="Lap", y_label=f"{metric} (s)")

        st.markdown("**Stints** (gaps in a driver's lap trace are pit laps and "
                    "safety-car laps, which are filtered out of the modelling set)")
        stints = (rr.groupby(["driver", "stint_number", "compound"])
                  .agg(first_lap=("lap_number", "min"),
                       last_lap=("lap_number", "max"),
                       green_laps=("lap_number", "count"),
                       max_tyre_age=("tyre_life", "max"),
                       mean_pace=("lap_time_vs_field_baseline", "mean"))
                  .reset_index().sort_values(["driver", "first_lap"]))
        st.dataframe(stints.round(3), hide_index=True, width='stretch')

        te = r.groupby("lap_number")["track_evolution_gain_s"].first()
        st.markdown("**Track evolution** - field-wide pace gain vs the start of "
                    "the race (positive = track has rubbered in and is faster)")
        st.line_chart(te, x_label="Lap", y_label="Gain vs lap 1 (s)")

# ============================================================== TAB 3
with tab3:
    st.subheader("Pit-window model")
    if pit_model is None:
        st.warning("models/pit_window.pkl not found. Run `python src/models.py`.")
    else:
        m = pit_model["metrics"]
        st.markdown(
            f"**Question:** at lap L, does this driver pit within the next "
            f"**{m['horizon_laps']} laps**?  \n"
            f"Trained on seasons {m.get('refit_seasons_for_test', m['train_seasons'])}, "
            f"tested on {m['test_seasons']}. No random splits anywhere."
        )
        t = m.get("test", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC (test)", f"{t.get('roc_auc', float('nan')):.3f}")
        c2.metric("PR-AUC (test)", f"{t.get('pr_auc', float('nan')):.3f}",
                  f"base rate {t.get('base_rate', float('nan')):.3f}")
        c3.metric("Precision", f"{t.get('precision', float('nan')):.3f}")
        c4.metric("Recall", f"{t.get('recall', float('nan')):.3f}")
        st.warning(
            f"**Base rate is {t.get('base_rate', 0)*100:.1f}%** - pit stops are "
            f"rare per lap. Accuracy is meaningless on this problem; a model "
            f"predicting 'never pits' would score "
            f"{(1-t.get('base_rate', 0))*100:.1f}% accuracy and be useless. "
            f"PR-AUC against the base rate is the number that matters."
        )

        st.markdown("---")
        yr = st.selectbox("Season", sorted(df["year"].unique()), key="y3",
                          index=len(sorted(df["year"].unique())) - 1)
        races = df[df["year"] == yr][["round", "event_name"]].drop_duplicates().sort_values("round")
        ev = st.selectbox("Race", races["event_name"].tolist(), key="r3")
        r = df[(df["year"] == yr) & (df["event_name"] == ev)]
        drv = st.selectbox("Driver", sorted(r["driver"].unique()), key="d3")
        rd = r[r["driver"] == drv].sort_values("lap_number")

        if yr in m["train_seasons"] or yr in m.get("refit_seasons_for_test", []):
            st.info(f"Season {yr} is part of this model's TRAINING data. "
                    f"Predictions shown here are in-sample. Select "
                    f"{m['test_seasons']} for a genuine out-of-sample view.")

        if not rd.empty:
            prob = pit_model["pipeline"].predict_proba(rd[pit_model["features"]])[:, 1]
            out = pd.DataFrame({
                "lap": rd["lap_number"].to_numpy(),
                "predicted P(pit within 3 laps)": prob,
                "actually pitted within 3 laps": rd["pits_within_horizon"].to_numpy(),
            }).set_index("lap")
            st.line_chart(out, x_label="Lap")
            st.caption("Blue = model probability. Orange = ground truth (0/1). "
                       "Laps missing from the x-axis were filtered out of the "
                       "modelling set (pit, safety-car or inaccurate laps).")
            st.dataframe(
                rd[["lap_number", "compound", "tyre_life", "position",
                    "lap_time_vs_field_baseline", "stops_so_far",
                    "pits_within_horizon"]].assign(pred=prob.round(3)).round(3),
                hide_index=True, width='stretch', height=280,
            )

# ============================================================== TAB 4
with tab4:
    st.subheader("Data cleaning and validation")
    if attrition is not None:
        st.markdown("**Lap attrition** - what each cleaning filter removed")
        st.dataframe(attrition, hide_index=True, width='stretch')
    st.markdown("**Modelling assumptions**")
    st.markdown(
        "- Fuel correction: `lap_time - fuel_remaining_kg * 0.03`, with "
        "`fuel_remaining_kg = 100 * (total_laps - lap) / total_laps`. "
        "The 0.03 s/kg figure is an industry rule of thumb, not a measurement, "
        "and its sensitivity is untested.\n"
        "- Wet races are dropped entirely (any logged rainfall). Wet-compound "
        "degradation is different physics.\n"
        "- Track evolution: per-race field median fuel-corrected lap time by "
        "lap number, smoothed with a centred 5-lap rolling median."
    )
    if leak is not None:
        st.markdown("**Leakage checklist** (verified mechanically by "
                    "`src/leakage_check.py`)")
        st.dataframe(leak, hide_index=True, width='stretch')
    cv = load_csv("expanding_window_cv.csv")
    if cv is not None:
        st.markdown("**Expanding-window validation** - train on <= Y, test on Y+1")
        st.dataframe(cv, hide_index=True, width='stretch')
