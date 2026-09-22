"""Clean raw race parquets and build the modelling feature table.

Pipeline, in order:
  1. concatenate data/raw/*.parquet
  2. clean  - accurate green-flag racing laps only
  3. fuel correction
  4. track evolution baseline
  5. derived stint / pace / pit features
  6. write data/features.parquet

Everything downstream reads data/features.parquet and nothing else.
"""
from __future__ import annotations

import pandas as pd

from config import (
    RAW_DIR, DATA_DIR, DRY_COMPOUNDS, FUEL_START_KG, SECONDS_PER_KG,
    LAP_TIME_MAX_RATIO, LAP_TIME_MIN_RATIO, NON_GREEN_STATUS_DIGITS, PIT_HORIZON,
)

RACE_KEY = ["year", "round"]
DRIVER_KEY = ["year", "round", "driver"]


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw parquets in {RAW_DIR}; run src/ingest.py first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.rename(columns={
        "Driver": "driver", "Team": "team", "LapNumber": "lap_number",
        "Stint": "stint", "Compound": "compound", "TyreLife": "tyre_life",
        "LapTime_s": "lap_time", "AirTemp": "air_temp", "TrackTemp": "track_temp",
        "Humidity": "humidity", "Rainfall": "rainfall", "WindSpeed": "wind_speed",
        "Position": "position", "GridPosition": "grid_position",
        "FinalPosition": "final_position", "FinalClassified": "final_classified",
        "Status": "status", "TeamName": "team_name", "IsAccurate": "is_accurate",
        "TrackStatus": "track_status",
    })
    return df


def _non_green(status: object) -> bool:
    """TrackStatus is a concatenation of digits seen during the lap."""
    if pd.isna(status):
        return True
    return any(d in NON_GREEN_STATUS_DIGITS for d in str(status))


def clean(df: pd.DataFrame, verbose: bool = True,
          drop_wet: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to laps that represent genuine green-flag racing pace.

    Returns (clean_laps, audit_table). The audit table records how many laps
    each filter removed; it is the attrition table in the README.
    """
    steps = []
    n0 = len(df)
    steps.append(("raw driver-laps ingested", n0, 0))

    d = df.copy()
    d["lap_number"] = d["lap_number"].astype(float)

    # A race is "wet" if any rainfall was logged during it. Wet-compound
    # degradation is different physics; we model dry racing only and say so.
    wet_race = d.groupby(RACE_KEY)["rainfall"].transform(
        lambda s: bool(s.fillna(False).any())
    )
    d["wet_race"] = wet_race

    def step(mask, label):
        nonlocal d
        before = len(d)
        d = d[mask(d)].copy()
        steps.append((label, len(d), before - len(d)))

    # Order matters for the audit: run the explicit physical filters FIRST so
    # the table shows what each one actually removes. FastF1's IsAccurate flag
    # happens to subsume all of them (verified: 0 pit laps and 0 non-green laps
    # carry IsAccurate == True), but relying on that would make the cleaning
    # depend on an undocumented internal definition. It runs last as a backstop.
    step(lambda x: x["lap_time"].notna(), "have a timed lap")
    step(lambda x: ~x["is_pit_in"].fillna(False).astype(bool), "drop in-laps (PitInTime)")
    step(lambda x: ~x["is_pit_out"].fillna(False).astype(bool), "drop out-laps (PitOutTime)")
    step(lambda x: ~x["track_status"].map(_non_green), "drop SC / VSC / red-flag laps")
    step(lambda x: x["is_accurate"].fillna(False).astype(bool), "IsAccurate == True (backstop)")
    step(lambda x: x["compound"].isin(DRY_COMPOUNDS), "dry slick compounds only")
    if drop_wet:
        step(lambda x: ~x["wet_race"], "drop races with logged rainfall")
    step(lambda x: x["tyre_life"].notna() & (x["tyre_life"] > 0), "have tyre age")

    # Relative outlier filter: laps far from the session median are traffic,
    # damage or timing artefacts that TrackStatus did not flag.
    med = d.groupby(RACE_KEY)["lap_time"].transform("median")
    d["lap_time_vs_session_median"] = d["lap_time"] / med
    step(lambda x: x["lap_time_vs_session_median"].between(LAP_TIME_MIN_RATIO,
                                                           LAP_TIME_MAX_RATIO),
         f"lap within [{LAP_TIME_MIN_RATIO}, {LAP_TIME_MAX_RATIO}]x session median")

    # Hard guarantees the degradation fit depends on. A surviving in-lap is
    # 20+ seconds slow and would wreck every slope, so this is an assert, not
    # a comment.
    assert not d["is_pit_in"].fillna(False).any(), "in-laps survived cleaning"
    assert not d["is_pit_out"].fillna(False).any(), "out-laps survived cleaning"
    assert not d["track_status"].map(_non_green).any(), "non-green laps survived cleaning"
    assert d["lap_time"].notna().all(), "untimed laps survived cleaning"

    audit = pd.DataFrame(steps, columns=["step", "laps_remaining", "laps_dropped"])
    audit["pct_of_raw"] = (audit["laps_remaining"] / n0 * 100).round(1)
    if verbose:
        print("\n--- lap attrition ---")
        print(audit.to_string(index=False))
        print(f"retained {len(d)/n0*100:.1f}% of raw laps "
              f"({len(d):,} of {n0:,})")
    return d, audit


def add_fuel_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Correct each lap to an equivalent empty-tank lap time.

        fuel_remaining = FUEL_START_KG * (total_laps - lap_number) / total_laps
        corrected      = lap_time - fuel_remaining * SECONDS_PER_KG

    Cars get lighter and therefore faster through a stint. Without this the
    fuel-burn gain partly cancels tyre degradation and every slope is biased
    toward zero. Corrected times are SLOWER than raw, by design.
    """
    d = df.copy()
    total = d["total_laps"].astype(float)
    d["fuel_remaining_kg"] = (
        FUEL_START_KG * (total - d["lap_number"]).clip(lower=0) / total
    )
    d["lap_time_fuel_corrected"] = d["lap_time"] - d["fuel_remaining_kg"] * SECONDS_PER_KG
    return d


def add_track_evolution(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Per-session track-evolution baseline.

    The circuit rubbers in and the whole field speeds up over a race,
    independently of tyres and fuel. We take the field's median fuel-corrected
    lap time as a function of lap number within each race, smooth it with a
    centred rolling median, and express each lap relative to it.

    Leakage note: this is a within-race aggregate over contemporaneous laps by
    other drivers, which the pit wall does see live. The centred window also
    reaches two laps ahead, so it is not strictly past-only. See leakage_check.py.
    """
    d = df.copy()
    field = (
        d.groupby(RACE_KEY + ["lap_number"])["lap_time_fuel_corrected"]
        .median()
        .rename("field_median_fc")
        .reset_index()
    )
    field["track_evolution_baseline"] = (
        field.groupby(RACE_KEY)["field_median_fc"]
        .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
    )
    # Track evolution measured as gain versus the start of the race.
    first = field.groupby(RACE_KEY)["track_evolution_baseline"].transform("first")
    field["track_evolution_gain_s"] = first - field["track_evolution_baseline"]

    d = d.merge(field, on=RACE_KEY + ["lap_number"], how="left")
    d["lap_time_vs_field_baseline"] = (
        d["lap_time_fuel_corrected"] - d["track_evolution_baseline"]
    )
    return d


def add_stint_and_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver-lap derived features. All strictly backward looking.

    Deliberately absent: total stint length. At lap L a strategist does not
    know how long the stint will end up being - that IS the pit-window answer.
    """
    d = df.sort_values(DRIVER_KEY + ["lap_number"]).copy()
    g = d.groupby(DRIVER_KEY, sort=False)

    d["laps_remaining"] = d["total_laps"] - d["lap_number"]
    d["race_progress"] = d["lap_number"] / d["total_laps"]

    # Mean pace over laps L-2..L (lap L is complete when the decision is made).
    # min_periods=1 keeps early-stint laps usable.
    d["pace_last3"] = g["lap_time_vs_field_baseline"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    # Pace trend: change in the 3-lap mean over the previous 3 laps.
    d["pace_trend_3"] = d["pace_last3"] - g["pace_last3"].shift(3)
    d["lap_time_delta_prev"] = d["lap_time_vs_field_baseline"] - g[
        "lap_time_vs_field_baseline"
    ].shift(1)

    # Position change so far, known at lap L.
    d["position_vs_grid"] = d["grid_position"] - d["position"]
    d["stint_number"] = d["stint"].astype(float)
    return d


def add_pit_labels(df_clean: pd.DataFrame, df_all: pd.DataFrame,
                   horizon: int = PIT_HORIZON) -> pd.DataFrame:
    """Label: does this driver pit within the next `horizon` laps?

    The pit event itself is read from the FULL raw lap table, not the cleaned
    one, because cleaning removes in-laps by construction. Using the cleaned
    table would delete most of the positive class.
    """
    pits = (
        df_all[df_all["is_pit_in"].fillna(False).astype(bool)]
        [DRIVER_KEY + ["lap_number"]]
        .rename(columns={"lap_number": "pit_lap"})
    )
    d = df_clean.copy()
    d["pits_within_horizon"] = 0

    merged = d[DRIVER_KEY + ["lap_number"]].merge(pits, on=DRIVER_KEY, how="left")
    merged["hit"] = (
        (merged["pit_lap"] > merged["lap_number"])
        & (merged["pit_lap"] <= merged["lap_number"] + horizon)
    ).fillna(False)
    hits = merged.groupby(DRIVER_KEY + ["lap_number"])["hit"].any().reset_index()
    d = d.merge(hits, on=DRIVER_KEY + ["lap_number"], how="left")
    d["pits_within_horizon"] = d["hit"].fillna(False).astype(int)
    d = d.drop(columns=["hit"])

    # Number of stops made SO FAR (backward looking, safe at lap L).
    merged2 = d[DRIVER_KEY + ["lap_number"]].merge(pits, on=DRIVER_KEY, how="left")
    merged2["past"] = (merged2["pit_lap"] <= merged2["lap_number"]).fillna(False)
    past = merged2.groupby(DRIVER_KEY + ["lap_number"])["past"].sum().reset_index()
    past = past.rename(columns={"past": "stops_so_far"})
    d = d.merge(past, on=DRIVER_KEY + ["lap_number"], how="left")
    d["stops_so_far"] = d["stops_so_far"].fillna(0).astype(int)

    # Total stops made in the whole race, counted from the RAW lap table so a
    # stop on the final laps is not missed (cleaning removes in-laps, and
    # stops_so_far is capped at the driver's last CLEAN lap).
    # This is a whole-race summary: legitimate for the post-race
    # finishing-position model, and deliberately excluded from the pit-window
    # model's feature list, where it would be a look-ahead.
    tot = pits.groupby(DRIVER_KEY).size().rename("race_total_stops").reset_index()
    d = d.merge(tot, on=DRIVER_KEY, how="left")
    d["race_total_stops"] = d["race_total_stops"].fillna(0).astype(int)
    return d


FEATURE_COLUMNS = [
    "year", "round", "circuit", "event_name", "driver", "team_name", "team",
    "lap_number", "stint_number", "compound", "tyre_life", "fresh_tyre",
    "lap_time", "lap_time_fuel_corrected", "lap_time_vs_field_baseline",
    "field_median_fc", "track_evolution_baseline", "track_evolution_gain_s",
    "fuel_remaining_kg", "pace_last3", "pace_trend_3", "lap_time_delta_prev",
    "air_temp", "track_temp", "humidity", "rainfall", "wind_speed",
    "max_speed", "mean_speed", "mean_throttle", "full_throttle_pct",
    "brake_pct", "drs_pct", "max_rpm", "mean_gear",
    "position", "grid_position", "position_vs_grid",
    "laps_remaining", "race_progress", "total_laps",
    "stops_so_far", "race_total_stops", "pits_within_horizon",
    "final_position", "final_classified", "status",
]


def _assemble(raw: pd.DataFrame, drop_wet: bool, verbose: bool):
    clean_laps, audit = clean(raw, verbose=verbose, drop_wet=drop_wet)
    clean_laps = add_fuel_correction(clean_laps)
    clean_laps = add_track_evolution(clean_laps)
    clean_laps = add_stint_and_pace_features(clean_laps)
    clean_laps = add_pit_labels(clean_laps, raw)
    clean_laps["fresh_tyre"] = clean_laps.get("FreshTyre", pd.Series(index=clean_laps.index)).astype("boolean")
    out = clean_laps[[c for c in FEATURE_COLUMNS if c in clean_laps.columns]].copy()
    return out, audit


def build(verbose: bool = True) -> pd.DataFrame:
    raw = load_raw()
    if verbose:
        print(f"loaded {len(raw):,} raw driver-laps from "
              f"{raw.groupby(RACE_KEY).ngroups} races, "
              f"seasons {sorted(raw['year'].unique())}")

    # Primary table: dry races only. Used for the degradation curves and the
    # pit-window model, where wet-compound physics and wet pit behaviour
    # (everyone stops for intermediates at once) would be a different process.
    out, audit = _assemble(raw, drop_wet=True, verbose=verbose)
    out.to_parquet(DATA_DIR / "features.parquet", index=False)
    audit.to_csv(DATA_DIR / "lap_attrition.csv", index=False)
    if verbose:
        print(f"\nwrote {DATA_DIR / 'features.parquet'}  shape={out.shape}")
        print(f"telemetry coverage: {out['max_speed'].notna().mean()*100:.1f}% of feature laps")
        print(f"pit-window base rate: {out['pits_within_horizon'].mean()*100:.2f}%")

    # Secondary table: wet races KEPT. Excluding them is a decision about
    # modelling tyre degradation, not a decision about predicting where a car
    # finishes - and dropping them costs the race-level model roughly half its
    # training rows, which it cannot afford.
    if verbose:
        print("\n--- all-weather variant (finishing-position model only) ---")
    out_all, audit_all = _assemble(raw, drop_wet=False, verbose=False)
    out_all.to_parquet(DATA_DIR / "features_all_weather.parquet", index=False)
    audit_all.to_csv(DATA_DIR / "lap_attrition_all_weather.csv", index=False)
    if verbose:
        print(f"wrote {DATA_DIR / 'features_all_weather.parquet'}  "
              f"shape={out_all.shape} "
              f"({out_all.groupby(RACE_KEY).ngroups} races vs "
              f"{out.groupby(RACE_KEY).ngroups} dry)")
    return out


if __name__ == "__main__":
    build()
