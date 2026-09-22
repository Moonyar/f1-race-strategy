"""Ingest F1 race sessions from the FastF1 API into per-race parquet files.

One parquet per race in data/raw/. Full-season telemetry does not fit in memory
at once, so each race is loaded, reduced to one row per driver-lap, and written
out before the next race is touched.

Three sources are joined here:
  * lap timing   - session.laps
  * telemetry    - session.car_data, aggregated per driver-lap
  * weather      - session.weather_data, merge_asof on nearest timestamp
plus session.results for grid position and final classification.
"""
from __future__ import annotations

import argparse
import logging
import warnings

import fastf1
import numpy as np
import pandas as pd

from config import RAW_DIR, CACHE_DIR, SEASONS

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)
log = logging.getLogger("ingest")

# DRS channel: values 10, 12, 14 mean the flap is open. 0/1/8 mean closed or
# merely eligible. This mapping is FastF1's documented interpretation.
DRS_OPEN_VALUES = {10, 12, 14}


def enable_cache() -> None:
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def aggregate_telemetry(session) -> pd.DataFrame:
    """Reduce per-sample car telemetry to one row per driver-lap.

    Telemetry is sampled several times a second. Binning it onto lap boundaries
    with searchsorted is far cheaper than calling lap.get_car_data() per lap,
    which re-slices the whole channel every time.
    """
    rows = []
    laps = session.laps
    for drv, car in session.car_data.items():
        drv_laps = laps[laps["DriverNumber"] == drv]
        if drv_laps.empty or car.empty:
            continue

        car = car.sort_values("SessionTime")
        t = car["SessionTime"].to_numpy()

        starts = drv_laps["LapStartTime"].to_numpy()
        ends = drv_laps["Time"].to_numpy()
        lap_nums = drv_laps["LapNumber"].to_numpy()

        speed = car["Speed"].to_numpy(dtype="float64")
        throttle = np.clip(car["Throttle"].to_numpy(dtype="float64"), 0, 100)
        brake = car["Brake"].to_numpy().astype(bool)
        drs_open = np.isin(car["DRS"].to_numpy(), list(DRS_OPEN_VALUES))
        rpm = car["RPM"].to_numpy(dtype="float64")
        gear = car["nGear"].to_numpy(dtype="float64")

        for start, end, ln in zip(starts, ends, lap_nums):
            if pd.isna(start) or pd.isna(end):
                continue
            i0, i1 = np.searchsorted(t, [start, end])
            if i1 - i0 < 20:          # too few samples to trust an average
                continue
            sl = slice(i0, i1)
            rows.append(
                {
                    "DriverNumber": drv,
                    "LapNumber": float(ln),
                    "max_speed": float(np.nanmax(speed[sl])),
                    "mean_speed": float(np.nanmean(speed[sl])),
                    "mean_throttle": float(np.nanmean(throttle[sl])),
                    "full_throttle_pct": float(np.mean(throttle[sl] > 95) * 100),
                    "brake_pct": float(np.mean(brake[sl]) * 100),
                    "drs_pct": float(np.mean(drs_open[sl]) * 100),
                    "max_rpm": float(np.nanmax(rpm[sl])),
                    "mean_gear": float(np.nanmean(gear[sl])),
                    "n_telemetry_samples": int(i1 - i0),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["DriverNumber", "LapNumber", "max_speed", "mean_speed",
                     "mean_throttle", "full_throttle_pct", "brake_pct",
                     "drs_pct", "max_rpm", "mean_gear", "n_telemetry_samples"]
        )
    return pd.DataFrame(rows)


def join_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest weather observation to each lap.

    Weather is logged roughly once a minute and laps last 60-110s, so the
    nearest observation is used. The join key is LapStartTime, so a lap is
    described by the conditions at its start, never by end-of-race conditions.
    """
    wcols = ["Time", "AirTemp", "TrackTemp", "Humidity", "Pressure",
             "Rainfall", "WindSpeed", "WindDirection"]
    w = weather[[c for c in wcols if c in weather.columns]].copy()
    w = w.sort_values("Time")
    l = laps.sort_values("LapStartTime").copy()
    l = l[l["LapStartTime"].notna()]
    merged = pd.merge_asof(
        l, w, left_on="LapStartTime", right_on="Time",
        direction="nearest", suffixes=("", "_weather"),
    )
    return merged


def ingest_race(year: int, rnd: int, with_telemetry: bool = True) -> pd.DataFrame | None:
    """Load one race and return a tidy one-row-per-driver-lap frame."""
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=with_telemetry, weather=True, messages=False)

    laps = session.laps
    if laps is None or laps.empty:
        return None

    keep = ["Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint",
            "Compound", "TyreLife", "FreshTyre", "PitInTime", "PitOutTime",
            "TrackStatus", "Position", "IsAccurate", "LapStartTime",
            "LapStartDate", "Time", "Deleted",
            "Sector1Time", "Sector2Time", "Sector3Time", "SpeedST", "SpeedFL"]
    df = laps[[c for c in keep if c in laps.columns]].copy()

    df = join_weather(df, session.weather_data)

    if with_telemetry:
        tel = aggregate_telemetry(session)
        if not tel.empty:
            df = df.merge(tel, on=["DriverNumber", "LapNumber"], how="left")

    # Race-level result columns. GridPosition is known before the race and is a
    # legitimate feature. Status / final Position are targets or post-race only.
    res = session.results
    if res is not None and not res.empty:
        r = res[["DriverNumber", "GridPosition", "ClassifiedPosition",
                 "Position", "Status", "Points", "TeamName"]].copy()
        r = r.rename(columns={"Position": "FinalPosition",
                              "ClassifiedPosition": "FinalClassified"})
        df = df.merge(r, on="DriverNumber", how="left")

    ev = session.event
    df["year"] = year
    df["round"] = int(ev["RoundNumber"])
    df["event_name"] = ev["EventName"]
    df["circuit"] = ev["Location"]
    df["session_date"] = pd.to_datetime(ev["EventDate"])
    df["total_laps"] = float(session.total_laps) if session.total_laps else float(df["LapNumber"].max())

    # Timedeltas -> seconds so parquet and downstream maths stay simple.
    for c in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if c in df.columns:
            df[c + "_s"] = df[c].dt.total_seconds()
    df["is_pit_in"] = df["PitInTime"].notna()
    df["is_pit_out"] = df["PitOutTime"].notna()

    drop = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "PitInTime", "PitOutTime", "LapStartTime", "Time", "Time_weather"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS)
    ap.add_argument("--no-telemetry", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    enable_cache()
    ok = fail = skip = 0
    for year in args.seasons:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in sched.iterrows():
            rnd = int(ev["RoundNumber"])
            if rnd == 0:
                continue
            out = RAW_DIR / f"{year}_{rnd:02d}.parquet"
            if out.exists() and not args.force:
                skip += 1
                continue
            try:
                df = ingest_race(year, rnd, with_telemetry=not args.no_telemetry)
                if df is None or df.empty:
                    print(f"  EMPTY  {year} r{rnd:02d} {ev['EventName']}")
                    fail += 1
                    continue
                df.to_parquet(out, index=False)
                ntel = int(df["max_speed"].notna().sum()) if "max_speed" in df else 0
                print(f"  ok     {year} r{rnd:02d} {ev['EventName']:<32s} "
                      f"laps={len(df):5d} tel={ntel:5d}")
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL   {year} r{rnd:02d} {ev['EventName']}: {type(e).__name__}: {e}")
                fail += 1
    print(f"\ningest done: ok={ok} failed={fail} skipped={skip}")


if __name__ == "__main__":
    main()
