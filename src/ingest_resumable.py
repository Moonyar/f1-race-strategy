"""Rate-limit-aware driver around ingest.ingest_race.

The FastF1 upstream API allows 500 requests per rolling hour. A race costs
roughly 20 requests on first download and ~0 once cached, so a full multi-season
backfill has to be spread across several hourly windows.

This script does NOT try to defeat the limit (restarting the process would
reset the in-process counter, but the limit exists to protect a shared public
API and FastF1's own source warns that abusing it gets everyone blocked). It
waits the window out and resumes. Races already written to data/raw are skipped,
so it is safe to stop and restart at any point.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import fastf1
from fastf1.exceptions import RateLimitExceededError

from config import RAW_DIR, SEASONS
from ingest import enable_cache, ingest_race

SLEEP_ON_LIMIT_S = 3660  # one rolling hour plus a minute of slack


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=SEASONS)
    ap.add_argument("--max-waits", type=int, default=6)
    args = ap.parse_args()

    enable_cache()

    todo: list[tuple[int, int, str]] = []
    for year in args.seasons:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in sched.iterrows():
            rnd = int(ev["RoundNumber"])
            if rnd == 0:
                continue
            if (RAW_DIR / f"{year}_{rnd:02d}.parquet").exists():
                continue
            todo.append((year, rnd, ev["EventName"]))

    log(f"{len(todo)} races still to ingest")
    waits = 0
    ok = permafail = 0
    while todo and waits <= args.max_waits:
        remaining = []
        for year, rnd, name in todo:
            out = RAW_DIR / f"{year}_{rnd:02d}.parquet"
            if out.exists():
                continue
            try:
                df = ingest_race(year, rnd, with_telemetry=True)
                if df is None or df.empty:
                    log(f"  EMPTY  {year} r{rnd:02d} {name}")
                    permafail += 1
                    continue
                df.to_parquet(out, index=False)
                ntel = int(df["max_speed"].notna().sum()) if "max_speed" in df else 0
                log(f"  ok     {year} r{rnd:02d} {name:<34s} laps={len(df):5d} tel={ntel:5d}")
                ok += 1
            except RateLimitExceededError:
                log(f"  RATE LIMIT hit at {year} r{rnd:02d}. "
                    f"{len(remaining) + 1} races left in this pass.")
                remaining.append((year, rnd, name))
                remaining += todo[todo.index((year, rnd, name)) + 1:]
                break
            except Exception as e:  # noqa: BLE001
                log(f"  FAIL   {year} r{rnd:02d} {name}: {type(e).__name__}: {e}")
                permafail += 1
        else:
            remaining = []

        todo = [t for t in remaining
                if not (RAW_DIR / f"{t[0]}_{t[1]:02d}.parquet").exists()]
        if todo:
            waits += 1
            log(f"sleeping {SLEEP_ON_LIMIT_S}s for the rate-limit window to "
                f"clear (wait {waits}/{args.max_waits}); {len(todo)} races left")
            time.sleep(SLEEP_ON_LIMIT_S)

    log(f"DONE ok={ok} permanently_failed={permafail} still_todo={len(todo)}")
    if todo:
        log("remaining: " + ", ".join(f"{y}r{r}" for y, r, _ in todo[:30]))


if __name__ == "__main__":
    main()
