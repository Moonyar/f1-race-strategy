"""Paths, season splits, and the modelling constants, each with its reasoning."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"

for _d in (CACHE_DIR, DATA_DIR, RAW_DIR, MODEL_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- seasons
# 2022 is the start of the ground-effect regulation era. The 2022 rule change
# altered aerodynamics, car weight and tyre construction (18" wheels arrived in
# 2022), so tyre degradation behaviour before and after is not the same process.
# Restricting to 2022+ is a modelling decision, not only a runtime shortcut.
SEASONS = [2022, 2023, 2024]

# Time-based splits. No random shuffling anywhere in this project.
TRAIN_SEASONS = [2022]
VALID_SEASONS = [2023]
TEST_SEASONS = [2024]

# ---------------------------------------------------------------- fuel model
# Regulation maximum fuel load is 110 kg; teams start near but under it.
FUEL_START_KG = 100.0
# Widely used rule of thumb is ~0.03-0.035 s per kg of fuel, circuit dependent.
# Using the low end. This is an assumption, not a measurement, and the
# sensitivity of the degradation slopes to it is not tested.
SECONDS_PER_KG = 0.03

# ---------------------------------------------------------------- cleaning
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]

# TrackStatus digits that mean the lap was not run at full racing speed.
# 1 green, 2 yellow, 4 safety car, 5 red flag, 6 VSC deployed, 7 VSC ending.
NON_GREEN_STATUS_DIGITS = set("4567")

# Laps outside this window are timing artefacts or laps behind a slow zone that
# TrackStatus missed. Applied as a per-session relative filter, not absolute.
LAP_TIME_MIN_RATIO = 0.90   # faster than 90% of session median => suspicious
LAP_TIME_MAX_RATIO = 1.20   # slower than 120% of session median => not a green lap

# ---------------------------------------------------------------- pit window
# "Does this driver pit within the next PIT_HORIZON laps?"
PIT_HORIZON = 3

RANDOM_STATE = 42
