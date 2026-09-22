#!/usr/bin/env bash
# Full pipeline, in dependency order. Safe to re-run: ingest skips races
# already in data/raw, everything downstream is deterministic.
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python

$PY src/ingest_resumable.py   # rate-limit aware; use src/ingest.py for a single pass
$PY src/features.py
$PY src/degradation.py        # exits non-zero if SOFT is not the fastest-degrading compound
$PY src/models.py
$PY src/leakage_check.py
$PY src/report.py
