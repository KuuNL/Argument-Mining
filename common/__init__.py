"""Shared foundation for all approaches: loading, validating, scoring, splitting, timing (PLAN Phase 0)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SPLITS = ROOT / "splits"
