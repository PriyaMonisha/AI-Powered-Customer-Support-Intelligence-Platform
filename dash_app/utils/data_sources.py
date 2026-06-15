# filename: dash_app/utils/data_sources.py
# purpose:  Static JSON artifact loaders for the Dash dashboard (artifacts/reports,
#           artifacts/metrics) — all loaders are defensive and never raise.
# version:  1.0

import json
import logging
from pathlib import Path

from config import CONFIDENCE_THRESHOLDS_PATH, DRIFT_SUMMARY_PATH, MODEL_REGISTRY_PATH, REPORTS_DIR

log = logging.getLogger(__name__)

# Section 9's completed Colab run scored DistilBERT at test F1-macro=0.1954 —
# statistically tied with LGBM's val F1-macro=0.1997 (both at the ~0.20 noise
# floor for 5-class Ticket Type). section_10_consolidation.json predates that
# run and still claims "DistilBERT is primary classifier" — this correction is
# rendered alongside the artifact note rather than rewriting it.
DASHBOARD_CORRECTIONS = [
    "Section 9 (DistilBERT, completed after this report was generated): test "
    "F1-macro = 0.1954, statistically tied with LGBM's val F1-macro = 0.1997 — "
    "both at the ~0.20 noise floor for 5-class Ticket Type. The note above "
    "('DistilBERT is primary classifier') is outdated; the served model "
    "(lgbm_type_classifier) is the correct choice and was not changed.",
]


def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("Could not load %s: %s", path, e)
        return {}


def load_model_registry() -> dict:
    return _load_json(MODEL_REGISTRY_PATH)


def load_section10_consolidation() -> dict:
    return _load_json(REPORTS_DIR / "section_10_consolidation.json")


def load_confidence_thresholds() -> dict:
    return _load_json(CONFIDENCE_THRESHOLDS_PATH)


def load_drift_metrics() -> dict:
    return _load_json(DRIFT_SUMMARY_PATH)


def load_model_report() -> dict:
    """Most recent artifacts/reports/model_report_<YYYY-MM-DD>.json — lexicographic
    sort on the trailing ISO date string is chronological, no datetime parsing needed."""
    candidates = sorted(
        REPORTS_DIR.glob("model_report_*.json"),
        key=lambda p: p.stem.split("_")[-1],
        reverse=True,
    )
    if not candidates:
        return _load_json(REPORTS_DIR / "model_report_2026-06-12.json")
    return _load_json(candidates[0])


def get_dashboard_notes() -> tuple[list[str], list[str]]:
    """Returns (artifact_notes, dashboard_corrections) as two separate lists —
    artifact_notes is section_10_consolidation.json["notes"] verbatim,
    dashboard_corrections annotates outdated artifact notes without rewriting them."""
    consolidation = load_section10_consolidation()
    artifact_notes = consolidation.get("notes", [])
    return artifact_notes, DASHBOARD_CORRECTIONS
