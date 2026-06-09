# filename: src/monitoring/drift.py
# purpose:  PSI-based feature drift detection against the Section 6 training baseline.
#           Lightweight by design — runs synchronously inside an API request handler
#           (off the event loop via run_in_executor). Evidently's heavier DataDriftPreset
#           runs separately, offline, in notebooks/12_monitoring.py.
# version:  1.0

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import DRIFT_BASELINE_PATH, DRIFT_PSI_THRESHOLD

logger = logging.getLogger(__name__)

EPSILON = 1e-6


def load_baseline(path: "Path | None" = None) -> dict:
    """Loads the Section 6 drift baseline JSON ({"stats": {feature: {...}}, ...})."""
    baseline_path = path or DRIFT_BASELINE_PATH
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Drift baseline not found at {baseline_path}. "
            "Regenerate via notebooks/06_advanced_ml.py."
        )
    try:
        with open(baseline_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Drift baseline at {baseline_path} is malformed: {e}. "
            "Delete and regenerate via notebooks/06_advanced_ml.py."
        ) from e


def _psi_categorical(ref_freqs: dict, cur_freqs: dict) -> float:
    """PSI between two category -> proportion dicts (e.g. baseline value_frequencies)."""
    categories = sorted(set(ref_freqs) | set(cur_freqs))
    ref_vec = np.array([ref_freqs.get(c, EPSILON) for c in categories], dtype=float)
    cur_vec = np.array([cur_freqs.get(c, EPSILON) for c in categories], dtype=float)

    # Renormalize AFTER epsilon injection — epsilon alone breaks the
    # probability-distribution assumption PSI relies on, biasing every score.
    ref_vec = np.maximum(ref_vec, EPSILON)
    cur_vec = np.maximum(cur_vec, EPSILON)
    ref_vec /= ref_vec.sum()
    cur_vec /= cur_vec.sum()

    psi = np.sum((cur_vec - ref_vec) * np.log(cur_vec / ref_vec))
    return float(max(psi, 0.0))


def _psi_continuous(ref_stats: dict, current_values: np.ndarray) -> float:
    """
    PSI for a continuous feature using quantile bins derived from the baseline's
    percentile summary (min/p25/p50/p75/max) — the only reference shape available
    without raw rows.

    Reference proportions are taken as uniform across the resulting bins. This is an
    approximation: true quartile proportions are exactly 25% only for continuous
    distributions without ties. For heavily skewed features (e.g. days_since_purchase,
    where p25 == p50), deduplication shrinks the bin count and the uniform assumption
    holds better than it first appears — documented here rather than hidden.
    """
    raw_edges = sorted(
        {
            ref_stats["min"],
            ref_stats.get("p25", ref_stats["min"]),
            ref_stats["p50"],
            ref_stats.get("p75", ref_stats["max"]),
            ref_stats["max"],
        }
    )
    if len(raw_edges) < 2:
        return 0.0  # degenerate (constant) feature — nothing to compare

    # Open the outer edges to +/-inf so production values outside the training range
    # land in the boundary bins instead of silently vanishing from np.histogram.
    edges = [-np.inf] + raw_edges[1:-1] + [np.inf]
    n_bins = len(edges) - 1

    ref_vec = np.full(n_bins, 1.0 / n_bins)
    cur_counts, _ = np.histogram(current_values, bins=edges)
    cur_vec = cur_counts / max(len(current_values), 1)

    ref_vec = np.maximum(ref_vec, EPSILON)
    cur_vec = np.maximum(cur_vec, EPSILON)
    ref_vec /= ref_vec.sum()
    cur_vec /= cur_vec.sum()

    psi = np.sum((cur_vec - ref_vec) * np.log(cur_vec / ref_vec))
    return float(max(psi, 0.0))


def check_drift(current_df: pd.DataFrame, baseline: "dict | None" = None) -> dict:
    """
    Computes per-feature PSI between `current_df` and the training baseline.

    Returns:
        {
            "feature_scores": {feature: psi, ...}   # sorted by PSI descending
            "max_psi": float,
            "n_drifted": int,
            "drift_detected": bool,
            "checked_at": "<UTC ISO-8601>",
        }
    """
    if baseline is None:
        baseline = load_baseline()
    stats = baseline.get("stats", baseline)

    feature_scores: dict[str, float] = {}
    for feature, ref_stats in stats.items():
        if feature not in current_df.columns:
            logger.warning("Baseline feature %r not present in current data — skipping", feature)
            continue

        values = current_df[feature].dropna().values
        if len(values) == 0:
            logger.warning("Feature %r has no current values — skipping", feature)
            continue

        if "value_frequencies" in ref_stats:
            unique, counts = np.unique(values.astype(str), return_counts=True)
            cur_freqs = {k: v / len(values) for k, v in zip(unique, counts)}
            score = _psi_categorical(ref_stats["value_frequencies"], cur_freqs)
        else:
            score = _psi_continuous(ref_stats, values.astype(float))

        feature_scores[feature] = round(score, 6)

    # Most-drifted first — more useful to a caller than insertion/alphabetical order.
    feature_scores = dict(sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True))

    max_psi = max(feature_scores.values(), default=0.0)
    n_drifted = sum(1 for v in feature_scores.values() if v > DRIFT_PSI_THRESHOLD)

    return {
        "feature_scores": feature_scores,
        "max_psi": round(max_psi, 6),
        "n_drifted": n_drifted,
        "drift_detected": max_psi > DRIFT_PSI_THRESHOLD,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
