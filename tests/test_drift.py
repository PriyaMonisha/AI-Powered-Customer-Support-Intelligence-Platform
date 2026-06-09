# filename: tests/test_drift.py
# purpose:  PSI drift module — categorical/continuous PSI math + check_drift orchestration
# version:  1.0

import json

import numpy as np
import pandas as pd
import pytest

from config import DRIFT_PSI_THRESHOLD
from src.monitoring.drift import (
    _psi_categorical,
    _psi_continuous,
    check_drift,
    load_baseline,
)


# --- _psi_categorical --------------------------------------------------------------

def test_psi_categorical_identical_distributions_is_near_zero():
    ref = {"0.0": 0.5, "1.0": 0.3, "2.0": 0.2}
    cur = {"0.0": 0.5, "1.0": 0.3, "2.0": 0.2}
    assert _psi_categorical(ref, cur) < 0.01


def test_psi_categorical_shifted_distribution_exceeds_threshold():
    ref = {"0.0": 0.5, "1.0": 0.3, "2.0": 0.2}
    cur = {"0.0": 0.05, "1.0": 0.05, "2.0": 0.9}
    assert _psi_categorical(ref, cur) > DRIFT_PSI_THRESHOLD


def test_psi_categorical_unseen_category_is_finite_and_positive():
    ref = {"A": 0.5, "B": 0.5}
    cur = {"A": 0.4, "C": 0.6}  # "C" absent from reference, "B" absent from current

    psi = _psi_categorical(ref, cur)

    # A category fully appearing/disappearing legitimately produces a large PSI (the
    # epsilon floor makes the log-ratio term blow up) — that *is* the drift signal.
    # The meaningful guarantees are: finite (epsilon prevents log(0)/div-by-zero),
    # positive (a real distributional change occurred), and not astronomically large
    # (renormalization keeps it bounded rather than diverging to the epsilon floor's
    # full -13.8 log-magnitude per category).
    assert not np.isnan(psi)
    assert not np.isinf(psi)
    assert 0.0 < psi < 50.0


# --- _psi_continuous ----------------------------------------------------------------

_REF_STATS = {"min": 0.0, "p25": 25.0, "p50": 50.0, "p75": 75.0, "max": 100.0}


def test_psi_continuous_identical_distribution_is_near_zero():
    rng = np.random.default_rng(42)
    current = rng.uniform(0, 100, size=2000)
    assert _psi_continuous(_REF_STATS, current) < 0.05


def test_psi_continuous_shifted_distribution_exceeds_threshold():
    rng = np.random.default_rng(42)
    # Current values concentrated far outside the reference's central bins
    current = rng.uniform(90, 100, size=2000)
    assert _psi_continuous(_REF_STATS, current) > DRIFT_PSI_THRESHOLD


def test_psi_continuous_handles_skewed_reference_with_duplicate_percentiles():
    # p25 == p50 — common for zero-inflated features like days_since_purchase
    skewed_ref = {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 10.0, "max": 100.0}
    current = np.array([0.0, 0.0, 0.0, 5.0, 50.0])

    psi = _psi_continuous(skewed_ref, current)

    assert not np.isnan(psi)
    assert not np.isinf(psi)
    assert psi >= 0.0


def test_psi_continuous_out_of_range_values_land_in_boundary_bins():
    # Values far outside [min, max] must be counted, not silently dropped by np.histogram
    current = np.array([-500.0, 500.0, 50.0])
    psi = _psi_continuous(_REF_STATS, current)
    assert not np.isnan(psi)
    assert psi > 0.0


def test_psi_continuous_degenerate_feature_returns_zero():
    constant_ref = {"min": 5.0, "p25": 5.0, "p50": 5.0, "p75": 5.0, "max": 5.0}
    assert _psi_continuous(constant_ref, np.array([5.0, 5.0, 5.0])) == 0.0


# --- check_drift ---------------------------------------------------------------------

@pytest.fixture
def toy_baseline():
    return {
        "stats": {
            "cat_feature": {
                "mean": 1.0, "std": 0.8, "min": 0.0, "max": 2.0,
                "p25": 0.0, "p50": 1.0, "p75": 2.0,
                "value_frequencies": {"0.0": 0.4, "1.0": 0.4, "2.0": 0.2},
            },
            "num_feature": {
                "mean": 50.0, "std": 28.0,
                "min": 0.0, "p25": 25.0, "p50": 50.0, "p75": 75.0, "max": 100.0,
            },
        }
    }


def test_check_drift_returns_expected_keys(toy_baseline):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "cat_feature": rng.choice([0.0, 1.0, 2.0], size=500, p=[0.4, 0.4, 0.2]),
        "num_feature": rng.uniform(0, 100, size=500),
    })

    result = check_drift(df, toy_baseline)

    assert set(result) == {"feature_scores", "max_psi", "n_drifted", "drift_detected", "checked_at"}
    assert set(result["feature_scores"]) == {"cat_feature", "num_feature"}
    assert result["max_psi"] == max(result["feature_scores"].values())


def test_check_drift_flags_drift_when_distribution_shifts(toy_baseline):
    rng = np.random.default_rng(7)
    drifted_df = pd.DataFrame({
        "cat_feature": np.full(500, 2.0),          # was 40/40/20 — now 100% category "2"
        "num_feature": rng.uniform(95, 100, size=500),  # was uniform[0,100] — now squeezed to top
    })

    result = check_drift(drifted_df, toy_baseline)

    assert result["drift_detected"] is True
    assert result["max_psi"] > DRIFT_PSI_THRESHOLD
    assert result["n_drifted"] >= 1


def test_check_drift_no_drift_for_matching_distribution(toy_baseline):
    rng = np.random.default_rng(7)
    matching_df = pd.DataFrame({
        "cat_feature": rng.choice([0.0, 1.0, 2.0], size=2000, p=[0.4, 0.4, 0.2]),
        "num_feature": rng.uniform(0, 100, size=2000),
    })

    result = check_drift(matching_df, toy_baseline)

    assert result["drift_detected"] is False
    assert result["n_drifted"] == 0


def test_check_drift_skips_missing_feature_without_raising(toy_baseline):
    df = pd.DataFrame({"num_feature": np.linspace(0, 100, 500)})  # cat_feature absent
    result = check_drift(df, toy_baseline)
    assert "cat_feature" not in result["feature_scores"]
    assert "num_feature" in result["feature_scores"]


# --- load_baseline --------------------------------------------------------------------

def test_load_baseline_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="notebooks/06_advanced_ml.py"):
        load_baseline(path=tmp_path / "does_not_exist.json")


def test_load_baseline_malformed_json_raises_with_hint(tmp_path):
    bad_file = tmp_path / "baseline.json"
    bad_file.write_text("{not valid json")
    with pytest.raises(ValueError, match="malformed"):
        load_baseline(path=bad_file)


def test_load_baseline_loads_real_artifact():
    baseline = load_baseline()
    assert "stats" in baseline
    assert "feature_names" in baseline
    assert len(baseline["stats"]) == len(baseline["feature_names"])


def test_load_baseline_real_artifact_round_trips_with_check_drift():
    """Sanity check against the actual Section 6 artifact + a synthetic current sample."""
    baseline = load_baseline()
    feature_names = baseline["feature_names"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(0, 1, size=(200, len(feature_names))), columns=feature_names)

    result = check_drift(df, baseline)

    assert len(result["feature_scores"]) == len(feature_names)
    assert isinstance(result["drift_detected"], bool)
