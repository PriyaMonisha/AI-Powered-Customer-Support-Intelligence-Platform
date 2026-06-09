# filename: notebooks/12_monitoring.py
# purpose:  Evidently 0.4.30 drift reports (offline) + cross-check against the live
#           API's lightweight custom-PSI module (Section 12)
# version:  1.0

FAST_MODE = True   # FIRST LINE

# ---------------------------------------------------------------------------
# Imports + PROJECT_ROOT
# ---------------------------------------------------------------------------
import datetime
import json
import logging
import sys
import warnings
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import mlflow

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from config import (
    ARTIFACTS_DIR,
    DRIFT_PSI_THRESHOLD,
    DRIFT_SUMMARY_PATH,
    FEATURES_DIR,
    MLFLOW_TRACKING_URI,
)
from src.monitoring.drift import check_drift, load_baseline
from src.utils.helpers import NumpyEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,   # Section 9 lesson — some environments pre-attach root handlers
)
logger = logging.getLogger("section12")
logger.info("FAST_MODE=%s", FAST_MODE)

REPORTS_DIR = ARTIFACTS_DIR / "reports"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cell 1 — Load reference (train) / current (test) tabular splits as DataFrames
# ---------------------------------------------------------------------------
with open(FEATURES_DIR / "tabular_columns.json") as fh:
    tab_col_names: list[str] = json.load(fh)

X_train = np.load(FEATURES_DIR / "X_train_tabular.npy")
X_test = np.load(FEATURES_DIR / "X_test_tabular.npy")

df_train = pd.DataFrame(X_train, columns=tab_col_names)
df_test = pd.DataFrame(X_test, columns=tab_col_names)
logger.info("Reference (train): %s | Current (test): %s", df_train.shape, df_test.shape)

baseline = load_baseline()

# Categorical vs. continuous split — taken directly from the baseline's
# value_frequencies presence, the SAME criterion src.monitoring.drift.check_drift uses.
# Keeping both pipelines' notion of "categorical" identical is what makes the
# cross-check below meaningful (otherwise Evidently and the custom PSI module could
# legitimately disagree simply because they're answering slightly different questions).
categorical_features = [f for f, s in baseline["stats"].items() if "value_frequencies" in s]
numerical_features = [f for f, s in baseline["stats"].items() if "value_frequencies" not in s]
logger.info("Categorical (%d): %s", len(categorical_features), categorical_features)
logger.info("Numerical   (%d): %s", len(numerical_features), numerical_features)

column_mapping = ColumnMapping(
    target=None,
    prediction=None,
    numerical_features=numerical_features,
    categorical_features=categorical_features,
)

# ---------------------------------------------------------------------------
# Cell 2 — Evidently report #1: train vs. test (negative control — i.i.d. splits)
# ---------------------------------------------------------------------------
report_baseline = Report(metrics=[DataDriftPreset()])
report_baseline.run(reference_data=df_train, current_data=df_test, column_mapping=column_mapping)

baseline_report_path = REPORTS_DIR / "drift_report_baseline.html"
report_baseline.save_html(str(baseline_report_path))
baseline_evidently_result = report_baseline.as_dict()["metrics"][0]["result"]
baseline_evidently_drifted = int(baseline_evidently_result["number_of_drifted_columns"])
baseline_evidently_share = float(baseline_evidently_result["share_of_drifted_columns"])
logger.info(
    "Evidently (train vs test, expect ~no drift): %d/%d columns drifted (share=%.4f) → %s",
    baseline_evidently_drifted, len(tab_col_names), baseline_evidently_share, baseline_report_path,
)

# ---------------------------------------------------------------------------
# Cell 3 — Principled synthetic shift + Evidently report #2 (positive control)
# ---------------------------------------------------------------------------
SHIFT_FEATURE = "Customer Age"
if SHIFT_FEATURE not in baseline["stats"]:
    raise KeyError(
        f"{SHIFT_FEATURE!r} not found in training_baseline.json stats. "
        f"Available features: {list(baseline['stats'].keys())}"
    )

shift_std = float(baseline["stats"][SHIFT_FEATURE]["std"])
SHIFT_MAGNITUDE_STD_UNITS = 2.0
shift_amount = SHIFT_MAGNITUDE_STD_UNITS * shift_std

df_shifted = df_test.copy()
df_shifted[SHIFT_FEATURE] = df_shifted[SHIFT_FEATURE] + shift_amount
logger.info(
    "Synthetic shift: %s += %.4f (= %.1f x training std %.4f) — a statistically "
    "meaningful, reproducible perturbation, not an arbitrary nudge",
    SHIFT_FEATURE, shift_amount, SHIFT_MAGNITUDE_STD_UNITS, shift_std,
)

report_shifted = Report(metrics=[DataDriftPreset()])
report_shifted.run(reference_data=df_train, current_data=df_shifted, column_mapping=column_mapping)

shifted_report_path = REPORTS_DIR / "drift_report_shifted.html"
report_shifted.save_html(str(shifted_report_path))
shifted_evidently_result = report_shifted.as_dict()["metrics"][0]["result"]
shifted_evidently_drifted = int(shifted_evidently_result["number_of_drifted_columns"])
shifted_evidently_share = float(shifted_evidently_result["share_of_drifted_columns"])
logger.info(
    "Evidently (train vs shifted-test, expect drift flagged): %d/%d columns drifted (share=%.4f) → %s",
    shifted_evidently_drifted, len(tab_col_names), shifted_evidently_share, shifted_report_path,
)

# ---------------------------------------------------------------------------
# Cell 4 — Cross-check: custom PSI module vs. Evidently, on the same two scenarios
# ---------------------------------------------------------------------------
psi_baseline_result = check_drift(df_test, baseline)
psi_shifted_result = check_drift(df_shifted, baseline)

logger.info("")
logger.info("%-28s %12s %12s", "Scenario", "Evidently", "Custom-PSI")
logger.info(
    "%-28s %12s %12s",
    "train vs test (no shift)",
    f"drift={baseline_evidently_drifted > 0}",
    f"drift={psi_baseline_result['drift_detected']}",
)
logger.info(
    "%-28s %12s %12s",
    f"train vs shifted ({SHIFT_FEATURE})",
    f"drift={shifted_evidently_drifted > 0}",
    f"drift={psi_shifted_result['drift_detected']}",
)
logger.info(
    "Custom-PSI max scores — baseline=%.6f, shifted=%.6f (threshold=%.2f)",
    psi_baseline_result["max_psi"], psi_shifted_result["max_psi"], DRIFT_PSI_THRESHOLD,
)
logger.info(
    "Custom-PSI top-drifted feature in shifted scenario: %s",
    next(iter(psi_shifted_result["feature_scores"])),
)

agreement = (
    (baseline_evidently_drifted > 0) == psi_baseline_result["drift_detected"]
    and (shifted_evidently_drifted > 0) == psi_shifted_result["drift_detected"]
)
logger.info(
    "Two independent methods %s on both scenarios — %s",
    "AGREE" if agreement else "DISAGREE",
    "supports trusting the lightweight PSI module in the live API hot path"
    if agreement else "investigate before relying on either in production",
)

# ---------------------------------------------------------------------------
# Cell 5 — MLflow logging (experiment "csip-monitoring")
# ---------------------------------------------------------------------------
_mlflow_uri = MLFLOW_TRACKING_URI
try:
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-monitoring")
    logger.info("MLflow connected: %s", _mlflow_uri)
except Exception:
    _mlflow_uri = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-monitoring")
    logger.warning("MLflow server unavailable -- using local file store: %s", _mlflow_uri)

with mlflow.start_run(run_name="drift_baseline_vs_shifted"):
    mlflow.log_params({
        "shift_feature":              SHIFT_FEATURE,
        "shift_magnitude_std_units":  str(SHIFT_MAGNITUDE_STD_UNITS),
        "shift_amount":               f"{shift_amount:.6f}",
        "drift_psi_threshold":        str(DRIFT_PSI_THRESHOLD),
        "fast_mode":                  str(FAST_MODE),
    })
    mlflow.log_metric("evidently_baseline_n_drifted", baseline_evidently_drifted)
    mlflow.log_metric("evidently_baseline_share_drifted", baseline_evidently_share)
    mlflow.log_metric("evidently_shifted_n_drifted", shifted_evidently_drifted)
    mlflow.log_metric("evidently_shifted_share_drifted", shifted_evidently_share)
    mlflow.log_metric("custom_psi_baseline_max", psi_baseline_result["max_psi"])
    mlflow.log_metric("custom_psi_baseline_n_drifted", psi_baseline_result["n_drifted"])
    mlflow.log_metric("custom_psi_shifted_max", psi_shifted_result["max_psi"])
    mlflow.log_metric("custom_psi_shifted_n_drifted", psi_shifted_result["n_drifted"])
    mlflow.log_artifact(str(baseline_report_path))
    mlflow.log_artifact(str(shifted_report_path))
    logger.info("MLflow run 'drift_baseline_vs_shifted' logged")

# ---------------------------------------------------------------------------
# Cell 6 — Save section_12_metrics.json
# ---------------------------------------------------------------------------
section_12_metrics = {
    "section": "12",
    "fast_mode": FAST_MODE,
    "generated_at": datetime.datetime.now().isoformat(),
    "drift_psi_threshold": DRIFT_PSI_THRESHOLD,
    "synthetic_shift": {
        "feature": SHIFT_FEATURE,
        "magnitude_std_units": SHIFT_MAGNITUDE_STD_UNITS,
        "training_std": shift_std,
        "absolute_shift": shift_amount,
    },
    "scenarios": {
        "train_vs_test": {
            "evidently_n_drifted_columns": baseline_evidently_drifted,
            "evidently_share_drifted": round(baseline_evidently_share, 6),
            "custom_psi": psi_baseline_result,
        },
        "train_vs_shifted_test": {
            "evidently_n_drifted_columns": shifted_evidently_drifted,
            "evidently_share_drifted": round(shifted_evidently_share, 6),
            "custom_psi": psi_shifted_result,
        },
    },
    "methods_agree": agreement,
    "reports": {
        "baseline_html": str(baseline_report_path.relative_to(PROJECT_ROOT)),
        "shifted_html": str(shifted_report_path.relative_to(PROJECT_ROOT)),
    },
}

with open(DRIFT_SUMMARY_PATH, "w") as fh:
    json.dump(section_12_metrics, fh, indent=2, cls=NumpyEncoder)
logger.info("Saved metrics → %s", DRIFT_SUMMARY_PATH)
logger.info("Section 12 monitoring notebook complete.")
