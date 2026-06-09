# filename: notebooks/13_airflow.py
# purpose:  Section 13 — Validate Airflow DAG task logic without the Airflow scheduler.
#           Runs in terminal (no GPU required). Tests: ETL callables, PSI drift check,
#           3-branch promotion guard simulation, drift baseline regeneration, MLflow pull.
# version:  1.0

FAST_MODE = True  # gates Optuna trial count in branch guard simulation

# ── Cell 1: Imports + project root ──────────────────────────────────────────
import json
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

try:
    _NB_DIR = Path(__file__).resolve().parent
except NameError:
    _NB_DIR = Path.cwd()

PROJECT_ROOT = _NB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DRIFT_BASELINE_PATH,
    DRIFT_DIR,
    FEATURES_DIR,
    FAST_N_TRIALS,
    FULL_N_TRIALS,
    MODEL_REGISTRY_PATH,
    PROCESSED_DATA_DIR,
    PROMOTE_THRESHOLD,
    RAW_DATA_DIR,
    RANDOM_STATE,
    REGRESSION_THRESHOLD,
    REPORTS_DIR,
    MLFLOW_TRACKING_URI,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR   = ARTIFACTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Section 13: Airflow DAGs — validation notebook")
logger.info("FAST_MODE=%s | PROJECT_ROOT=%s", FAST_MODE, PROJECT_ROOT)

# ── Cell 2: DAG topology display ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("DAG Topology Summary")
print("=" * 62)

dag_topologies = {
    "csip_etl": {
        "schedule": "0 2 * * *  (02:00 UTC daily)",
        "tasks": [
            "check_raw_data",
            "clean_data",
            "validate_schema",
            "etl_to_postgres",
            "build_feature_arrays",
        ],
        "dependency": "linear chain",
    },
    "csip_drift_monitor": {
        "schedule": "0 3 * * *  (03:00 UTC daily - 1h after ETL)",
        "tasks": [
            "load_recent_data",
            "compute_psi",
            "save_drift_report",
            "check_drift_branch",
            "no_drift_log",
            "alert_drift",
        ],
        "dependency": "linear -> 2-branch",
    },
    "csip_retrain": {
        "schedule": "0 2 * * 0  (02:00 UTC Sundays)",
        "tasks": [
            "load_training_data",
            "retrain_type_clf",
            "retrain_priority_clf",
            "retrain_regressor",
            "evaluate_new_models",
            "promotion_branch",
            "promote_models",
            "regenerate_drift_baseline",
            "skip_retrain",
            "alert_regression",
        ],
        "dependency": "fan-out -> fan-in -> 3-branch",
    },
    "csip_model_report": {
        "schedule": "0 6 * * 0  (06:00 UTC Sundays - 4h after retrain)",
        "tasks": [
            "load_mlflow_runs",
            "compute_leaderboard",
            "update_model_registry",
            "save_report",
            "log_summary",
        ],
        "dependency": "linear chain",
    },
}

task_counts = {}
for dag_id, info in dag_topologies.items():
    n = len(info["tasks"])
    task_counts[dag_id] = n
    print(f"\n{dag_id}  [{info['schedule']}]  {n} tasks  ({info['dependency']})")
    for t in info["tasks"]:
        print(f"    - {t}")

print(f"\nTotal DAGs: {len(dag_topologies)} | Total tasks: {sum(task_counts.values())}")
print("=" * 62 + "\n")

# ── Cell 3: Airflow soft-require ──────────────────────────────────────────────
try:
    import airflow
    AIRFLOW_AVAILABLE = True
    _af_ver = getattr(airflow, "__version__", "unknown")
    logger.info("Airflow %s available", _af_ver)
except ImportError:
    AIRFLOW_AVAILABLE = False
    logger.warning(
        "Airflow not installed — callable logic validation continues without scheduler context."
    )

# ── Cell 4: Validate DAG 1 logic (ETL callables) ─────────────────────────────
print("\n--- Cell 4: DAG 1 — ETL callable validation ---")

import pandas as pd
from src.data.clean import clean_pipeline
from src.data.validate import validate

raw_path = RAW_DATA_DIR / "customer_support_tickets.csv"
assert raw_path.exists(), f"Raw data not found: {raw_path}"

with open(raw_path, encoding="utf-8", errors="replace") as fh:
    n_rows_raw = sum(1 for _ in fh) - 1  # subtract header

df_raw = pd.read_csv(raw_path)
df_clean = clean_pipeline(df_raw)

df_clean_for_val = pd.read_csv(
    PROCESSED_DATA_DIR / "cleaned_tickets.csv",
    parse_dates=["Date of Purchase", "First Response Time", "Time to Resolution"],
)
validate(df_clean_for_val)  # raises on schema errors

logger.info("DAG 1 ETL validation PASSED")
logger.info("  raw rows=%d | cleaned rows=%d cols=%d", n_rows_raw, len(df_clean), df_clean.shape[1])

# ── Cell 5: Validate DAG 2 logic (PSI drift check) ──────────────────────────
print("\n--- Cell 5: DAG 2 — Drift monitor callable validation ---")

import numpy as np
from src.monitoring.drift import check_drift, load_baseline

X_test = np.load(str(FEATURES_DIR / "X_test_tabular.npy"))
with open(str(FEATURES_DIR / "tabular_columns.json")) as fh:
    tabular_columns: list = json.load(fh)

current_df = pd.DataFrame(X_test, columns=tabular_columns)
baseline   = load_baseline()
drift_result = check_drift(current_df, baseline=baseline)

logger.info("DAG 2 PSI check PASSED")
logger.info(
    "  max_psi=%.4f | n_drifted=%d | drift_detected=%s | n_features=%d",
    drift_result["max_psi"],
    drift_result["n_drifted"],
    drift_result["drift_detected"],
    len(drift_result["feature_scores"]),
)
# Show top 3 by PSI
top3 = list(drift_result["feature_scores"].items())[:3]
for feat, psi in top3:
    logger.info("  %s: PSI=%.4f", feat, psi)

# ── Cell 6: Validate DAG 3 setup (champion metrics) ─────────────────────────
print("\n--- Cell 6: DAG 3 — Champion metrics from model_registry.json ---")

assert MODEL_REGISTRY_PATH.exists(), f"Model registry not found: {MODEL_REGISTRY_PATH}"
with open(str(MODEL_REGISTRY_PATH)) as fh:
    registry: dict = json.load(fh)

champion_type_f1     = float(registry["ticket_type"]["val_f1_macro"])
champion_priority_f1 = float(registry["ticket_priority"]["val_f1_macro"])
mean_champion_f1     = float((champion_type_f1 + champion_priority_f1) / 2)

logger.info("Champion metrics loaded:")
logger.info("  ticket_type     val_f1_macro=%.4f", champion_type_f1)
logger.info("  ticket_priority val_f1_macro=%.4f", champion_priority_f1)
logger.info("  mean_champion_f1=%.4f", mean_champion_f1)

# ── Cell 7: Simulate 3-branch guard ─────────────────────────────────────────
print("\n--- Cell 7: DAG 3 — 3-branch promotion guard simulation ---")

class _MockTI:
    """Minimal Airflow TaskInstance mock for xcom_pull simulation."""

    def __init__(self, eval_metrics: dict):
        self._metrics = eval_metrics

    def xcom_pull(self, task_ids: str, key: str) -> dict | None:
        if task_ids == "evaluate_new_models" and key == "eval_metrics":
            return self._metrics
        return None


def _simulate_decide_branch(eval_metrics: dict) -> str:
    """Mirrors csip_retrain._decide_branch logic for standalone testing."""
    if eval_metrics is None:
        raise ValueError("evaluate_new_models XCom is None — task may have failed.")
    delta = eval_metrics["mean_new_f1"] - eval_metrics["mean_champion_f1"]
    if delta >= PROMOTE_THRESHOLD:
        return "promote_models"
    elif delta >= -REGRESSION_THRESHOLD:
        return "skip_retrain"
    else:
        return "alert_regression"


# Test case 1: F1 improvement >= PROMOTE_THRESHOLD (0.02) → promote
promote_metrics = {
    "mean_new_f1":      mean_champion_f1 + 0.05,
    "mean_champion_f1": mean_champion_f1,
    "type_f1":          champion_type_f1 + 0.05,
    "priority_f1":      champion_priority_f1 + 0.05,
    "regressor_rmse":   7.09,
}
case1 = _simulate_decide_branch(promote_metrics)
assert case1 == "promote_models", f"Expected promote_models, got {case1}"
logger.info("Case 1 (delta=+0.05 >= %.2f): → %s ✓", PROMOTE_THRESHOLD, case1)

# Test case 2: Small improvement within bounds → skip
skip_metrics = {
    "mean_new_f1":      mean_champion_f1 + 0.01,
    "mean_champion_f1": mean_champion_f1,
    "type_f1":          champion_type_f1 + 0.01,
    "priority_f1":      champion_priority_f1 + 0.01,
    "regressor_rmse":   7.10,
}
case2 = _simulate_decide_branch(skip_metrics)
assert case2 == "skip_retrain", f"Expected skip_retrain, got {case2}"
logger.info("Case 2 (delta=+0.01 in bounds): → %s ✓", case2)

# Test case 3: Regression >= REGRESSION_THRESHOLD (0.05) → alert
alert_metrics = {
    "mean_new_f1":      mean_champion_f1 - 0.08,
    "mean_champion_f1": mean_champion_f1,
    "type_f1":          champion_type_f1 - 0.08,
    "priority_f1":      champion_priority_f1 - 0.08,
    "regressor_rmse":   9.00,
}
case3 = _simulate_decide_branch(alert_metrics)
assert case3 == "alert_regression", f"Expected alert_regression, got {case3}"
logger.info("Case 3 (delta=-0.08 <= -%.2f): → %s ✓", REGRESSION_THRESHOLD, case3)

# Edge case: exact boundary — promote threshold
boundary_metrics = {
    "mean_new_f1":      mean_champion_f1 + PROMOTE_THRESHOLD,
    "mean_champion_f1": mean_champion_f1,
    "type_f1":          champion_type_f1,
    "priority_f1":      champion_priority_f1,
    "regressor_rmse":   7.09,
}
boundary_result = _simulate_decide_branch(boundary_metrics)
assert boundary_result == "promote_models", f"Boundary test failed: {boundary_result}"
logger.info("Edge case (delta=exactly +%.2f): → %s ✓", PROMOTE_THRESHOLD, boundary_result)

branch_simulation_results = {
    "promote_case_result": case1,
    "skip_case_result":    case2,
    "alert_case_result":   case3,
    "boundary_test":       boundary_result,
}
logger.info("3-branch guard simulation: ALL 4 ASSERTIONS PASSED")

# ── Cell 8: Regenerate drift baseline ────────────────────────────────────────
print("\n--- Cell 8: DAG 3 — drift baseline regeneration (direct callable) ---")

X_train = np.load(str(FEATURES_DIR / "X_train_tabular.npy"))

# Load before-state for comparison
baseline_before = load_baseline()
n_features_before = len(baseline_before.get("feature_names", []))

# Regenerate baseline (same logic as _regenerate_drift_baseline in csip_retrain.py)
n_train = len(X_train)
stats: dict = {}

for i, col in enumerate(tabular_columns):
    vals = X_train[:, i]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        continue
    feat_stats = {
        "mean": round(float(np.mean(vals)), 6),
        "std":  round(float(np.std(vals)), 6),
        "min":  round(float(np.min(vals)), 6),
        "max":  round(float(np.max(vals)), 6),
        "p25":  round(float(np.percentile(vals, 25)), 6),
        "p50":  round(float(np.percentile(vals, 50)), 6),
        "p75":  round(float(np.percentile(vals, 75)), 6),
    }
    unique_vals = np.unique(vals)
    if len(unique_vals) <= 25:
        freq = {
            str(float(v)): round(float(np.sum(vals == v) / len(vals)), 6)
            for v in unique_vals
        }
        feat_stats["value_frequencies"] = freq
    stats[col] = feat_stats

new_baseline = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "n_train":      n_train,
    "feature_names": tabular_columns,
    "stats":        stats,
}

DRIFT_DIR.mkdir(parents=True, exist_ok=True)
tmp_path = DRIFT_BASELINE_PATH.with_suffix(".json.tmp")
tmp_path.write_text(json.dumps(new_baseline, indent=2))
shutil.move(str(tmp_path), str(DRIFT_BASELINE_PATH))

# Verify after-state
baseline_after = load_baseline()
n_features_after = len(baseline_after.get("feature_names", []))

logger.info("Drift baseline regenerated:")
logger.info("  features before=%d after=%d | n_train=%d", n_features_before, n_features_after, n_train)
assert n_features_before == n_features_after, (
    f"Feature count changed: {n_features_before} → {n_features_after}"
)
logger.info("  Baseline regeneration: PASSED ✓")

# ── Cell 9: Validate DAG 4 logic (MLflow pull + leaderboard) ─────────────────
print("\n--- Cell 9: DAG 4 — MLflow run pull + leaderboard ---")

import mlflow

# Use local file-based mlruns for terminal validation (HTTP server is Docker-only)
_mlruns_dir = PROJECT_ROOT / "mlruns"
_local_tracking_uri = _mlruns_dir.as_uri()  # file:///...
mlflow.set_tracking_uri(_local_tracking_uri)
logger.info("MLflow tracking URI (local file): %s", _local_tracking_uri)

CLASSIFIER_EXPERIMENTS = ["csip-baseline-classifiers", "csip-advanced-classifiers"]
REGRESSOR_EXPERIMENTS  = ["csip-regression-models"]

classifier_runs: list[dict] = []
regressor_runs:  list[dict] = []
mlflow_runs_found = 0

for exp_name in CLASSIFIER_EXPERIMENTS + REGRESSOR_EXPERIMENTS:
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        logger.warning("Experiment %r not found — skipping", exp_name)
        continue
    runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs_df.empty:
        logger.info("Experiment %r: 0 runs", exp_name)
        continue
    mlflow_runs_found += len(runs_df)
    logger.info("Experiment %r: %d runs", exp_name, len(runs_df))
    for _, row in runs_df.iterrows():
        record = {
            "experiment_name": exp_name,
            "run_id":  str(row.get("run_id", "")),
            "algo":    str(row.get("params.model_name", row.get("params.algo", "unknown"))),
            "task":    str(row.get("params.task", "unknown")),
        }
        val_f1 = row.get("metrics.val_f1_macro")
        val_rmse = row.get("metrics.val_rmse")
        if val_f1 is not None and str(val_f1) != "nan":
            record["val_f1_macro"] = float(val_f1)
        if val_rmse is not None and str(val_rmse) != "nan":
            record["val_rmse"] = float(val_rmse)

        if exp_name in CLASSIFIER_EXPERIMENTS:
            classifier_runs.append(record)
        else:
            regressor_runs.append(record)

# Build leaderboard
clf_leaderboard = sorted(
    [r for r in classifier_runs if r.get("val_f1_macro") is not None],
    key=lambda r: r["val_f1_macro"],
    reverse=True,
)
reg_leaderboard = sorted(
    [r for r in regressor_runs if r.get("val_rmse") is not None],
    key=lambda r: r["val_rmse"],
    reverse=False,
)

logger.info("DAG 4 leaderboard built:")
logger.info("  total MLflow runs found=%d", mlflow_runs_found)
logger.info("  clf_leaderboard entries=%d | reg_leaderboard entries=%d",
            len(clf_leaderboard), len(reg_leaderboard))
if clf_leaderboard:
    best = clf_leaderboard[0]
    logger.info("  Best classifier: algo=%s val_f1_macro=%.4f",
                best["algo"], best["val_f1_macro"])
if reg_leaderboard:
    best = reg_leaderboard[0]
    logger.info("  Best regressor:  algo=%s val_rmse=%.4f",
                best["algo"], best["val_rmse"])

# ── Cell 10: Log to MLflow ────────────────────────────────────────────────────
print("\n--- Cell 10: MLflow validation run ---")

# Tracking URI already set to local file store in Cell 9
mlflow.set_experiment("csip-airflow-validation")
with mlflow.start_run(run_name="section_13_dag_validation"):
    mlflow.log_params({
        "fast_mode":   str(FAST_MODE),
        "dag_count":   "4",
        "total_tasks": str(sum(task_counts.values())),
    })
    mlflow.log_metrics({
        "dag1_task_count":         float(task_counts["csip_etl"]),
        "dag2_task_count":         float(task_counts["csip_drift_monitor"]),
        "dag3_task_count":         float(task_counts["csip_retrain"]),
        "dag4_task_count":         float(task_counts["csip_model_report"]),
        "drift_max_psi":           float(drift_result["max_psi"]),
        "drift_n_drifted":         float(drift_result["n_drifted"]),
        "branch_test_cases_passed": 4.0,
        "baseline_n_features":     float(n_features_after),
        "mlflow_runs_found":       float(mlflow_runs_found),
    })
    run_id = mlflow.active_run().info.run_id

logger.info("MLflow run logged: run_id=%s", run_id)

# ── Cell 11: Save section_13_metrics.json ────────────────────────────────────
print("\n--- Cell 11: Save section_13_metrics.json ---")

section_13_metrics = {
    "fast_mode": FAST_MODE,
    "dag_count": len(dag_topologies),
    "task_counts": task_counts,
    "validation_results": {
        "dag1_clean_pipeline": "pass",
        "dag1_raw_rows":       n_rows_raw,
        "dag1_cleaned_rows":   int(len(df_clean)),
        "dag2_check_drift": {
            "max_psi":        float(drift_result["max_psi"]),
            "n_drifted":      int(drift_result["n_drifted"]),
            "drift_detected": bool(drift_result["drift_detected"]),
        },
        "dag3_champion_metrics": {
            "type_f1":          champion_type_f1,
            "priority_f1":      champion_priority_f1,
            "mean_champion_f1": mean_champion_f1,
        },
        "dag3_branch_simulation": branch_simulation_results,
        "dag3_baseline_regen": {
            "n_features_before": n_features_before,
            "n_features_after":  n_features_after,
            "n_train":           n_train,
            "status":            "pass",
        },
        "dag4_mlflow_runs_found": mlflow_runs_found,
        "dag4_clf_leaderboard_entries": len(clf_leaderboard),
        "dag4_reg_leaderboard_entries": len(reg_leaderboard),
    },
    "mlflow_run_id": run_id,
}

metrics_path = METRICS_DIR / "section_13_metrics.json"
with open(str(metrics_path), "w") as fh:
    json.dump(section_13_metrics, fh, indent=2)

logger.info("section_13_metrics.json saved: %s", metrics_path)

# ── Cell 12: Completion summary ──────────────────────────────────────────────
print("\n" + "=" * 62)
print("Section 13 Complete — Airflow DAGs")
print("=" * 62)
print(f"  DAGs written:         {len(dag_topologies)}")
print(f"  Total tasks:          {sum(task_counts.values())}")
print()
print("  DAG 1 csip_etl:          5 tasks | 02:00 UTC daily")
print("  DAG 2 csip_drift_monitor: 6 tasks | 03:00 UTC daily")
print("  DAG 3 csip_retrain:      10 tasks | 02:00 UTC Sunday (max_active_runs=1)")
print("  DAG 4 csip_model_report:  5 tasks | 06:00 UTC Sunday")
print()
print("  Validations passed:")
print(f"    DAG 1: ETL clean + Pandera schema")
print(f"    DAG 2: PSI drift check (max_psi={drift_result['max_psi']:.4f}, n_drifted={drift_result['n_drifted']})")
print(f"    DAG 3: 3-branch guard (promote/skip/alert — 4/4 cases)")
print(f"    DAG 3: drift baseline regen ({n_features_after} features)")
print(f"    DAG 4: MLflow pull ({mlflow_runs_found} runs found)")
print()
print(f"  Metrics: {metrics_path}")
print("=" * 62 + "\n")
