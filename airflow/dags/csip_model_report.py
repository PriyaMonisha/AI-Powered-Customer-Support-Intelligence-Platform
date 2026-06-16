# filename: airflow/dags/csip_model_report.py
# purpose:  DAG 4 — Weekly MLflow model report: pull experiment runs →
#           build leaderboard → update model registry → save dated JSON report.
#           Scheduled 4h after csip_retrain (06:00 vs 02:00 UTC Sunday) to avoid
#           TOCTOU race on model_registry.json after promotion.
# version:  1.0

import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.exceptions import AirflowSkipException  # noqa: F401 — module-level: cross_project_ml.md rule
from airflow.models import DAG
from airflow.operators.python import PythonOperator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "csip-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# MLflow experiment names — must match names used in Sections 5–10
# (verified via mlflow.search_experiments() on local mlruns at Section 13 build time)
_CLASSIFIER_EXPERIMENTS = [
    "csip-baseline-classifiers",
    "csip-advanced-classifiers",
]
_REGRESSOR_EXPERIMENTS = [
    "csip-regression-models",
]


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _load_mlflow_runs(**context) -> None:
    """
    Pull runs from all CSIP MLflow experiments into XCom.
    Uses mlflow.get_experiment_by_name() — NOT hardcoded IDs (IDs differ between environments).
    Skips experiments with logger.warning if not found (experiment may not exist in all envs).
    All numeric values cast to float() to prevent np.float64 JSON serialization errors.
    """
    import mlflow
    from config import MLFLOW_TRACKING_URI

    # Override with Docker service hostname if set; fall back to config value
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    classifier_runs: list[dict] = []
    regressor_runs: list[dict] = []

    def _pull_experiment(exp_name: str) -> list[dict]:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning("Experiment %r not found in MLflow — skipping", exp_name)
            return []
        runs_df = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
        )
        if runs_df.empty:
            logger.info("Experiment %r has no runs", exp_name)
            return []
        records = []
        for _, row in runs_df.iterrows():
            record: dict = {
                "experiment_name": exp_name,
                "run_id":    str(row.get("run_id", "")),
                "algo":      str(row.get("params.model_name", row.get("params.algo", "unknown"))),
                "task":      str(row.get("params.task",       "unknown")),
                "status":    str(row.get("status",            "UNKNOWN")),
                "start_time": str(row.get("start_time",       "")),
            }
            # Classifier metrics
            for metric in ("val_f1_macro", "test_f1_macro", "val_f1_weighted", "accuracy"):
                raw = row.get(f"metrics.{metric}")
                record[metric] = float(raw) if raw is not None and str(raw) != "nan" else None
            # Regressor metrics
            for metric in ("val_rmse", "val_mae", "val_r2", "val_mape"):
                raw = row.get(f"metrics.{metric}")
                record[metric] = float(raw) if raw is not None and str(raw) != "nan" else None
            records.append(record)
        logger.info("Pulled %d runs from experiment %r", len(records), exp_name)
        return records

    for exp_name in _CLASSIFIER_EXPERIMENTS:
        classifier_runs.extend(_pull_experiment(exp_name))
    for exp_name in _REGRESSOR_EXPERIMENTS:
        regressor_runs.extend(_pull_experiment(exp_name))

    context["ti"].xcom_push(key="classifier_runs", value=classifier_runs)
    context["ti"].xcom_push(key="regressor_runs",  value=regressor_runs)
    logger.info(
        "MLflow pull complete: %d classifier runs, %d regressor runs",
        len(classifier_runs), len(regressor_runs),
    )


def _compute_leaderboard(**context) -> None:
    """
    Build two sorted leaderboard tables:
      classifiers — sorted descending by val_f1_macro
      regressors  — sorted ascending by val_rmse
    Mixed-direction sort avoided: each table has a single, consistent sort direction.
    """
    ti = context["ti"]
    classifier_runs: list[dict] = ti.xcom_pull(task_ids="load_mlflow_runs", key="classifier_runs") or []
    regressor_runs:  list[dict] = ti.xcom_pull(task_ids="load_mlflow_runs", key="regressor_runs")  or []

    # Classifiers: descending F1 — best model first
    clf_leaderboard = sorted(
        [r for r in classifier_runs if r.get("val_f1_macro") is not None],
        key=lambda r: r["val_f1_macro"],
        reverse=True,
    )

    # Regressors: ascending RMSE — best model first
    reg_leaderboard = sorted(
        [r for r in regressor_runs if r.get("val_rmse") is not None],
        key=lambda r: r["val_rmse"],
        reverse=False,
    )

    if clf_leaderboard:
        best = clf_leaderboard[0]
        logger.info(
            "Top classifier: algo=%s task=%s val_f1_macro=%.4f",
            best["algo"], best["task"], best["val_f1_macro"],
        )
    else:
        logger.warning("No classifier runs with val_f1_macro found")

    if reg_leaderboard:
        best = reg_leaderboard[0]
        logger.info(
            "Top regressor: algo=%s task=%s val_rmse=%.4f",
            best["algo"], best["task"], best["val_rmse"],
        )
    else:
        logger.warning("No regressor runs with val_rmse found")

    leaderboard = {
        "classifiers": clf_leaderboard,
        "regressors":  reg_leaderboard,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    ti.xcom_push(key="leaderboard", value=leaderboard)
    logger.info(
        "Leaderboard built: %d clf entries, %d reg entries",
        len(clf_leaderboard), len(reg_leaderboard),
    )


def _update_model_registry(**context) -> None:
    """
    Update model_registry.json with best run metrics from this week's leaderboard.
    Only updates if leaderboard has scored entries — skips silently otherwise.
    Atomic write: .json.tmp → shutil.move (C-5 / atomic write pattern).
    """
    import json
    from config import MODEL_REGISTRY_PATH

    ti = context["ti"]
    leaderboard: dict = ti.xcom_pull(task_ids="compute_leaderboard", key="leaderboard")
    if leaderboard is None:
        raise ValueError("compute_leaderboard XCom is None — check task logs")

    if not MODEL_REGISTRY_PATH.exists():
        logger.warning("model_registry.json not found — skipping registry update")
        return

    with open(str(MODEL_REGISTRY_PATH)) as fh:
        registry: dict = json.load(fh)

    clf_rows = leaderboard.get("classifiers", [])

    # Update best classifier metrics if better than current
    for row in clf_rows:
        task = row.get("task", "")
        val_f1 = row.get("val_f1_macro")
        if val_f1 is None:
            continue
        if "type" in task and "ticket_type" in registry:
            current = float(registry["ticket_type"].get("val_f1_macro", 0.0))
            if val_f1 > current:
                registry["ticket_type"]["val_f1_macro"] = round(val_f1, 6)
                registry["ticket_type"]["last_updated"] = context["ds"]
                logger.info("Registry updated: ticket_type val_f1_macro=%.4f", val_f1)
            break
        if "priority" in task and "ticket_priority" in registry:
            current = float(registry["ticket_priority"].get("val_f1_macro", 0.0))
            if val_f1 > current:
                registry["ticket_priority"]["val_f1_macro"] = round(val_f1, 6)
                registry["ticket_priority"]["last_updated"] = context["ds"]
                logger.info("Registry updated: ticket_priority val_f1_macro=%.4f", val_f1)
            break

    registry["last_report_run"] = context["ds"]

    # Atomic write
    tmp_path = MODEL_REGISTRY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(registry, indent=2))
    shutil.move(str(tmp_path), str(MODEL_REGISTRY_PATH))
    logger.info("model_registry.json updated atomically")


def _save_report(**context) -> None:
    """Save the full dated report JSON to artifacts/reports/."""
    import json
    from config import REPORTS_DIR

    ti = context["ti"]
    leaderboard: dict = ti.xcom_pull(task_ids="compute_leaderboard", key="leaderboard")
    if leaderboard is None:
        raise ValueError("compute_leaderboard XCom is None — check task logs")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ds = context["ds"]
    report = {
        "report_date": ds,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard": leaderboard,
        "task_counts": {
            "csip_etl":           5,
            "csip_drift_monitor": 6,
            "csip_retrain":       10,
            "csip_model_report":  5,
        },
    }
    report_path = REPORTS_DIR / f"model_report_{ds}.json"
    with open(str(report_path), "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Model report saved: %s", report_path)
    ti.xcom_push(key="report_path", value=str(report_path))


def _log_summary(**context) -> None:
    """Log a concise weekly model report summary."""
    ti = context["ti"]
    leaderboard: dict = ti.xcom_pull(task_ids="compute_leaderboard", key="leaderboard") or {}
    report_path: str  = ti.xcom_pull(task_ids="save_report",         key="report_path") or "unknown"

    clf_rows = leaderboard.get("classifiers", [])
    reg_rows = leaderboard.get("regressors", [])

    clf_summary = (
        f"algo={clf_rows[0]['algo']} val_f1_macro={clf_rows[0]['val_f1_macro']:.4f}"
        if clf_rows else "no data"
    )
    reg_summary = (
        f"algo={reg_rows[0]['algo']} val_rmse={reg_rows[0]['val_rmse']:.4f}"
        if reg_rows else "no data"
    )

    logger.info(
        "=== Weekly Model Report %s ===\n"
        "  Best classifier: %s\n"
        "  Best regressor:  %s\n"
        "  Report path:     %s\n"
        "  Total clf runs:  %d  |  Total reg runs: %d",
        context["ds"],
        clf_summary,
        reg_summary,
        report_path,
        len(clf_rows),
        len(reg_rows),
    )

    # XCom push for downstream inspection (all float to prevent serialization errors)
    summary = {
        "total_classifier_runs": int(len(clf_rows)),
        "total_regressor_runs":  int(len(reg_rows)),
        "best_clf_f1":  float(clf_rows[0]["val_f1_macro"]) if clf_rows else None,
        "best_reg_rmse": float(reg_rows[0]["val_rmse"])   if reg_rows else None,
    }
    ti.xcom_push(key="summary", value=summary)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="csip_model_report",
    default_args=_DEFAULT_ARGS,
    description="Weekly MLflow model report: pull runs → leaderboard → registry → save",
    schedule_interval="0 6 * * 0",    # 06:00 UTC every Sunday (4h after csip_retrain)
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["csip", "reporting", "mlflow"],
) as dag:

    load_mlflow_runs = PythonOperator(
        task_id="load_mlflow_runs",
        python_callable=_load_mlflow_runs,
    )

    compute_leaderboard = PythonOperator(
        task_id="compute_leaderboard",
        python_callable=_compute_leaderboard,
    )

    update_model_registry = PythonOperator(
        task_id="update_model_registry",
        python_callable=_update_model_registry,
    )

    save_report = PythonOperator(
        task_id="save_report",
        python_callable=_save_report,
    )

    log_summary = PythonOperator(
        task_id="log_summary",
        python_callable=_log_summary,
    )

    # Task dependency chain
    load_mlflow_runs >> compute_leaderboard >> update_model_registry >> save_report >> log_summary
