# filename: airflow/dags/csip_drift_monitor.py
# purpose:  DAG 2 — Daily drift monitoring: load recent tabular features →
#           compute PSI vs training baseline → 2-branch: no_drift_log | alert_drift.
# version:  1.0

import logging
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.exceptions import AirflowSkipException  # module-level: cross_project_ml.md rule
from airflow.models import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

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

# ---------------------------------------------------------------------------
# Notification stub — shared by alert_drift
# ---------------------------------------------------------------------------

def _send_notification_stub(subject: str, body: str) -> None:
    """
    Notification hook. Reads CSIP_ALERT_WEBHOOK env var.
    If set: POSTs JSON payload to Slack-compatible webhook URL.
    If not set: logs warning only (no external call).
    """
    webhook = os.getenv("CSIP_ALERT_WEBHOOK", "")
    if webhook:
        import json as _json
        payload = _json.dumps({"text": f"*{subject}*\n{body}"}).encode()
        req = urllib.request.Request(
            webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        logger.info("Alert sent to webhook: %s", subject)
    else:
        logger.warning("No CSIP_ALERT_WEBHOOK configured — alert logged only: %s", subject)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _load_recent_data(**context) -> None:
    """
    Load current production proxy data for drift detection.

    FILE CHOICE: X_test_tabular.npy (post-encoding) is used as the "recent data" proxy.
    This matches training_baseline.json which stores post-encoding statistics
    (Ticket Channel_enc, Customer Gender_enc, etc.) — verified Section 12.

    Production note: replace with a rolling window of live inference feature logs.
    ExternalTaskSensor on csip_etl.build_feature_arrays would guarantee freshness.
    """
    import json

    import numpy as np
    from config import FEATURES_DIR

    X_test = np.load(str(FEATURES_DIR / "X_test_tabular.npy"))
    with open(str(FEATURES_DIR / "tabular_columns.json")) as fh:
        columns: list = json.load(fh)

    assert X_test.shape[1] == len(columns), (
        f"Shape mismatch: X_test has {X_test.shape[1]} cols, "
        f"tabular_columns.json has {len(columns)}"
    )

    context["ti"].xcom_push(key="n_samples", value=int(X_test.shape[0]))
    context["ti"].xcom_push(key="n_features", value=int(X_test.shape[1]))
    logger.info("Recent data loaded: shape=%s", X_test.shape)


def _compute_psi(**context) -> None:
    """
    Compute PSI between recent data and training baseline.
    All values pushed to XCom as float() to prevent np.float64 JSON serialization errors.
    """
    import json

    import numpy as np
    import pandas as pd
    from config import FEATURES_DIR
    from src.monitoring.drift import check_drift, load_baseline

    X_test = np.load(str(FEATURES_DIR / "X_test_tabular.npy"))
    with open(str(FEATURES_DIR / "tabular_columns.json")) as fh:
        columns: list = json.load(fh)

    current_df = pd.DataFrame(X_test, columns=columns)
    baseline = load_baseline()
    result = check_drift(current_df, baseline=baseline)

    # All values cast to float() / int — no np.float64 in XCom JSON
    xcom_result = {
        "feature_scores": {k: float(v) for k, v in result["feature_scores"].items()},
        "max_psi": float(result["max_psi"]),
        "n_drifted": int(result["n_drifted"]),
        "drift_detected": bool(result["drift_detected"]),
        "checked_at": result["checked_at"],
    }
    context["ti"].xcom_push(key="drift_result", value=xcom_result)
    logger.info(
        "PSI check complete: max_psi=%.4f n_drifted=%d drift_detected=%s",
        xcom_result["max_psi"],
        xcom_result["n_drifted"],
        xcom_result["drift_detected"],
    )


def _save_drift_report(**context) -> None:
    """Save the drift result to a dated JSON report file."""
    import json
    from config import DRIFT_DIR

    ds = context["ds"]
    result = context["ti"].xcom_pull(task_ids="compute_psi", key="drift_result")
    if result is None:
        raise ValueError("compute_psi XCom is None — check task logs")

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DRIFT_DIR / f"dag_drift_report_{ds}.json"
    with open(str(report_path), "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Drift report saved: %s", report_path)


def _check_drift_branch(**context) -> str:
    """2-branch: return task_id based on drift_detected flag."""
    result = context["ti"].xcom_pull(task_ids="compute_psi", key="drift_result")
    if result is None:
        raise ValueError("compute_psi XCom is None — check task logs")

    if result["drift_detected"]:
        return "alert_drift"
    return "no_drift_log"


def _no_drift_log(**context) -> None:
    """Log that no drift was detected — no action required."""
    result = context["ti"].xcom_pull(task_ids="compute_psi", key="drift_result")
    logger.info(
        "No drift detected | max_psi=%.4f (threshold=0.10) | checked_at=%s",
        result["max_psi"] if result else 0.0,
        result.get("checked_at", "unknown") if result else "unknown",
    )


def _alert_drift(**context) -> None:
    """Log WARNING with top-3 drifted features by PSI. Send webhook notification if configured."""
    result = context["ti"].xcom_pull(task_ids="compute_psi", key="drift_result")
    if result is None:
        raise ValueError("compute_psi XCom is None — check task logs")

    # Top-3 features sorted by PSI descending (already sorted by check_drift, but re-sort to be safe)
    top3 = sorted(result["feature_scores"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    top3_str = ", ".join(f"{f}={s:.4f}" for f, s in top3)

    logger.warning(
        "DRIFT DETECTED | max_psi=%.4f | n_drifted=%d | top features: %s",
        result["max_psi"],
        result["n_drifted"],
        top3_str,
    )

    _send_notification_stub(
        subject="[CSIP] Feature Drift Detected",
        body=(
            f"max_psi={result['max_psi']:.4f} | n_drifted={result['n_drifted']}\n"
            f"Top features: {top3_str}\n"
            f"Checked at: {result['checked_at']}"
        ),
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="csip_drift_monitor",
    default_args=_DEFAULT_ARGS,
    description="Daily PSI drift check vs training baseline; alerts on drift",
    schedule_interval="0 3 * * *",    # 03:00 UTC daily (1h after csip_etl)
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["csip", "monitoring", "drift"],
    # Production improvement: add ExternalTaskSensor waiting for
    # csip_etl.build_feature_arrays to ensure fresh .npy files.
) as dag:

    load_recent_data = PythonOperator(
        task_id="load_recent_data",
        python_callable=_load_recent_data,
    )

    compute_psi = PythonOperator(
        task_id="compute_psi",
        python_callable=_compute_psi,
    )

    save_drift_report = PythonOperator(
        task_id="save_drift_report",
        python_callable=_save_drift_report,
    )

    check_drift_branch = BranchPythonOperator(
        task_id="check_drift_branch",
        python_callable=_check_drift_branch,
    )

    no_drift_log = PythonOperator(
        task_id="no_drift_log",
        python_callable=_no_drift_log,
    )

    alert_drift = PythonOperator(
        task_id="alert_drift",
        python_callable=_alert_drift,
    )

    # Task dependency chain
    load_recent_data >> compute_psi >> save_drift_report >> check_drift_branch
    check_drift_branch >> [no_drift_log, alert_drift]
