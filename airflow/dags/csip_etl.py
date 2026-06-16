# filename: airflow/dags/csip_etl.py
# purpose:  DAG 1 — Daily ETL pipeline: raw CSV → clean → Pandera validate →
#           PostgreSQL upsert → rebuild tabular feature arrays.
# version:  1.0

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.exceptions import AirflowSkipException  # noqa: F401 — module-level: cross_project_ml.md rule
from airflow.models import DAG
from airflow.operators.python import PythonOperator

# --- project root on sys.path so config + src imports work inside callables ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared DAG config
# ---------------------------------------------------------------------------
_DEFAULT_ARGS = {
    "owner": "csip-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


# ---------------------------------------------------------------------------
# Task callables  (heavy imports inside — scheduler parses this file every 30s)
# ---------------------------------------------------------------------------

def _check_raw_data(**context) -> None:
    """Assert raw CSV exists and is non-empty. Logs row count."""
    import pandas as pd
    from config import RAW_DATA_DIR

    path = RAW_DATA_DIR / "customer_support_tickets.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path, nrows=5)
    if df.empty:
        raise ValueError(f"Raw data at {path} is empty.")
    # Quick full count without loading all data
    with open(path) as fh:
        n_rows = sum(1 for _ in fh) - 1  # subtract header
    logger.info("Raw data verified: %s | rows=%d", path, n_rows)


def _clean_data(**context) -> None:
    """Run clean_pipeline() on raw CSV; save cleaned_tickets.csv."""
    import pandas as pd
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from src.data.clean import clean_pipeline

    df_raw = pd.read_csv(RAW_DATA_DIR / "customer_support_tickets.csv")
    df_clean = clean_pipeline(df_raw)
    out_path = PROCESSED_DATA_DIR / "cleaned_tickets.csv"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    logger.info("Cleaned data saved: %d rows × %d cols → %s", len(df_clean), df_clean.shape[1], out_path)


def _validate_schema(**context) -> None:
    """Run Pandera validation on cleaned_tickets.csv. Raises on schema errors."""
    import pandas as pd
    from config import PROCESSED_DATA_DIR
    from src.data.validate import validate

    cleaned_path = PROCESSED_DATA_DIR / "cleaned_tickets.csv"
    df = pd.read_csv(
        cleaned_path,
        parse_dates=["Date of Purchase", "First Response Time", "Time to Resolution"],
    )
    validate(df)
    logger.info("Pandera validation PASSED for %s", cleaned_path)


def _etl_to_postgres(**context) -> None:
    """Upsert cleaned tickets into PostgreSQL tickets table."""
    from config import PROCESSED_DATA_DIR
    from src.data.etl import load_csv_to_postgres

    cleaned_path = PROCESSED_DATA_DIR / "cleaned_tickets.csv"
    n_upserted = load_csv_to_postgres(csv_path=cleaned_path)
    logger.info("Upserted %d rows to PostgreSQL", n_upserted)


def _build_feature_arrays(**context) -> None:
    """
    Rebuild tabular feature arrays for all splits from preprocessed_tickets.csv.

    Transform-only: uses the EXISTING fitted TabularEncoder (loaded from
    TABULAR_ENCODER_PATH). Does NOT refit the encoder — refitting happens only
    in csip_retrain DAG. Unknown categories get mean-encoding (production behaviour).
    """
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from config import (
        FEATURES_DIR,
        PREPROCESSED_DATA_PATH,
        SPLIT_INDICES_PATH,
        TABULAR_ENCODER_PATH,
    )
    from src.features.tabular_features import ALL_TABULAR_FEATURES

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load encoder — must be fitted
    encoder = joblib.load(str(TABULAR_ENCODER_PATH))
    assert encoder._fitted, "TabularEncoder is not fitted. Run training pipeline first."

    # Load preprocessed data (Section 3 output — has text meta-features already)
    df = pd.read_csv(str(PREPROCESSED_DATA_PATH))

    # Load split ticket ID lists
    with open(str(SPLIT_INDICES_PATH)) as fh:
        split_ids: dict = json.load(fh)

    for split_name in ("train", "val", "test"):
        ids = set(split_ids[split_name])
        split_df = df[df["Ticket ID"].isin(ids)].copy()

        # Ensure all required tabular columns are present
        missing = [c for c in ALL_TABULAR_FEATURES if c not in split_df.columns]
        if missing:
            raise ValueError(f"Missing columns for {split_name} split: {missing}")

        X_enc = encoder.transform(split_df[ALL_TABULAR_FEATURES])
        out_path = FEATURES_DIR / f"X_{split_name}_tabular.npy"
        np.save(str(out_path), X_enc.values.astype("float32"))
        logger.info("Saved %s: shape=%s → %s", split_name, X_enc.shape, out_path)

    # Also save column names for downstream tasks
    col_path = FEATURES_DIR / "tabular_columns.json"
    encoder_out = encoder.transform(df[ALL_TABULAR_FEATURES].head(1))
    with open(str(col_path), "w") as fh:
        json.dump(list(encoder_out.columns), fh)
    logger.info("tabular_columns.json updated: %d columns", len(encoder_out.columns))


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="csip_etl",
    default_args=_DEFAULT_ARGS,
    description="Daily ETL: raw CSV → clean → validate → PostgreSQL → feature arrays",
    schedule_interval="0 2 * * *",    # 02:00 UTC daily
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["csip", "etl", "data"],
) as dag:

    check_raw_data = PythonOperator(
        task_id="check_raw_data",
        python_callable=_check_raw_data,
    )

    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=_clean_data,
    )

    validate_schema = PythonOperator(
        task_id="validate_schema",
        python_callable=_validate_schema,
    )

    etl_to_postgres = PythonOperator(
        task_id="etl_to_postgres",
        python_callable=_etl_to_postgres,
    )

    build_feature_arrays = PythonOperator(
        task_id="build_feature_arrays",
        python_callable=_build_feature_arrays,
    )

    # Task dependency chain
    check_raw_data >> clean_data >> validate_schema >> etl_to_postgres >> build_feature_arrays
