# filename: src/data/etl.py
# purpose:  Load cleaned ticket data into PostgreSQL tickets table.
#           Idempotent upsert on Ticket ID — safe to re-run on Airflow retry.
# version:  1.0

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_URL, PROCESSED_DATA_DIR

log = logging.getLogger(__name__)

TABLE_NAME = "tickets"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    "Ticket ID"                    INTEGER PRIMARY KEY,
    "Customer Age"                 SMALLINT,
    "Customer Gender"              VARCHAR(20),
    "Product Purchased"            VARCHAR(100),
    "Date of Purchase"             DATE,
    "Ticket Type"                  VARCHAR(50),
    "Ticket Priority"              VARCHAR(20),
    "Ticket Channel"               VARCHAR(30),
    "Ticket Status"                VARCHAR(40),
    "Ticket Subject"               VARCHAR(200),
    "Ticket Description"           TEXT,
    "Resolution"                   TEXT,
    "First Response Time"          TIMESTAMP,
    "Time to Resolution"           TIMESTAMP,
    "Customer Satisfaction Rating" NUMERIC(3,1),
    "hours_to_resolve"             NUMERIC(8,4),
    "days_since_purchase"          INTEGER,
    "response_hour_of_day"         SMALLINT,
    "is_resolved"                  SMALLINT,
    "has_first_response"           SMALLINT,
    "csat_available"               SMALLINT
);
"""

UPSERT_SQL = f"""
INSERT INTO {TABLE_NAME} ({{}})
VALUES ({{}})
ON CONFLICT ("Ticket ID") DO UPDATE SET
    {{}}
"""


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


def create_table_if_not_exists(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    log.info("Table '%s' ensured.", TABLE_NAME)


def upsert_tickets(df: pd.DataFrame, engine, chunk_size: int = 500) -> int:
    """
    Upsert df into the tickets table in chunks.
    Uses INSERT ... ON CONFLICT (Ticket ID) DO UPDATE SET — idempotent.
    Returns total rows upserted.
    """
    cols = list(df.columns)
    col_list  = ", ".join(f'"{c}"' for c in cols)
    val_list  = ", ".join(f":{c.replace(' ', '_')}" for c in cols)
    update_list = ", ".join(
        f'"{c}" = EXCLUDED."{c}"' for c in cols if c != "Ticket ID"
    )

    upsert = text(
        f"INSERT INTO {TABLE_NAME} ({col_list}) VALUES ({val_list}) "
        f'ON CONFLICT ("Ticket ID") DO UPDATE SET {update_list}'
    )

    # Rename columns to valid SQL parameter names
    param_df = df.copy()
    param_df.columns = [c.replace(" ", "_") for c in param_df.columns]

    total = 0
    for i in range(0, len(param_df), chunk_size):
        chunk = param_df.iloc[i : i + chunk_size]
        records = chunk.astype(object).where(pd.notna(chunk), other=None).to_dict("records")
        with engine.begin() as conn:
            conn.execute(upsert, records)
        total += len(records)
        log.debug("Upserted chunk %d–%d", i, i + len(records))

    log.info("Upserted %d rows into '%s'.", total, TABLE_NAME)
    return total


def load_csv_to_postgres(
    csv_path: Path | None = None,
    df: pd.DataFrame | None = None,
) -> int:
    """
    Main entry point: load cleaned CSV (or DataFrame) to PostgreSQL.
    Accepts either a file path or a pre-loaded DataFrame.
    Returns rows upserted.
    """
    if df is None:
        if csv_path is None:
            csv_path = PROCESSED_DATA_DIR / "cleaned_tickets.csv"
        log.info("Loading from %s", csv_path)
        df = pd.read_csv(
            csv_path,
            parse_dates=["Date of Purchase", "First Response Time", "Time to Resolution"],
        )

    engine = get_engine()
    create_table_if_not_exists(engine)
    return upsert_tickets(df, engine)
