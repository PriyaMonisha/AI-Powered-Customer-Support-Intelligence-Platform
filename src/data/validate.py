# filename: src/data/validate.py
# purpose:  Pandera schema for cleaned ticket data — all rules use REAL values from data gate.
#           Called after clean_pipeline() to enforce structural guarantees before PostgreSQL load.
# version:  1.0

import logging

import pandas as pd
from pandera import Check, Column, DataFrameSchema

log = logging.getLogger(__name__)

# ── Real values confirmed from initial_analysis.ipynb ─────────────────────────
# NOTE: verify "Account issue" as the 4th Ticket Type value from your notebook's value_counts
TICKET_TYPES    = {
    "Technical issue",
    "Billing inquiry",
    "Product inquiry",
    "Refund request",
    "Cancellation request",
}  # 5 classes confirmed from initial_analysis.ipynb — NOT the 4 in GUVI spec
TICKET_PRIORITIES = {"Critical", "High", "Medium", "Low"}
TICKET_CHANNELS = {"Email", "Chat", "Phone", "Social media"}
TICKET_STATUSES = {"Open", "Pending Customer Response", "Closed"}
GENDERS         = {"Male", "Female", "Other"}

# Row count bounds: actual count ± 10%
ROW_COUNT_MIN = 7_622   # 8469 × 0.90
ROW_COUNT_MAX = 9_316   # 8469 × 1.10


def build_schema() -> DataFrameSchema:
    """
    Returns the Pandera DataFrameSchema for cleaned_tickets.csv.
    All column names and category values are from the REAL dataset (data gate confirmed).
    coerce=True handles dtype mismatches introduced by CSV round-trips.
    """
    return DataFrameSchema(
        columns={
            # ── Identity ────────────────────────────────────────────────────
            "Ticket ID": Column(
                int, Check.greater_than(0), nullable=False, unique=True,
                description="Primary key — uniqueness validated"
            ),

            # ── Customer profile ─────────────────────────────────────────────
            "Customer Age": Column(
                int, Check.in_range(10, 100), nullable=False
            ),
            "Customer Gender": Column(
                str, Check.isin(GENDERS), nullable=False
            ),

            # ── Product / purchase ───────────────────────────────────────────
            "Product Purchased": Column(str, nullable=False),
            "Date of Purchase":  Column("datetime64[ns]", nullable=False),

            # ── Ticket classification targets ─────────────────────────────────
            "Ticket Type": Column(
                str, Check.isin(TICKET_TYPES), nullable=False,
                description="Primary NLP classification target"
            ),
            "Ticket Priority": Column(
                str, Check.isin(TICKET_PRIORITIES), nullable=False,
                description="Secondary tabular classification target"
            ),
            "Ticket Channel": Column(
                str, Check.isin(TICKET_CHANNELS), nullable=False
            ),
            "Ticket Status": Column(
                str, Check.isin(TICKET_STATUSES), nullable=False
            ),

            # ── Text fields ──────────────────────────────────────────────────
            "Ticket Subject":      Column(str, nullable=False),
            "Ticket Description":  Column(str, nullable=False),
            "Resolution":          Column(str, nullable=False),  # "Unresolved" sentinel for open tickets

            # ── Timestamp columns (nullable — open tickets lack resolution/response) ──
            "First Response Time": Column("datetime64[ns]", nullable=True),
            "Time to Resolution":  Column("datetime64[ns]", nullable=True),

            # ── CSAT (nullable — only closed tickets have ratings) ────────────
            "Customer Satisfaction Rating": Column(
                float, Check.in_range(1.0, 5.0), nullable=True
            ),

            # ── Computed regression target (nullable — open tickets) ──────────
            "hours_to_resolve": Column(
                float, Check.greater_than_or_equal_to(0), nullable=True,
                description="Regression target: (Time to Resolution - First Response Time) in hours"
            ),

            # ── Derived features ─────────────────────────────────────────────
            "days_since_purchase": Column(
                int, Check.greater_than_or_equal_to(0), nullable=False
            ),
            "response_hour_of_day": Column(
                int, Check.in_range(-1, 23), nullable=False,
                description="-1 = no response yet (open tickets)"
            ),
            "is_resolved": Column(
                int, Check.isin([0, 1]), nullable=False
            ),
            "has_first_response": Column(
                int, Check.isin([0, 1]), nullable=False
            ),
            "csat_available": Column(
                int, Check.isin([0, 1]), nullable=False
            ),
        },
        checks=[
            # Row count sanity
            Check(
                lambda df: ROW_COUNT_MIN <= len(df) <= ROW_COUNT_MAX,
                error=f"Row count must be between {ROW_COUNT_MIN} and {ROW_COUNT_MAX}"
            ),
            # Coherence: closed tickets must have hours_to_resolve
            Check(
                lambda df: df.loc[df["is_resolved"] == 1, "hours_to_resolve"].notna().all(),
                error="All closed tickets (is_resolved=1) must have non-null hours_to_resolve"
            ),
            # Coherence: open tickets must NOT have hours_to_resolve
            Check(
                lambda df: df.loc[df["is_resolved"] == 0, "hours_to_resolve"].isna().all(),
                error="Open/pending tickets (is_resolved=0) must have null hours_to_resolve"
            ),
            # Coherence: is_resolved==1 ↔ Ticket Status=="Closed"
            Check(
                lambda df: (
                    (df["is_resolved"] == 1) == (df["Ticket Status"] == "Closed")
                ).all(),
                error="is_resolved flag must align exactly with Ticket Status=='Closed'"
            ),
            # No {product_purchased} placeholder remaining in descriptions
            Check(
                lambda df: ~df["Ticket Description"].str.contains(
                    r"\{product_purchased\}", na=False
                ).any(),
                error="Ticket Description still contains {product_purchased} placeholder"
            ),
        ],
        coerce=True,   # handles int64/float64 dtype mismatches from CSV round-trips
    )


CLEANED_TICKET_SCHEMA: DataFrameSchema = build_schema()


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a cleaned DataFrame against CLEANED_TICKET_SCHEMA.
    Returns the validated (and coerced) DataFrame or raises SchemaError.
    Log a summary before returning.
    """
    log.info("Running Pandera validation on %d rows × %d cols...", len(df), len(df.columns))
    validated = CLEANED_TICKET_SCHEMA.validate(df)

    closed = (validated["Ticket Status"] == "Closed").sum()
    open_pend = len(validated) - closed
    log.info(
        "Validation PASSED. Rows: %d | Closed: %d | Open/Pending: %d | "
        "hours_to_resolve non-null: %d",
        len(validated), closed, open_pend,
        validated["hours_to_resolve"].notna().sum()
    )
    return validated
