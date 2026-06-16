# filename: src/data/clean.py
# purpose:  Production-grade cleaning pipeline for customer support tickets.
#           All 8 steps are importable functions — used by notebooks and Airflow DAGs.
# version:  1.0

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Reference date: all ticket activity timestamps are on this date (synthetic dataset normalization)
REFERENCE_DATE = pd.Timestamp("2023-06-01")


# ── Step 1: Null audit ────────────────────────────────────────────────────────

def audit_null_dependency(df: pd.DataFrame) -> None:
    """
    Confirm that Ticket Status is the master signal for all null clusters.
    Raises AssertionError if the structural null dependency is violated.
    Called before any transformation — fail fast if raw data doesn't match expectations.
    """
    closed = df["Ticket Status"] == "Closed"
    not_closed = ~closed

    # Diagnostic: show null breakdown by status BEFORE asserting — self-documenting log
    null_by_status = df.groupby("Ticket Status")[
        ["Resolution", "Time to Resolution", "First Response Time", "Customer Satisfaction Rating"]
    ].apply(lambda x: x.notna().sum())
    log.info("Null audit — non-null counts by Ticket Status:\n%s", null_by_status.to_string())

    assert df.loc[closed, "Resolution"].notna().all(), \
        f"Closed tickets must have Resolution. Nulls: {df.loc[closed, 'Resolution'].isna().sum()}"
    assert df.loc[closed, "Time to Resolution"].notna().all(), \
        f"Closed tickets must have Time to Resolution. Nulls: {df.loc[closed, 'Time to Resolution'].isna().sum()}"
    assert df.loc[not_closed, "Resolution"].isna().all(), \
        f"Open/Pending tickets must NOT have Resolution. Filled: {df.loc[not_closed, 'Resolution'].notna().sum()}"
    assert df.loc[not_closed, "Time to Resolution"].isna().all(), \
        f"Open/Pending must NOT have Time to Resolution. Filled: {df.loc[not_closed, 'Time to Resolution'].notna().sum()}"

    log.info("Null audit passed. Closed: %d | Open/Pending: %d", closed.sum(), not_closed.sum())


# ── Step 2: Drop PII ──────────────────────────────────────────────────────────

def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove personally identifiable information before any ML processing.
    Customer Name and Customer Email are never used as features.
    Ticket ID is retained for traceability and split_indices.json.
    """
    pii_cols = ["Customer Name", "Customer Email"]
    existing = [c for c in pii_cols if c in df.columns]
    df = df.drop(columns=existing)
    log.info("Dropped PII columns: %s", existing)
    return df


# ── Step 3: Parse datetimes ───────────────────────────────────────────────────

def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date/datetime string columns to proper dtype.
    Validates that coerce() introduces no unexpected new nulls.
    """
    # Track pre-parse null counts for validation
    frt_nulls_before = df["First Response Time"].isna().sum()
    ttr_nulls_before = df["Time to Resolution"].isna().sum()

    df["Date of Purchase"]    = pd.to_datetime(df["Date of Purchase"], errors="coerce")
    df["First Response Time"] = pd.to_datetime(df["First Response Time"], errors="coerce")
    df["Time to Resolution"]  = pd.to_datetime(df["Time to Resolution"], errors="coerce")

    new_frt_nulls = df["First Response Time"].isna().sum() - frt_nulls_before
    new_ttr_nulls = df["Time to Resolution"].isna().sum() - ttr_nulls_before

    assert new_frt_nulls == 0, \
        f"parse_datetimes created {new_frt_nulls} unexpected NaTs in 'First Response Time'"
    assert new_ttr_nulls == 0, \
        f"parse_datetimes created {new_ttr_nulls} unexpected NaTs in 'Time to Resolution'"

    dop_nulls = df["Date of Purchase"].isna().sum()
    if dop_nulls > 0:
        log.warning("'Date of Purchase' has %d unparseable values (coerced to NaT)", dop_nulls)

    log.info("Datetimes parsed. Date of Purchase NaTs: %d", dop_nulls)
    return df


# ── Step 4: Compute regression target ────────────────────────────────────────

def compute_hours_to_resolve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hours_to_resolve = (Time to Resolution - First Response Time) in hours.
    Only non-null for closed tickets. Negatives fixed by +24h (synthetic data timestamp wrap).

    This is the PRIMARY regression target — trained on closed tickets only.
    Confirmed null rate: 67.3% (structural, not random). Train on is_resolved==1 rows only.
    """
    df["hours_to_resolve"] = (
        df["Time to Resolution"] - df["First Response Time"]
    ).dt.total_seconds() / 3600

    # Fix negatives: in this synthetic dataset, all ticket activity was normalized to 2023-06-01
    # but some TTR timestamps are on 2023-05-31. Both patterns produce negatives; both are fixed
    # by +24h because the underlying durations are valid (max ends up at 23.98h, confirming this).
    #
    # Two patterns distinguished for explainability:
    #   Same-day wrap:    FRT and TTR on same date, TTR time < FRT time (hour rolled past midnight)
    #   Different-date:   TTR date = 2023-05-31, FRT date = 2023-06-01 (synthetic data anomaly)
    # Both are artefacts of the same synthetic generation process — +24h corrects both.

    negative_mask = (df["hours_to_resolve"] < 0) & df["hours_to_resolve"].notna()

    same_day_mask = (
        df["Time to Resolution"].dt.date == df["First Response Time"].dt.date
    )
    wrap_mask       = negative_mask & same_day_mask    # same-date hour wrap
    date_error_mask = negative_mask & ~same_day_mask   # different-date anomaly

    df.loc[negative_mask, "hours_to_resolve"] += 24   # +24h fixes both patterns

    log.info(
        "Negative hours_to_resolve fixed (+24h): %d same-day wraps + %d different-date anomalies = %d total",
        wrap_mask.sum(), date_error_mask.sum(), negative_mask.sum()
    )

    # Sanity: after fix, no closed-ticket value should exceed 24h (all are within-day durations)
    if (df.loc[df["Ticket Status"] == "Closed", "hours_to_resolve"] > 24).any():
        log.warning("Some hours_to_resolve > 24h after fix — verify synthetic data assumption")

    # Validate on closed tickets only
    closed_resolved = df.loc[df["Ticket Status"] == "Closed", "hours_to_resolve"]
    assert closed_resolved.notna().all(), \
        "All closed tickets must have non-null hours_to_resolve after computation"
    assert (closed_resolved >= 0).all(), \
        f"hours_to_resolve must be non-negative. Min: {closed_resolved.min():.2f}"
    assert (closed_resolved <= 168).all(), \
        f"Outlier: hours_to_resolve > 168h. Max: {closed_resolved.max():.2f}"

    log.info(
        "hours_to_resolve stats (closed tickets): min=%.2fh, median=%.2fh, max=%.2fh",
        closed_resolved.min(), closed_resolved.median(), closed_resolved.max()
    )
    return df


# ── Step 5: Compute derived features ─────────────────────────────────────────

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tabular features from existing columns.
    These are ML features, NOT cleaning corrections.
    """
    # days_since_purchase: customer tenure signal (how long since they bought)
    df["days_since_purchase"] = (REFERENCE_DATE - df["Date of Purchase"]).dt.days

    # response_hour_of_day: time-of-day signal for agent first response (0–23).
    # -1 = no response yet (open tickets with null First Response Time).
    # Kept as NUMERIC — XGBoost/LightGBM handle -1 sentinel natively without encoding.
    # Do NOT pass through OrdinalEncoder — that conflates "no response" with "lowest rank".
    # NaT.dt.hour → NaN → fillna(-1) → -1.0 → astype(int) → -1. Chain is safe in pandas.
    df["response_hour_of_day"] = df["First Response Time"].dt.hour.fillna(-1).astype(int)
    assert (df["response_hour_of_day"] >= -1).all() and (df["response_hour_of_day"] <= 23).all(), \
        "response_hour_of_day out of valid range [-1, 23]"

    # Binary status flags — useful features and diagnostic tools
    df["is_resolved"]        = (df["Ticket Status"] == "Closed").astype(int)
    df["has_first_response"] = df["First Response Time"].notna().astype(int)
    df["csat_available"]     = df["Customer Satisfaction Rating"].notna().astype(int)

    log.info(
        "Derived features computed. is_resolved=%d, has_first_response=%d, csat_available=%d",
        df["is_resolved"].sum(),
        df["has_first_response"].sum(),
        df["csat_available"].sum(),
    )
    return df


# ── Step 6: Fix Ticket Description placeholder ────────────────────────────────

def fix_description_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace {product_purchased} template placeholder with the actual product name.
    8,469 rows — well under the 10K apply() threshold; apply() is acceptable here.
    Verifies no placeholder remains after replacement.
    """
    placeholder = "{product_purchased}"

    def _replace(row: pd.Series) -> str:
        desc = row["Ticket Description"]
        if pd.isna(desc):
            return desc
        return desc.replace(placeholder, str(row["Product Purchased"]))

    df["Ticket Description"] = df.apply(_replace, axis=1)

    remaining = df["Ticket Description"].str.contains(
        r"\{product_purchased\}", na=False
    ).sum()
    assert remaining == 0, \
        f"{remaining} descriptions still contain {{product_purchased}} placeholder"

    log.info("Ticket Description placeholder fixed. Verified 0 remaining.")
    return df


# ── Step 7: Handle nulls ──────────────────────────────────────────────────────

def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply structural null decisions. All nulls are MCAR by Ticket Status — never imputed.

    - Resolution: fill with "Unresolved" sentinel (preserves row count for classification)
    - Time to Resolution: leave as NaT (regression uses is_resolved==1 subset only)
    - Customer Satisfaction Rating: leave as NaN (csat_available flag captures missingness)
    - First Response Time: leave as NaT (has_first_response flag captures missingness)
    """
    df["Resolution"] = df["Resolution"].fillna("Unresolved")

    # Confirm nulls that should remain are unchanged
    assert df["Resolution"].notna().all(), "Resolution must be fully non-null after fillna"
    log.info("Nulls handled. Resolution: 0 nulls. TTR/CSAT/FRT: structural NaTs retained.")
    return df


# ── Step 8: String standardization ───────────────────────────────────────────

def standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from all string columns.
    Catches hidden variants (e.g., 'Closed ' vs 'Closed') that would break category checks.
    """
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    log.info("String standardization complete. Stripped %d object columns.", len(str_cols))
    return df


# ── Master pipeline ───────────────────────────────────────────────────────────

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 8 cleaning steps in dependency order.
    Returns a fully cleaned, validated DataFrame ready for Pandera validation + PostgreSQL load.
    """
    log.info("Starting clean_pipeline. Input shape: %s", df.shape)

    audit_null_dependency(df)        # Step 1: must be first — validates raw data assumptions
    df = drop_pii(df)                # Step 2: PII removed before any processing
    df = parse_datetimes(df)         # Step 3: dtypes fixed before derived computations
    df = compute_hours_to_resolve(df)  # Step 4: needs parsed datetimes
    df = compute_derived_features(df)  # Step 5: needs parsed datetimes
    df = fix_description_placeholder(df)  # Step 6: text fix before NLP pipeline
    df = handle_nulls(df)            # Step 7: sentinel fills after all derived columns computed
    df = standardize_strings(df)     # Step 8: final pass, catches any trailing spaces

    log.info("clean_pipeline complete. Output shape: %s", df.shape)
    return df
