# filename: notebooks/01_data_cleaning.py
# purpose:  End-to-end data cleaning notebook. Runs BEFORE 02_eda.py.
#           Produces data/processed/cleaned_tickets.csv and loads to PostgreSQL.
# version:  1.0
# run:      python notebooks/01_data_cleaning.py   (terminal, no GPU needed)

# %% ── Notebook compatibility (terminal + Colab) ──────────────────────────────
import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from google.colab import drive
    drive.mount("/content/drive")
    sys.path.insert(0, "/content/drive/MyDrive/CSIP")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# %% ── Imports ────────────────────────────────────────────────────────────────
import logging

import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.clean import clean_pipeline
from src.data.validate import TICKET_TYPES, validate
from src.data.etl import load_csv_to_postgres

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("01_data_cleaning")

# %% ── 1. Load raw CSV ────────────────────────────────────────────────────────
raw_files = list(RAW_DATA_DIR.glob("*.csv"))
assert len(raw_files) > 0, f"No CSV found in {RAW_DATA_DIR}. Place dataset there first."
RAW_CSV = raw_files[0]
log.info("Loading raw data from: %s", RAW_CSV.name)

df_raw = pd.read_csv(RAW_CSV)
print(f"\nRaw data loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"Columns: {df_raw.columns.tolist()}")

# %% ── 2. Pre-clean null summary ──────────────────────────────────────────────
print("\n=== NULL COUNTS (raw) ===")
null_summary = df_raw.isnull().sum()
print(null_summary[null_summary > 0].to_string())

print("\n=== TICKET STATUS DISTRIBUTION ===")
print(df_raw["Ticket Status"].value_counts(dropna=False).to_string())

print("\n=== TICKET TYPE VALUE COUNTS ===")
print(df_raw["Ticket Type"].value_counts(dropna=False).to_string())
print("\n[CHECK] Ticket Type values vs validate.py schema:")
actual_types = set(df_raw["Ticket Type"].dropna().unique())
print(f"  Actual:   {sorted(actual_types)}")
print(f"  Expected: {sorted(TICKET_TYPES)}")
if not actual_types.issubset(TICKET_TYPES):
    unexpected = actual_types - TICKET_TYPES
    print(f"  [WARNING] UNEXPECTED VALUES - schema mismatch: {unexpected}")
elif actual_types != TICKET_TYPES:
    missing = TICKET_TYPES - actual_types
    print(f"  [INFO] Values in schema not in data: {missing}")

# %% ── 3. Run cleaning pipeline ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("RUNNING CLEAN PIPELINE")
print("=" * 60)
df_clean = clean_pipeline(df_raw.copy())

print(f"\nAfter cleaning: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"New columns added: {sorted(set(df_clean.columns) - set(df_raw.columns))}")

# %% ── 4. Post-clean verification ─────────────────────────────────────────────
print("\n=== NULL COUNTS (cleaned) ===")
null_after = df_clean.isnull().sum()
print(null_after[null_after > 0].to_string())

print("\n=== HOURS TO RESOLVE (closed tickets) ===")
closed = df_clean[df_clean["is_resolved"] == 1]
open_pend = df_clean[df_clean["is_resolved"] == 0]
print(f"Closed tickets:       {len(closed):,}")
print(f"Open/Pending tickets: {len(open_pend):,}")
if len(closed) > 0:
    print(f"\nhours_to_resolve stats:")
    print(closed["hours_to_resolve"].describe().round(2).to_string())

print("\n=== DAYS SINCE PURCHASE ===")
print(df_clean["days_since_purchase"].describe().round(1).to_string())

print("\n=== DERIVED FLAG COUNTS ===")
print(f"  is_resolved:        {df_clean['is_resolved'].sum()} closed")
print(f"  has_first_response: {df_clean['has_first_response'].sum()} with response")
print(f"  csat_available:     {df_clean['csat_available'].sum()} with CSAT score")

print("\n=== SAMPLE CLEANED DESCRIPTION (placeholder replaced) ===")
sample_row = df_clean[df_clean["Ticket Description"].str.len() > 30].iloc[0]
print(f"  Product: {sample_row['Product Purchased']}")
print(f"  Desc:    {sample_row['Ticket Description'][:120]}")

# %% ── 5. Pandera validation ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RUNNING PANDERA VALIDATION")
print("=" * 60)

try:
    df_validated = validate(df_clean)
    print("\n[PASS] Pandera validation PASSED")
except Exception as e:
    print(f"\n[FAIL] Pandera validation FAILED:\n{e}")
    raise

# %% ── 6. Save cleaned CSV ────────────────────────────────────────────────────
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
output_path = PROCESSED_DATA_DIR / "cleaned_tickets.csv"
df_validated.to_csv(output_path, index=False)
print(f"\n[SAVED] {output_path}")
print(f"   Rows: {len(df_validated):,} | Columns: {len(df_validated.columns)}")

# %% ── 7. Load to PostgreSQL ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOADING TO POSTGRESQL")
print("=" * 60)

try:
    rows_loaded = load_csv_to_postgres(df=df_validated)
    print(f"[PASS] PostgreSQL load complete. Rows upserted: {rows_loaded:,}")
except Exception as e:
    print(f"[SKIP] PostgreSQL load failed (Docker not running?): {e}")
    print("   CSV saved successfully — PostgreSQL load can be retried later.")

# %% ── 8. Final summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1 — DATA CLEANING COMPLETE")
print("=" * 60)
print(f"  Raw rows:       {len(df_raw):,}")
print(f"  Cleaned rows:   {len(df_validated):,}  (same — no rows dropped)")
print(f"  Cleaned cols:   {len(df_validated.columns)}  ({len(df_raw.columns)} raw + {len(df_validated.columns)-len(df_raw.columns)+2} new - 2 PII)")
print(f"  Output file:    {output_path}")
print(f"  Closed tickets: {df_validated['is_resolved'].sum():,}  (regression training data)")
print(f"  Open/Pending:   {(df_validated['is_resolved']==0).sum():,}  (classification only)")
print("\nNext step: run  python notebooks/02_eda.py")
