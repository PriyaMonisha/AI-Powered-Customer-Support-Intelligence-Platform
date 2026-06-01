# filename: notebooks/03_preprocessing.py
# purpose:  Section 3 — text meta-features (VADER sentiment, char/word counts) + TF-IDF EDA
# version:  1.0
#
# Convention: logger.info() for status/progress messages
#             print() only for DataFrame/table display (logging mangles whitespace)
# Smoke tests in notebooks use assert (notebooks not run with -O flag).
# src/ library code uses explicit if+raise (may run with -O in Docker prod images).

# %% Block 0 — PROJECT_ROOT / sys.path
import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# %% Block 1 — Imports + logging
import logging

import pandas as pd

from config import (
    FAST_MODE,
    PROCESSED_DATA_DIR,
    PREPROCESSED_DATA_PATH,
    VADER_POSITIVE_THRESHOLD,
    VADER_NEGATIVE_THRESHOLD,
)
from src.features.text_features import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("section_03")
logger.info("FAST_MODE = %s", FAST_MODE)

# %% Block 2 — Load cleaned_tickets.csv
df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_tickets.csv")
original_col_count = df.shape[1]   # stored for Block 6 increment assertion
logger.info("Loaded cleaned_tickets.csv: %d rows x %d cols", df.shape[0], df.shape[1])

if df.shape[0] < 8000:
    logger.warning("Row count %d lower than expected (~8469). Check ETL.", df.shape[0])
if df.shape[0] > 10000:
    logger.warning(
        "Row count %d higher than expected. Check for duplicates.", df.shape[0]
    )
if df.shape[0] < 1000:
    raise ValueError(f"Dataset too small ({df.shape[0]} rows). ETL may have failed.")

# %% Block 3 — Add text meta-features
preprocessor = TextPreprocessor()
df = preprocessor.add_text_meta_features(df)   # reassign — returns copy, never mutates input
logger.info("After meta-features: %d rows x %d cols", df.shape[0], df.shape[1])

# Block 3 assertion guards Block 6 save — if add_text_meta_features fails, save is blocked
for col in ["char_count", "word_count", "subject_word_count", "sentiment_compound"]:
    assert col in df.columns, f"Missing expected column: {col}"

print("\n--- Text meta-feature stats ---")
print(
    df[["char_count", "word_count", "subject_word_count", "sentiment_compound"]].describe()
)

# %% Block 4 — Sentiment distribution
sc = df["sentiment_compound"]

# NaN values return False for all three conditions → excluded from all counts (correct)
_pos = (sc > VADER_POSITIVE_THRESHOLD).sum()
_neg = (sc < VADER_NEGATIVE_THRESHOLD).sum()
_neu = sc.between(
    VADER_NEGATIVE_THRESHOLD,
    VADER_POSITIVE_THRESHOLD,
    inclusive="both",   # explicit — pandas >= 1.2.0; boundaries → neutral
).sum()

# int() cast: explicit intent, prevents numpy scalar comparison surprises
assert _pos + _neu + _neg == int(sc.notna().sum()), (
    f"Sentiment counts ({_pos}+{_neu}+{_neg}={_pos+_neu+_neg}) "
    f"!= notna ({sc.notna().sum()})"
)

_n_valid = int(_pos + _neu + _neg)

if _n_valid == 0:
    logger.warning(
        "No sentiment scores computed. "
        "All Ticket Description values may be null. Check ETL output."
    )
    _pct = lambda n: "N/A"
else:
    _pct = lambda n: f"{100 * n / _n_valid:.1f}%"

logger.info("Sentiment distribution:")
logger.info("  Positive (> %.2f):  %d (%s of scored)", VADER_POSITIVE_THRESHOLD, _pos, _pct(_pos))
logger.info("  Neutral  (%.2f..%.2f): %d (%s of scored)",
            VADER_NEGATIVE_THRESHOLD, VADER_POSITIVE_THRESHOLD, _neu, _pct(_neu))
logger.info("  Negative (< %.2f): %d (%s of scored)", VADER_NEGATIVE_THRESHOLD, _neg, _pct(_neg))
logger.info("  N/A (null desc):  %d", len(df) - _n_valid)

# %% Block 5 — Fit TF-IDF (exploratory only)
logger.info("Fitting exploratory TF-IDF (NOT saved -- Section 4 re-fits on train split)")
preprocessor.fit_tfidf_exploratory(df["Ticket Description"])
logger.info("TF-IDF vocabulary size: %s", preprocessor.exploratory_vocab_size)

# Both calls below process df["Ticket Description"] through the same vectorizer.
# Full corpus cleaned and transformed twice (once per call) — acceptable EDA cost.
# Production path (Section 4+) calls transform_tfidf() once per request.
top_overall = preprocessor.get_top_terms_overall(df["Ticket Description"])
logger.info("Top 20 TF-IDF terms (overall):")
print("  " + " | ".join(top_overall))   # print() per convention — list in log = truncated

top_per_class = preprocessor.get_top_terms_per_class(
    df, "Ticket Description", "Ticket Type", n=10
)
logger.info("Top 10 TF-IDF terms per Ticket Type:")
for cls, terms in top_per_class.items():
    print(f"  {cls:<30} {' | '.join(terms)}")

# %% Block 6 — Save preprocessed_tickets.csv
# Block 3's assert ensures all 4 columns exist before we save
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(PREPROCESSED_DATA_PATH, index=False)

# Read back header to verify — real I/O smoke test
# (assert acceptable in notebooks — not run with -O flag per file-top comment)
saved = pd.read_csv(PREPROCESSED_DATA_PATH, nrows=0)
assert saved.shape[1] == original_col_count + 4, (
    f"Expected {original_col_count + 4} cols, got {saved.shape[1]}"
)
for col in ["char_count", "word_count", "subject_word_count", "sentiment_compound"]:
    assert col in saved.columns, f"Missing expected column in saved file: {col}"

logger.info(
    "[OK] preprocessed_tickets.csv saved: %d rows x %d cols",
    df.shape[0],
    df.shape[1],
)

# %% Block 7 — Completion summary (fully self-contained)
# Re-derives from saved file — does NOT depend on Block 4/5 variables.
# _b7_pct defined here rather than reusing Block 4's _pct — intentional:
# Block 7 must be independently re-runnable; reusing _pct reintroduces cross-block dep.
df_check = pd.read_csv(
    PREPROCESSED_DATA_PATH,
    usecols=["char_count", "word_count", "sentiment_compound"],  # targeted read
)
sc2 = df_check["sentiment_compound"]
_b7_pos = (sc2 > VADER_POSITIVE_THRESHOLD).sum()
_b7_neg = (sc2 < VADER_NEGATIVE_THRESHOLD).sum()
_b7_neu = sc2.between(
    VADER_NEGATIVE_THRESHOLD, VADER_POSITIVE_THRESHOLD, inclusive="both"
).sum()
_b7_valid = int(_b7_pos + _b7_neu + _b7_neg)
_b7_pct = (lambda n: f"{100 * n / _b7_valid:.1f}%") if _b7_valid > 0 else (lambda n: "N/A")
_b7_vocab = preprocessor.exploratory_vocab_size or "N/A"

logger.info("=" * 60)
logger.info("Section 3 Complete")
logger.info("  Output:          %s", PREPROCESSED_DATA_PATH)
logger.info("  Rows:            %d", df_check.shape[0])
logger.info("  Mean char_count: %.1f", df_check["char_count"].mean())
logger.info("  Mean word_count: %.1f", df_check["word_count"].mean())
logger.info("  Sentiment pos:   %d (%s of scored)", _b7_pos, _b7_pct(_b7_pos))
logger.info("  Sentiment neu:   %d (%s of scored)", _b7_neu, _b7_pct(_b7_neu))
logger.info("  Sentiment neg:   %d (%s of scored)", _b7_neg, _b7_pct(_b7_neg))
logger.info("  TF-IDF vocab:    %s terms", _b7_vocab)
logger.info("[OK] Section 3 preprocessing complete")
logger.info("=" * 60)
