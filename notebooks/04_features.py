# filename: notebooks/04_features.py
# purpose:  Section 4 — Train/Val/Test split + tabular encoding + TF-IDF fit +
#           feature matrix persistence + Redis feature store.
# version:  1.0
# run:      python notebooks/04_features.py  (terminal, no GPU needed)

# %% [markdown]
# # Section 4: Tabular Features + Train/Val/Test Split + Redis Feature Store

# %% Setup — FAST_MODE must be the absolute first executable line
FAST_MODE = True

# %% Imports
import json
import logging
import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── PROJECT_ROOT — terminal (.py) and Colab (.ipynb) compatible ──────────────
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:                                    # __file__ undefined in Jupyter/Colab cells
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Internal imports ──────────────────────────────────────────────────────────
from config import (
    RANDOM_STATE,
    PREPROCESSED_DATA_PATH,
    SPLIT_INDICES_PATH,
    TABULAR_ENCODER_PATH,
    PREPROCESSOR_PATH,
    FEATURES_DIR,
    LE_TYPE_PATH,
    LE_PRIORITY_PATH,
    REDIS_URL,
    MODELS_DIR,
)
from src.features.tabular_features import TabularEncoder, ALL_TABULAR_FEATURES
from src.features.text_features import TextPreprocessor
from src.features.feature_store import (
    queue_ticket_features,
    _get_client,            # shares the feature_store default pool with FastAPI reads
)
from src.utils.helpers import serialize_ticket_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("section4")

# %% [markdown]
# ## Cell 2: Load Data

# %%
df = pd.read_csv(PREPROCESSED_DATA_PATH)
df.reset_index(drop=True, inplace=True)   # guarantee 0-based integer index
logger.info("Loaded: %d rows x %d cols", *df.shape)

PRIMARY_TARGET    = "Ticket Type"
SECONDARY_TARGET  = "Ticket Priority"
REGRESSION_TARGET = "hours_to_resolve"

# Verify column renamed from "Time to Resolution" in Section 1 ETL
assert REGRESSION_TARGET in df.columns, (
    f"'{REGRESSION_TARGET}' not found. "
    f"Time-related columns: {[c for c in df.columns if 'resolut' in c.lower() or 'time' in c.lower()]}"
)
assert PRIMARY_TARGET in df.columns, f"'{PRIMARY_TARGET}' not in df"
assert SECONDARY_TARGET in df.columns, f"'{SECONDARY_TARGET}' not in df"
logger.info("Column assertions passed.")

# %% [markdown]
# ## Cell 3: Train / Val / Test Split (70 / 10 / 20)

# %%
# Pass 1: 80% train+val / 20% test (stratified on Ticket Type)
idx_trainval, idx_test = train_test_split(
    df.index.tolist(),
    test_size=0.20,
    stratify=df[PRIMARY_TARGET],
    random_state=RANDOM_STATE,
)

# Pass 2: 87.5% train / 12.5% val  (0.125 x 0.80 = 10.0% of total)
idx_train, idx_val = train_test_split(
    idx_trainval,
    test_size=0.125,
    stratify=df.loc[idx_trainval, PRIMARY_TARGET],  # .loc — label-safe under any index regime
    random_state=RANDOM_STATE,
)

# ── Integrity assertions — fail fast before saving anything ──────────────────
total = len(idx_train) + len(idx_val) + len(idx_test)
assert total == len(df), f"Split total {total} != dataset size {len(df)}"
assert len(set(idx_train) & set(idx_val))  == 0, "Train/val overlap detected"
assert len(set(idx_train) & set(idx_test)) == 0, "Train/test overlap detected"
assert len(set(idx_val)   & set(idx_test)) == 0, "Val/test overlap detected"

# ── Save as Ticket IDs (order-independent) ────────────────────────────────────
# serialize_ticket_id() is used by the Airflow retraining DAG too — consistent type
# prevents int vs str key mismatch in Colab .isin() lookups after retraining.
SPLIT_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SPLIT_INDICES_PATH, "w") as f:
    json.dump(
        {
            # sorted() AFTER extracting IDs — sorted by Ticket ID value, not row position
            "train": sorted([serialize_ticket_id(df.iloc[i]["Ticket ID"]) for i in idx_train]),
            "val":   sorted([serialize_ticket_id(df.iloc[i]["Ticket ID"]) for i in idx_val]),
            "test":  sorted([serialize_ticket_id(df.iloc[i]["Ticket ID"]) for i in idx_test]),
        },
        f,
        indent=2,
    )

# Round-trip verify — catches disk-full truncation and serialization bugs immediately
with open(SPLIT_INDICES_PATH) as f:
    _verify = json.load(f)
assert set(_verify.keys()) == {"train", "val", "test"}, "split_indices.json missing keys"
assert len(_verify["train"]) + len(_verify["val"]) + len(_verify["test"]) == len(df), \
    "split_indices.json Ticket ID count mismatch after save"
logger.info(
    "split_indices.json verified: %d/%d/%d IDs",
    len(_verify["train"]), len(_verify["val"]), len(_verify["test"]),
)
del _verify

# ── Sub-DataFrames — sort by Ticket ID so row order matches split_indices.json ──
# split_indices.json stores sorted Ticket IDs. The Colab reload pattern uses:
#   df_test = df[df["Ticket ID"].isin(splits["test"])].reset_index(drop=True)
# .isin() preserves CSV row order, which for this dataset is ≈ sorted by Ticket ID.
# Sorting here ensures saved .npy arrays and the Colab reload produce identical row order.
df_train = df.iloc[idx_train].sort_values("Ticket ID").reset_index(drop=True)
df_val   = df.iloc[idx_val].sort_values("Ticket ID").reset_index(drop=True)
df_test  = df.iloc[idx_test].sort_values("Ticket ID").reset_index(drop=True)

# Colab reload pattern (for Sections 8/9) — produces the same row order as above:
# with open(SPLIT_INDICES_PATH) as f: splits = json.load(f)
# df_test = df[df["Ticket ID"].isin(splits["test"])].sort_values("Ticket ID").reset_index(drop=True)

logger.info(
    "Split: train=%d  val=%d  test=%d  (total=%d)",
    len(df_train), len(df_val), len(df_test), len(df),
)

# %% [markdown]
# ## Cell 4: Fit TabularEncoder (train only)

# %%
# TabularEncoder.fit_transform() returns pd.DataFrame — .values and .columns are valid
MODELS_DIR.mkdir(parents=True, exist_ok=True)

encoder = TabularEncoder()
X_train_tab = encoder.fit_transform(df_train[ALL_TABULAR_FEATURES], df_train[PRIMARY_TARGET])
X_val_tab   = encoder.transform(df_val[ALL_TABULAR_FEATURES])
X_test_tab  = encoder.transform(df_test[ALL_TABULAR_FEATURES])

encoder.save(TABULAR_ENCODER_PATH)
logger.info("TabularEncoder saved | Output shape: %s | Columns: %s",
    X_train_tab.shape, list(X_train_tab.columns))

# %% [markdown]
# ## Cell 5: Fit Production TF-IDF (train only)

# %%
preprocessor = TextPreprocessor()
preprocessor.fit_tfidf(df_train["Ticket Description"])   # production fit on train only
preprocessor.save(PREPROCESSOR_PATH)                     # raises RuntimeError if vectorizer_ is None

X_train_tfidf = preprocessor.transform_tfidf(df_train["Ticket Description"])
X_val_tfidf   = preprocessor.transform_tfidf(df_val["Ticket Description"])
X_test_tfidf  = preprocessor.transform_tfidf(df_test["Ticket Description"])

assert preprocessor.vectorizer_ is not None  # set by fit_tfidf() above
logger.info(
    "TF-IDF vocab: %d | Train: %s | Val: %s | Test: %s",
    len(preprocessor.vectorizer_.vocabulary_),
    X_train_tfidf.shape, X_val_tfidf.shape, X_test_tfidf.shape,
)

# %% [markdown]
# ## Cell 6: Save Feature Matrices

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Tabular (dense) -> .npy
np.save(FEATURES_DIR / "X_train_tabular.npy", X_train_tab.values)
np.save(FEATURES_DIR / "X_val_tabular.npy",   X_val_tab.values)
np.save(FEATURES_DIR / "X_test_tabular.npy",  X_test_tab.values)

# TF-IDF (sparse) -> .npz
sp.save_npz(str(FEATURES_DIR / "X_train_tfidf.npz"), X_train_tfidf)
sp.save_npz(str(FEATURES_DIR / "X_val_tfidf.npz"),   X_val_tfidf)
sp.save_npz(str(FEATURES_DIR / "X_test_tfidf.npz"),  X_test_tfidf)

# Column names for SHAP (Section 8c)
with open(FEATURES_DIR / "tabular_columns.json", "w") as f:
    json.dump(list(X_train_tab.columns), f, indent=2)

# ROW-ORDER CONTRACT:
# X_{split}_tabular.npy, X_{split}_tfidf.npz, and y_{split}_*.npy all share
# identical row order — same as df_{split}.reset_index(drop=True).
# Section 7+ must load X and y together; apply the same boolean mask to all arrays.
# NEVER reload df from CSV or database between loading X and y — row order will differ.

logger.info("Feature matrices saved to %s", FEATURES_DIR)

# %% [markdown]
# ## Cell 7: Label-Encode Targets

# %%
# Fit on FULL dataset label space — not just the train split.
# Retraining-safe: a rolling window that drops a rare class still produces valid encodings.
ALL_TICKET_TYPES = sorted(df[PRIMARY_TARGET].dropna().unique())
ALL_PRIORITIES   = sorted(df[SECONDARY_TARGET].dropna().unique())

le_type = LabelEncoder().fit(ALL_TICKET_TYPES)   # LabelEncoder re-sorts internally anyway
le_prio = LabelEncoder().fit(ALL_PRIORITIES)

y_train_type = le_type.transform(df_train[PRIMARY_TARGET])
y_val_type   = le_type.transform(df_val[PRIMARY_TARGET])
y_test_type  = le_type.transform(df_test[PRIMARY_TARGET])

y_train_prio = le_prio.transform(df_train[SECONDARY_TARGET])
y_val_prio   = le_prio.transform(df_val[SECONDARY_TARGET])
y_test_prio  = le_prio.transform(df_test[SECONDARY_TARGET])

# Regression: raw float with NaN for open tickets; Section 7 filters with ~np.isnan()
y_train_reg = df_train[REGRESSION_TARGET].values.astype(float)
y_val_reg   = df_val[REGRESSION_TARGET].values.astype(float)
y_test_reg  = df_test[REGRESSION_TARGET].values.astype(float)

for name, arr in [
    ("y_train_type", y_train_type), ("y_val_type",  y_val_type),  ("y_test_type",  y_test_type),
    ("y_train_prio", y_train_prio), ("y_val_prio",  y_val_prio),  ("y_test_prio",  y_test_prio),
    ("y_train_reg",  y_train_reg),  ("y_val_reg",   y_val_reg),   ("y_test_reg",   y_test_reg),
]:
    np.save(FEATURES_DIR / f"{name}.npy", arr)

# label_maps.json — JSON keys are ALWAYS strings.
# FastAPI inference decoder: class_name = label_maps["ticket_type"][str(int(prediction))]
with open(FEATURES_DIR / "label_maps.json", "w") as f:
    json.dump(
        {
            "ticket_type":     {str(i): cls for i, cls in enumerate(le_type.classes_)},
            "ticket_priority": {str(i): cls for i, cls in enumerate(le_prio.classes_)},
        },
        f,
        indent=2,
    )

joblib.dump(le_type, LE_TYPE_PATH)
joblib.dump(le_prio, LE_PRIORITY_PATH)

# Verify all classes present after stratified split
unique_type_classes = len(np.unique(y_train_type))
assert unique_type_classes == len(ALL_TICKET_TYPES), (
    f"Train split missing ticket type classes: "
    f"expected {len(ALL_TICKET_TYPES)}, got {unique_type_classes}"
)

logger.info("Label encoders saved | Types: %s | Priorities: %s",
    list(le_type.classes_), list(le_prio.classes_))
# np.bincount safe here: LabelEncoder guarantees 0-based contiguous integers.
# If retraining DAG re-fits on a subset, use Counter(y_train_type) instead.
logger.info("Train class distribution (Ticket Type): %s",
    dict(zip(le_type.classes_, np.bincount(y_train_type))))
logger.info("Regression rows (closed tickets) - train: %d  val: %d  test: %d",
    (~np.isnan(y_train_reg)).sum(),
    (~np.isnan(y_val_reg)).sum(),
    (~np.isnan(y_test_reg)).sum(),
)

# %% [markdown]
# ## Cell 8: Redis Feature Store

# %%
redis_available = False
try:
    # _get_client() uses the feature_store default pool — same connection used by FastAPI reads.
    # Do NOT use redis.from_url() here — that creates a separate pool and risks URL drift.
    _rc = _get_client()
    if _rc is not None:
        _rc.ping()
        redis_available = True
        logger.info("Redis connected via feature_store pool: %s", REDIS_URL)
except Exception as e:
    logger.warning(
        "Redis unavailable (%s). Feature store skipped — inference will recompute features.",
        e,
    )

if redis_available:
    # Vectorized dict construction — one C-level call, not iterrows()
    records  = X_train_tab.assign(split="train").to_dict(orient="records")
    train_ids = df_train["Ticket ID"].tolist()

    # Pipeline: all SETEX commands in one network round-trip
    # queue_ticket_features handles NumpyEncoder serialization and key hashing
    try:
        client = _get_client()
        if client is not None:
            pipe = client.pipeline(transaction=False)
            for ticket_id, features in zip(train_ids, records):
                queue_ticket_features(pipe, ticket_id, features)
            pipe.execute()
            logger.info("Redis: stored %d / %d train tickets", len(train_ids), len(df_train))
    except Exception as e:
        logger.warning("Redis pipeline failed: %s. Cache not populated.", e)

# %% [markdown]
# ## Cell 9: Summary

# %%
# Read from disk — verifies actual saves completed (not just in-memory state)
actual_tab_cols   = np.load(FEATURES_DIR / "X_train_tabular.npy").shape[1]
actual_tfidf_cols = sp.load_npz(str(FEATURES_DIR / "X_train_tfidf.npz")).shape[1]

with open(FEATURES_DIR / "tabular_columns.json") as f:
    col_names = json.load(f)

logger.info("=" * 60)
logger.info("SECTION 4 COMPLETE")
logger.info(
    "Split:          train=%d  val=%d  test=%d  total=%d",
    len(df_train), len(df_val), len(df_test), len(df),
)
logger.info("Tabular:        %d cols: %s", actual_tab_cols, col_names)
logger.info("TF-IDF:         %d tokens (from saved npz)", actual_tfidf_cols)
logger.info("Total features: %d", actual_tab_cols + actual_tfidf_cols)
logger.info("Redis:          %s", "connected" if redis_available else "skipped (unavailable)")
for p in [SPLIT_INDICES_PATH, TABULAR_ENCODER_PATH, PREPROCESSOR_PATH,
          LE_TYPE_PATH, LE_PRIORITY_PATH]:
    logger.info("  Saved: %s", p)
logger.info("=" * 60)
