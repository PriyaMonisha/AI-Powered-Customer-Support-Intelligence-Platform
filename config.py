# filename: config.py
# purpose:  Central configuration — all constants, paths, thresholds
# version:  1.0

FAST_MODE = True   # FIRST LINE — set True for dev/interview, False for full production run

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREPROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "preprocessed_tickets.csv"
CHARTS_DIR         = ARTIFACTS_DIR / "charts"
REPORTS_DIR        = ARTIFACTS_DIR / "reports"
DRIFT_DIR          = ARTIFACTS_DIR / "drift"

# ---------------------------------------------------------------------------
# Connections (read from .env — NEVER hardcode real credentials)
# ---------------------------------------------------------------------------
DB_URL              = os.getenv("DB_URL",       "postgresql+psycopg2://postgres:postgres@localhost:5432/csip")
REDIS_URL           = os.getenv("REDIS_URL",    "redis://localhost:6379")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI",   "http://localhost:5001")
HF_HUB_TOKEN        = os.getenv("HF_HUB_TOKEN", "")
HF_USERNAME         = os.getenv("HF_USERNAME",  "")   # used as: f"{HF_USERNAME}/csip-distilbert"
ADMIN_API_KEY       = os.getenv("ADMIN_API_KEY", "")

# ---------------------------------------------------------------------------
# ML constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Confidence thresholds (business rules — locked, do not tune as hyperparameters)
AUTO_ROUTE      = 0.85   # confidence ≥ this → auto-route ticket to correct queue
FLAG_THRESHOLD  = 0.60   # [0.60, 0.85) → flag for human review; < 0.60 → human triage

# Retraining guards (used in model_retraining_dag)
PROMOTE_THRESHOLD    = 0.02   # new model F1 must exceed champion by ≥ 2%
REGRESSION_THRESHOLD = 0.05   # alert if new model F1 is ≥ 5% WORSE than champion

# Drift detection
DRIFT_KS_THRESHOLD = 0.10   # KS statistic above this triggers retraining DAG

# SLA breach thresholds (hours per priority level)
SLA_CRITICAL_HOURS = 4
SLA_HIGH_HOURS     = 24
SLA_MEDIUM_HOURS   = 72
SLA_LOW_HOURS      = 168    # 7 days

# ---------------------------------------------------------------------------
# Standardized model artifact paths (all sections reference these — no ad-hoc filenames)
# ---------------------------------------------------------------------------
BASELINE_TYPE_PATH         = MODELS_DIR / "baseline_type_classifier.pkl"
BASELINE_PRIORITY_PATH     = MODELS_DIR / "baseline_priority_classifier.pkl"
XGBOOST_REGRESSOR_PATH     = MODELS_DIR / "xgboost_regressor.pkl"

# Section 6 — advanced classifier paths (RF / XGBoost / LightGBM, one pkl per task)
RF_TYPE_PATH         = MODELS_DIR / "rf_type_classifier.pkl"
RF_PRIORITY_PATH     = MODELS_DIR / "rf_priority_classifier.pkl"
XGB_TYPE_PATH        = MODELS_DIR / "xgb_type_classifier.pkl"
XGB_PRIORITY_PATH    = MODELS_DIR / "xgb_priority_classifier.pkl"
LGBM_TYPE_PATH       = MODELS_DIR / "lgbm_type_classifier.pkl"
LGBM_PRIORITY_PATH   = MODELS_DIR / "lgbm_priority_classifier.pkl"
LIGHTGBM_CLASSIFIER_PATH = LGBM_PRIORITY_PATH   # alias — remove after S11 wires FastAPI
BILSTM_PATH              = MODELS_DIR / "bilstm.pt"
DISTILBERT_PATH          = MODELS_DIR / "distilbert"    # directory (save_pretrained output)
SHAP_EXPLAINER_PATH      = MODELS_DIR / "shap_explainer.pkl"
SPLIT_INDICES_PATH       = PROCESSED_DATA_DIR / "split_indices.json"
TABULAR_ENCODER_PATH     = MODELS_DIR / "tabular_encoder.pkl"
PREPROCESSOR_PATH        = MODELS_DIR / "preprocessor.pkl"
FEATURES_DIR             = PROCESSED_DATA_DIR / "features"
LE_TYPE_PATH             = MODELS_DIR / "le_ticket_type.pkl"
LE_PRIORITY_PATH         = MODELS_DIR / "le_ticket_priority.pkl"
DRIFT_BASELINE_PATH      = DRIFT_DIR / "training_baseline.json"
MODEL_REGISTRY_PATH      = REPORTS_DIR / "model_registry.json"
CONFIDENCE_THRESHOLDS_PATH = REPORTS_DIR / "confidence_thresholds.json"

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
API_TITLE   = "Customer Support Intelligence Platform"
API_VERSION = "1.0.0"
ADMIN_RELOAD_RATE_LIMIT_SECONDS = 600   # max 1 /admin/reload per 10 minutes

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
FAST_N_TRIALS  = 3     # Optuna trials in FAST_MODE
FAST_CV_FOLDS  = 3     # cross-val folds in FAST_MODE
FAST_SAMPLE_N  = 10_000  # max rows for Optuna CV in FAST_MODE

FULL_N_TRIALS  = 25
FULL_CV_FOLDS  = 5

# TF-IDF settings
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE  = (1, 2)

# VADER sentiment thresholds (standard boundaries per VADER paper)
VADER_POSITIVE_THRESHOLD =  0.05
VADER_NEGATIVE_THRESHOLD = -0.05

# Redis feature store
REDIS_FEATURE_TTL = 86_400   # 24 hours (seconds)
REDIS_KEY_PREFIX  = "csip:features"
