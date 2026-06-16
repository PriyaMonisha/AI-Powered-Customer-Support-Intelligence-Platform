# filename: src/features/inference.py
# purpose:  Build model-ready feature rows from a TicketRequest at inference time
# version:  1.0

import json
import logging

import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR
from src.features.tabular_features import ALL_TABULAR_FEATURES

log = logging.getLogger(__name__)

# Training median saved at notebook time — avoids reading the gitignored CSV at import.
_TRAINING_STATS_PATH = ARTIFACTS_DIR / "metrics" / "training_stats.json"
try:
    with open(_TRAINING_STATS_PATH) as _f:
        DAYS_SINCE_PURCHASE_MEDIAN: float = float(json.load(_f)["days_since_purchase_median"])
except Exception:
    DAYS_SINCE_PURCHASE_MEDIAN = 882.0  # fallback: training set median


def build_inference_row(req) -> pd.DataFrame:
    """Build a single-row raw DataFrame. Column names must match training exactly."""
    days = req.days_since_purchase
    if days is None:
        days = DAYS_SINCE_PURCHASE_MEDIAN

    return pd.DataFrame([{
        "Ticket Subject": req.ticket_subject,
        "Ticket Description": req.ticket_description,
        "Customer Age": req.customer_age,
        "Customer Gender": req.customer_gender.value,
        "Product Purchased": req.product_purchased,
        "Ticket Channel": req.ticket_channel.value,
        "days_since_purchase": days,
        "response_hour_of_day": req.response_hour_of_day,
        "is_resolved": 0,
        "has_first_response": 1 if req.response_hour_of_day >= 0 else 0,
        "csat_available": 0,
    }])


def build_tabular_features(req, models: dict) -> np.ndarray:
    """Returns a (1, 17) float64 array ready for classifier.predict_proba() / SHAP."""
    df_row = build_inference_row(req)
    df_with_meta = models["preprocessor"].add_text_meta_features(df_row)
    df_for_enc = df_with_meta[ALL_TABULAR_FEATURES]
    encoded = models["tabular_encoder"].transform(df_for_enc)
    result = encoded.values if isinstance(encoded, pd.DataFrame) else np.asarray(encoded)
    return result.astype(np.float64)
