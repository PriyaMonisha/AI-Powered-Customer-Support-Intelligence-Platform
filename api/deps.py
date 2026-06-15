# filename: api/deps.py
# purpose:  FastAPI dependencies — API key auth, model access, and shared model loading
# version:  1.0

import json
import secrets

import joblib
import numpy as np
from fastapi import Header, HTTPException, Request

from config import (
    ADMIN_API_KEY,
    CSIP_API_KEY,
    FEATURES_DIR,
    LE_PRIORITY_PATH,
    LE_TYPE_PATH,
    LGBM_TYPE_PATH,
    PREPROCESSOR_PATH,
    REGRESSOR_KEEP_MASK_PATH,
    RF_REGRESSOR_PATH,
    SHAP_EXPLAINER_PRIORITY_PATH,
    TABULAR_ENCODER_PATH,
    XGB_PRIORITY_PATH,
)
from src.features.tabular_features import TabularEncoder
from src.models.advanced_classifier import AdvancedClassifier
from src.models.advanced_regressor import AdvancedRegressor
from src.monitoring.drift import load_baseline


async def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    if not ADMIN_API_KEY:
        raise HTTPException(503, "API key not configured on server")
    if not x_api_key or not secrets.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(403, "Invalid API key")
    return x_api_key


async def verify_read_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """Read-only auth for /predict and /explain — accepts CSIP_API_KEY or ADMIN_API_KEY
    (admin is a superset). Used by the Dash app, which only ever holds CSIP_API_KEY."""
    if not CSIP_API_KEY and not ADMIN_API_KEY:
        raise HTTPException(503, "API key not configured on server")
    if not x_api_key:
        raise HTTPException(403, "Invalid API key")
    if CSIP_API_KEY and secrets.compare_digest(x_api_key, CSIP_API_KEY):
        return x_api_key
    if ADMIN_API_KEY and secrets.compare_digest(x_api_key, ADMIN_API_KEY):
        return x_api_key
    raise HTTPException(403, "Invalid API key")


def get_models(request: Request) -> dict:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(503, "Models not loaded yet")
    return request.app.state.models


def _load_all_models() -> dict:
    """Blocking — must run in an executor. Returns a complete dict for atomic state assignment."""
    clf_type_wrap = AdvancedClassifier.load(LGBM_TYPE_PATH)
    clf_pri_wrap = AdvancedClassifier.load(XGB_PRIORITY_PATH)
    reg = AdvancedRegressor.load(RF_REGRESSOR_PATH)
    preprocessor_obj = joblib.load(PREPROCESSOR_PATH)
    preprocessor = (
        preprocessor_obj["preprocessor"]
        if isinstance(preprocessor_obj, dict)
        else preprocessor_obj
    )
    with open(FEATURES_DIR / "tabular_columns.json") as f:
        drift_columns = json.load(f)

    return {
        "preprocessor": preprocessor,
        "tabular_encoder": TabularEncoder.load(TABULAR_ENCODER_PATH),
        "le_type": joblib.load(LE_TYPE_PATH),
        "le_priority": joblib.load(LE_PRIORITY_PATH),
        "clf_type": clf_type_wrap.model,
        "clf_priority": clf_pri_wrap.model,
        "reg": reg,
        "keep_mask": np.load(REGRESSOR_KEEP_MASK_PATH),
        "shap_priority": joblib.load(SHAP_EXPLAINER_PRIORITY_PATH),
        # Section 12 — drift monitoring: cached once at startup since the test split
        # and baseline are static (no point re-reading a 1.3 MB .npy on every call)
        "drift_current": np.load(FEATURES_DIR / "X_test_tabular.npy"),
        "drift_columns": drift_columns,
        "drift_baseline": load_baseline(),
    }


def _verify_models(models: dict) -> None:
    """Smoke-test all models with zero-input to catch a corrupt pkl at startup/reload."""
    dummy17 = np.zeros((1, 17), dtype=np.float64)
    dummy13 = dummy17[:, models["keep_mask"]]
    models["clf_type"].predict_proba(dummy17)
    models["clf_priority"].predict_proba(dummy17)
    models["reg"].predict(dummy13)
