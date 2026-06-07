# filename: api/deps.py
# purpose:  FastAPI dependencies — API key auth, model access, and shared model loading
# version:  1.0

import secrets

import joblib
import numpy as np
from fastapi import Header, HTTPException, Request

from config import (
    ADMIN_API_KEY,
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


async def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    if not ADMIN_API_KEY:
        raise HTTPException(503, "API key not configured on server")
    if not x_api_key or not secrets.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(403, "Invalid API key")
    return x_api_key


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
    }


def _verify_models(models: dict) -> None:
    """Smoke-test all models with zero-input to catch a corrupt pkl at startup/reload."""
    dummy17 = np.zeros((1, 17), dtype=np.float64)
    dummy13 = dummy17[:, models["keep_mask"]]
    models["clf_type"].predict_proba(dummy17)
    models["clf_priority"].predict_proba(dummy17)
    models["reg"].predict(dummy13)
