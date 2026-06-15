# filename: api/routers/explain.py
# purpose:  POST /explain/priority — SHAP-based explanation for the priority classifier
# version:  1.0

import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_models, verify_read_key
from api.metrics import csip_prediction_errors_total, csip_prediction_latency, csip_predictions_total
from api.schemas import ExplainPriorityResponse, ShapFeature, TicketRequest
from src.features.inference import build_tabular_features
from src.features.tabular_features import NUMERIC_PASSTHROUGH

log = logging.getLogger(__name__)
router = APIRouter(prefix="/explain")


@router.post("/priority", response_model=ExplainPriorityResponse)
async def explain_priority(
    ticket: TicketRequest,
    models: dict = Depends(get_models),
    _: str = Depends(verify_read_key),
):
    t0 = time.perf_counter()
    try:
        tabular_np = build_tabular_features(ticket, models)
        proba = models["clf_priority"].predict_proba(tabular_np)[0]
        class_idx = int(np.argmax(proba))
        label = models["le_priority"].classes_[class_idx]

        shap_raw = models["shap_priority"].shap_values(tabular_np)
        # SHAP 0.45.1 returns (N, F, C) ndarray for LGBM multi-class; older versions return a list
        shap_arr = np.stack(shap_raw, axis=2) if isinstance(shap_raw, list) else shap_raw
        shap_row = shap_arr[0, :, class_idx]

        feature_names = _feature_names(models)
        pairs = sorted(zip(feature_names, shap_row), key=lambda p: abs(p[1]), reverse=True)
        top_features = [
            ShapFeature(feature=name, shap_value=round(float(val), 6))
            for name, val in pairs
            if abs(val) > 0.001
        ][:5]
    except ValueError as e:
        csip_prediction_errors_total.labels(task="explain_priority", error_type="value_error").inc()
        raise HTTPException(422, detail=f"Feature extraction failed: {e}")
    except Exception:
        csip_prediction_errors_total.labels(task="explain_priority", error_type="unexpected").inc()
        log.exception("Unexpected error in /explain/priority")
        raise HTTPException(500, "Internal prediction error")

    elapsed = (time.perf_counter() - t0) * 1000
    csip_predictions_total.labels(task="explain_priority").inc()
    csip_prediction_latency.labels(task="explain_priority").observe(elapsed / 1000)

    return ExplainPriorityResponse(
        predicted_label=label,
        top_features=top_features,
        model_name="xgb_priority_classifier",
        processing_time_ms=round(elapsed, 2),
    )


def _feature_names(models: dict) -> list[str]:
    """Column names matching TabularEncoder.transform() output order (17 cols)."""
    enc = models["tabular_encoder"]
    ordinal_names = [f"{c}_enc" for c in enc.ordinal_enc.feature_names_in_]
    target_names = list(enc.target_enc.get_feature_names_out())
    return ordinal_names + target_names + list(NUMERIC_PASSTHROUGH)
