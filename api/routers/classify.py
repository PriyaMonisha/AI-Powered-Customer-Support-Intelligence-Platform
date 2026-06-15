# filename: api/routers/classify.py
# purpose:  POST /predict/type and POST /predict/priority — classifier endpoints
# version:  1.0

import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_models, verify_read_key
from api.metrics import (
    csip_prediction_confidence,
    csip_prediction_errors_total,
    csip_prediction_latency,
    csip_predictions_total,
)
from api.schemas import PredictPriorityResponse, PredictTypeResponse, TicketRequest
from config import AUTO_ROUTE, FLAG_THRESHOLD
from src.features.inference import build_tabular_features

log = logging.getLogger(__name__)
router = APIRouter(prefix="/predict")


@router.post("/type", response_model=PredictTypeResponse)
async def predict_type(
    ticket: TicketRequest,
    models: dict = Depends(get_models),
    _: str = Depends(verify_read_key),
):
    t0 = time.perf_counter()
    try:
        tabular_np = build_tabular_features(ticket, models)
        proba = models["clf_type"].predict_proba(tabular_np)[0]
    except ValueError as e:
        csip_prediction_errors_total.labels(task="type", error_type="value_error").inc()
        raise HTTPException(422, detail=f"Feature extraction failed: {e}")
    except Exception:
        csip_prediction_errors_total.labels(task="type", error_type="unexpected").inc()
        log.exception("Unexpected error in /predict/type")
        raise HTTPException(500, "Internal prediction error")

    class_idx = int(np.argmax(proba))
    label = models["le_type"].classes_[class_idx]
    confidence = float(proba[class_idx])
    elapsed = (time.perf_counter() - t0) * 1000

    csip_predictions_total.labels(task="type").inc()
    csip_prediction_latency.labels(task="type").observe(elapsed / 1000)
    csip_prediction_confidence.labels(task="type").observe(confidence)

    type_classes = models["le_type"].classes_
    return PredictTypeResponse(
        predicted_label=label,
        confidence=round(confidence, 4),
        probabilities={cls: round(float(p), 4) for cls, p in zip(type_classes, proba)},
        auto_route=(confidence >= AUTO_ROUTE),
        flag_for_review=(FLAG_THRESHOLD <= confidence < AUTO_ROUTE),
        model_name="lgbm_type_classifier",
        processing_time_ms=round(elapsed, 2),
    )


@router.post("/priority", response_model=PredictPriorityResponse)
async def predict_priority(
    ticket: TicketRequest,
    models: dict = Depends(get_models),
    _: str = Depends(verify_read_key),
):
    t0 = time.perf_counter()
    try:
        tabular_np = build_tabular_features(ticket, models)
        proba = models["clf_priority"].predict_proba(tabular_np)[0]
    except ValueError as e:
        csip_prediction_errors_total.labels(task="priority", error_type="value_error").inc()
        raise HTTPException(422, detail=f"Feature extraction failed: {e}")
    except Exception:
        csip_prediction_errors_total.labels(task="priority", error_type="unexpected").inc()
        log.exception("Unexpected error in /predict/priority")
        raise HTTPException(500, "Internal prediction error")

    class_idx = int(np.argmax(proba))
    label = models["le_priority"].classes_[class_idx]
    confidence = float(proba[class_idx])
    elapsed = (time.perf_counter() - t0) * 1000

    csip_predictions_total.labels(task="priority").inc()
    csip_prediction_latency.labels(task="priority").observe(elapsed / 1000)
    csip_prediction_confidence.labels(task="priority").observe(confidence)

    priority_classes = models["le_priority"].classes_
    return PredictPriorityResponse(
        predicted_label=label,
        confidence=round(confidence, 4),
        probabilities={cls: round(float(p), 4) for cls, p in zip(priority_classes, proba)},
        model_name="xgb_priority_classifier",
        processing_time_ms=round(elapsed, 2),
    )
