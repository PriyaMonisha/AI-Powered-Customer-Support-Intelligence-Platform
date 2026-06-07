# filename: api/routers/regress.py
# purpose:  POST /predict/resolution — resolution-time regressor endpoint
# version:  1.0

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_models, verify_api_key
from api.metrics import (
    csip_prediction_errors_total,
    csip_prediction_latency,
    csip_predictions_total,
)
from api.schemas import PredictResolutionResponse, TicketRequest
from src.features.inference import build_tabular_features

log = logging.getLogger(__name__)
router = APIRouter(prefix="/predict")


@router.post("/resolution", response_model=PredictResolutionResponse)
async def predict_resolution(
    ticket: TicketRequest,
    models: dict = Depends(get_models),
    _: str = Depends(verify_api_key),
):
    t0 = time.perf_counter()
    try:
        tabular_np = build_tabular_features(ticket, models)
        masked = tabular_np[:, models["keep_mask"]]
        hours = float(models["reg"].predict(masked)[0])
    except ValueError as e:
        csip_prediction_errors_total.labels(task="resolution", error_type="value_error").inc()
        raise HTTPException(422, detail=f"Feature extraction failed: {e}")
    except Exception:
        csip_prediction_errors_total.labels(task="resolution", error_type="unexpected").inc()
        log.exception("Unexpected error in /predict/resolution")
        raise HTTPException(500, "Internal prediction error")

    elapsed = (time.perf_counter() - t0) * 1000
    csip_predictions_total.labels(task="resolution").inc()
    csip_prediction_latency.labels(task="resolution").observe(elapsed / 1000)

    return PredictResolutionResponse(
        predicted_hours=round(hours, 2),
        model_name="rf_regressor",
        processing_time_ms=round(elapsed, 2),
        warning="Regressor trained on closed tickets only",
    )
