# filename: api/routers/admin.py
# purpose:  Admin operations — POST /admin/reload (model hot-reload), POST /admin/drift-check
# version:  1.0

import asyncio
import logging
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps import _load_all_models, _verify_models, verify_api_key
from api.metrics import (
    csip_drift_detected,
    csip_drift_last_check_timestamp,
    csip_feature_drift_psi,
    csip_models_loaded,
)
from api.schemas import DriftCheckResponse, ReloadResponse
from config import ADMIN_RELOAD_RATE_LIMIT_SECONDS
from src.monitoring.drift import check_drift

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")
_reload_lock = asyncio.Lock()


@router.post("/reload", response_model=ReloadResponse)
async def reload(request: Request, _: str = Depends(verify_api_key)):
    async with _reload_lock:
        now = time.time()
        last = request.app.state.last_reload_ts
        if now - last < ADMIN_RELOAD_RATE_LIMIT_SECONDS:
            remaining = int(ADMIN_RELOAD_RATE_LIMIT_SECONDS - (now - last))
            raise HTTPException(
                429,
                detail=f"Retry after {remaining}s",
                headers={"Retry-After": str(remaining)},
            )
        request.app.state.last_reload_ts = now

    # Load OUTSIDE the lock — old models keep serving while new ones load (intentional)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    new_models = await loop.run_in_executor(None, _load_all_models)
    await loop.run_in_executor(None, _verify_models, new_models)
    request.app.state.models = new_models
    csip_models_loaded.set(1)

    return ReloadResponse(
        status="ok",
        reload_time_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def _run_drift_check(X_current: np.ndarray, columns: list[str], baseline: dict) -> dict:
    """Plain function for run_in_executor — builds the DataFrame and runs PSI math off the event loop."""
    return check_drift(pd.DataFrame(X_current, columns=columns), baseline)


@router.post("/drift-check", response_model=DriftCheckResponse)
async def drift_check(request: Request, _: str = Depends(verify_api_key)):
    """
    Computes per-feature PSI drift vs. the Section 6 training baseline.

    Uses the cached test split as a stand-in for "current production data" — this
    portfolio deployment has no live traffic to sample from. Airflow (Section 13) will
    call this on a schedule once real inference logs exist.
    """
    models = request.app.state.models
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _run_drift_check,
        models["drift_current"], models["drift_columns"], models["drift_baseline"],
    )

    # Update Prometheus BEFORE returning — keeps the API response and the next scrape
    # in sync. Otherwise a client could see drift_detected=True while Prometheus still
    # reports 0 until the next interval, making an alert look "late".
    known = set(models["drift_columns"])
    for feature, score in result["feature_scores"].items():
        if feature in known:
            csip_feature_drift_psi.labels(feature=feature).set(score)
        else:
            logger.warning("Drift score for unregistered feature %r — skipping gauge update", feature)
    csip_drift_detected.set(1 if result["drift_detected"] else 0)
    csip_drift_last_check_timestamp.set(time.time())

    return DriftCheckResponse(**result)
