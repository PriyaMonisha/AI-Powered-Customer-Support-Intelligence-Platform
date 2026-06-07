# filename: api/routers/health.py
# purpose:  Public health check endpoint
# version:  1.0

import time

from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    ready = getattr(request.app.state, "ready", False)
    model_count = len(request.app.state.models) if ready else 0
    startup_ts = getattr(request.app.state, "startup_ts", time.time())
    return HealthResponse(
        status="healthy" if ready else "loading",
        models_loaded=ready,
        model_count=model_count,
        uptime_seconds=round(time.time() - startup_ts, 2),
    )
