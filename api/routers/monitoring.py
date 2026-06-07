# filename: api/routers/monitoring.py
# purpose:  Prometheus scrape endpoint — text/plain exposition format
# version:  1.0

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
