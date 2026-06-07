# filename: api/routers/admin.py
# purpose:  POST /admin/reload — rate-limited, lock-guarded model hot-reload
# version:  1.0

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps import _load_all_models, _verify_models, verify_api_key
from api.metrics import csip_models_loaded
from api.schemas import ReloadResponse
from config import ADMIN_RELOAD_RATE_LIMIT_SECONDS

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
