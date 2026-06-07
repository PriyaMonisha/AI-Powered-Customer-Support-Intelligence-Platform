# filename: api/main.py
# purpose:  FastAPI app — lifespan model loading, router wiring, root endpoint
# version:  1.0

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.deps import _load_all_models, _verify_models
from api.metrics import csip_models_loaded
from api.routers import admin, classify, explain, health, monitoring, regress
from config import API_TITLE, API_VERSION

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_ts = time.time()
    loop = asyncio.get_running_loop()
    models = await loop.run_in_executor(None, _load_all_models)
    await loop.run_in_executor(None, _verify_models, models)

    app.state.models = models
    app.state.ready = True
    app.state.startup_ts = startup_ts
    app.state.last_reload_ts = 0.0
    csip_models_loaded.set(1)
    log.info("CSIP API ready — %d model artifacts loaded", len(models))

    yield

    app.state.ready = False
    csip_models_loaded.set(0)


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

app.include_router(health.router)
app.include_router(monitoring.router)
app.include_router(classify.router)
app.include_router(regress.router)
app.include_router(admin.router)
app.include_router(explain.router)


@app.get("/")
async def root():
    return {"service": "CSIP FastAPI", "version": API_VERSION, "docs": "/docs"}
