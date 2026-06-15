# filename: dash_app/utils/api_client.py
# purpose:  Dash -> FastAPI/Prometheus HTTP client. Every call returns (data, error) —
#           never raises into a callback.
# version:  1.0

import logging
import os

import requests

from config import ADMIN_API_KEY, CSIP_API_KEY

log = logging.getLogger(__name__)

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
TIMEOUT = (2, 5)  # (connect, read) seconds

_session = requests.Session()


def _post(path: str, payload: dict) -> tuple[dict | None, str | None]:
    url = f"{FASTAPI_URL}{path}"
    try:
        resp = _session.post(
            url, json=payload, headers={"X-API-Key": CSIP_API_KEY}, timeout=TIMEOUT
        )
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except ValueError:
            detail = str(e)
        log.warning("POST %s -> %s", url, detail)
        return None, str(detail)
    except requests.exceptions.RequestException as e:
        log.warning("POST %s failed: %s", url, e)
        return None, str(e)


def predict_type(ticket: dict) -> tuple[dict | None, str | None]:
    return _post("/predict/type", ticket)


def predict_priority(ticket: dict) -> tuple[dict | None, str | None]:
    return _post("/predict/priority", ticket)


def predict_resolution(ticket: dict) -> tuple[dict | None, str | None]:
    return _post("/predict/resolution", ticket)


def explain_priority(ticket: dict) -> tuple[dict | None, str | None]:
    return _post("/explain/priority", ticket)


def get_health() -> tuple[dict | None, str | None]:
    url = f"{FASTAPI_URL}/health"
    try:
        resp = _session.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        log.warning("GET %s failed: %s", url, e)
        return None, str(e)


def trigger_drift_check() -> tuple[dict | None, str | None]:
    """POST /admin/drift-check using ADMIN_API_KEY — only called when
    DASH_ADMIN_ENABLED=true (the Dash container does not receive ADMIN_API_KEY
    by default, so this returns an auth error in the standard deployment)."""
    url = f"{FASTAPI_URL}/admin/drift-check"
    try:
        resp = _session.post(url, headers={"X-API-Key": ADMIN_API_KEY}, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except ValueError:
            detail = str(e)
        log.warning("POST %s -> %s", url, detail)
        return None, str(detail)
    except requests.exceptions.RequestException as e:
        log.warning("POST %s failed: %s", url, e)
        return None, str(e)


def query_prometheus(promql: str) -> tuple[dict | None, str | None]:
    url = f"{PROMETHEUS_URL}/api/v1/query"
    try:
        resp = _session.get(url, params={"query": promql}, timeout=TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            return None, body.get("error", "Prometheus query failed")
        return body["data"], None
    except requests.exceptions.RequestException as e:
        log.warning("GET %s failed: %s", url, e)
        return None, str(e)
