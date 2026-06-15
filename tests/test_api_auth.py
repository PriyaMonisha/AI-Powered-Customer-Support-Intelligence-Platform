# filename: tests/test_api_auth.py
# purpose:  auth dependencies — verify_api_key (admin-only) and verify_read_key
#           (CSIP_API_KEY or ADMIN_API_KEY, Section 15)
# version:  1.0

import pytest
from fastapi import HTTPException

from api import deps

ADMIN_KEY = "admin-secret-key"
READ_KEY = "csip-read-key"


# --- verify_api_key (admin-only, unchanged) -----------------------------------------

@pytest.mark.asyncio
async def test_verify_api_key_missing_header_returns_403(monkeypatch):
    monkeypatch.setattr(deps, "ADMIN_API_KEY", ADMIN_KEY)
    with pytest.raises(HTTPException) as exc_info:
        await deps.verify_api_key(x_api_key=None)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_api_key_wrong_key_returns_403(monkeypatch):
    monkeypatch.setattr(deps, "ADMIN_API_KEY", ADMIN_KEY)
    with pytest.raises(HTTPException) as exc_info:
        await deps.verify_api_key(x_api_key="wrong-key")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_api_key_correct_admin_key_ok(monkeypatch):
    monkeypatch.setattr(deps, "ADMIN_API_KEY", ADMIN_KEY)
    result = await deps.verify_api_key(x_api_key=ADMIN_KEY)
    assert result == ADMIN_KEY


@pytest.mark.asyncio
async def test_verify_api_key_csip_key_rejected(monkeypatch):
    """CSIP_API_KEY (read-only) must NOT satisfy the admin-only verify_api_key —
    /admin/* requires ADMIN_API_KEY specifically, not the lesser read key."""
    monkeypatch.setattr(deps, "ADMIN_API_KEY", ADMIN_KEY)
    with pytest.raises(HTTPException) as exc_info:
        await deps.verify_api_key(x_api_key=READ_KEY)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_api_key_not_configured_returns_503(monkeypatch):
    monkeypatch.setattr(deps, "ADMIN_API_KEY", "")
    with pytest.raises(HTTPException) as exc_info:
        await deps.verify_api_key(x_api_key=ADMIN_KEY)
    assert exc_info.value.status_code == 503


# --- verify_read_key (CSIP_API_KEY or ADMIN_API_KEY, new in Section 15) -------------

@pytest.mark.asyncio
async def test_verify_read_key_missing_header_returns_403(monkeypatch):
    monkeypatch.setattr(deps, "CSIP_API_KEY", READ_KEY)
    monkeypatch.setattr(deps, "ADMIN_API_KEY", ADMIN_KEY)
    with pytest.raises(HTTPException) as exc_info:
        await deps.verify_read_key(x_api_key=None)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_read_key_wrong_key_returns_403(monkeypatch):
    monkeypatch.setattr(deps, "CSIP_API_KEY", READ_KEY)
    monkeypatch.setattr(deps, "ADMIN_API_KEY", ADMIN_KEY)
    with pytest.raises(HTTPException) as exc_info:
        await deps.verify_read_key(x_api_key="wrong-key")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_read_key_csip_key_ok(monkeypatch):
    monkeypatch.setattr(deps, "CSIP_API_KEY", READ_KEY)
    monkeypatch.setattr(deps, "ADMIN_API_KEY", ADMIN_KEY)
    result = await deps.verify_read_key(x_api_key=READ_KEY)
    assert result == READ_KEY


@pytest.mark.asyncio
async def test_verify_read_key_admin_key_ok_as_superset(monkeypatch):
    monkeypatch.setattr(deps, "CSIP_API_KEY", READ_KEY)
    monkeypatch.setattr(deps, "ADMIN_API_KEY", ADMIN_KEY)
    result = await deps.verify_read_key(x_api_key=ADMIN_KEY)
    assert result == ADMIN_KEY


@pytest.mark.asyncio
async def test_verify_read_key_works_with_only_csip_key_configured(monkeypatch):
    """ADMIN_API_KEY unset (e.g. the Dash container only ever holds CSIP_API_KEY) —
    the read key alone must still be sufficient."""
    monkeypatch.setattr(deps, "CSIP_API_KEY", READ_KEY)
    monkeypatch.setattr(deps, "ADMIN_API_KEY", "")
    result = await deps.verify_read_key(x_api_key=READ_KEY)
    assert result == READ_KEY


@pytest.mark.asyncio
async def test_verify_read_key_both_keys_empty_returns_503(monkeypatch):
    monkeypatch.setattr(deps, "CSIP_API_KEY", "")
    monkeypatch.setattr(deps, "ADMIN_API_KEY", "")
    with pytest.raises(HTTPException) as exc_info:
        await deps.verify_read_key(x_api_key="anything")
    assert exc_info.value.status_code == 503
