# filename: src/features/feature_store.py
# purpose:  Redis feature store — key hashing, connection pooling, read/write/pipeline.
#           Class-based design: no module-level global primitives — safe for pytest
#           session reuse, Airflow worker reuse, and uvicorn hot reload.
# version:  1.0

import hashlib
import json
import logging
from typing import Any, Optional

import redis as _redis

from config import REDIS_KEY_PREFIX, REDIS_FEATURE_TTL, REDIS_URL
from src.utils.helpers import NumpyEncoder

logger = logging.getLogger(__name__)


def build_redis_key(ticket_id: "str | int") -> str:
    """SHA-256 hash -> 16-char hex prefix. Prevents sequential key enumeration."""
    h = hashlib.sha256(str(ticket_id).encode()).hexdigest()[:16]
    return f"{REDIS_KEY_PREFIX}:{h}"


class RedisFeatureStore:
    """
    Redis feature store with connection pooling.

    Testable: tests create RedisFeatureStore(url="redis://mock") — no shared global.
    Airflow: each DAG run creates its own instance (no stale pool from prior run).
    FastAPI: call store.close() in lifespan shutdown to release connections cleanly.
    """

    def __init__(self, url: str = REDIS_URL, max_connections: int = 10) -> None:
        self._url = url
        self._max_connections = max_connections
        self._pool: Optional[_redis.ConnectionPool] = None
        self._pool_failed = False          # tracks permanent failure — prevents log spam

    # ── Pool and client ───────────────────────────────────────────────────────

    def get_pool(self) -> Optional[_redis.ConnectionPool]:
        if self._pool_failed:
            return None                    # already failed permanently — no retry
        if self._pool is None:
            try:
                self._pool = _redis.ConnectionPool.from_url(
                    self._url,
                    max_connections=self._max_connections,
                    socket_connect_timeout=2,
                )
                logger.info(
                    "Redis pool created: %s (max_connections=%d)",
                    self._url, self._max_connections,
                )
            except Exception as e:
                logger.warning(
                    "Redis pool creation failed permanently: %s. "
                    "All feature store calls will be no-ops.",
                    e,
                )
                self._pool_failed = True
        return self._pool

    def get_client(
        self,
        redis_client: Optional[_redis.Redis] = None,
    ) -> Optional[_redis.Redis]:
        """Returns provided client or creates one from pool. Returns None if unavailable."""
        if redis_client is not None:
            return redis_client
        pool = self.get_pool()
        if pool is None:
            return None
        try:
            return _redis.Redis(connection_pool=pool)
        except Exception as e:
            logger.warning("Redis client creation failed: %s", e)
            return None

    def close(self) -> None:
        """Release pool connections. Call in FastAPI lifespan shutdown."""
        if self._pool is not None:
            try:
                self._pool.disconnect()
            except Exception:
                pass
            self._pool = None

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        ticket_id: "str | int",
        features: dict,
        redis_client: Optional[_redis.Redis] = None,
    ) -> bool:
        """Write features for one ticket. Returns True on success, False on any error."""
        client = self.get_client(redis_client)
        if client is None:
            return False
        try:
            payload = json.dumps(features, cls=NumpyEncoder)
        except (TypeError, ValueError) as e:
            logger.error(
                "Serialization failed for ticket %s: %s. "
                "Add unhandled type to NumpyEncoder.",
                ticket_id, e,
            )
            return False
        try:
            client.setex(build_redis_key(ticket_id), REDIS_FEATURE_TTL, payload)
            return True
        except _redis.RedisError as e:
            logger.debug("Redis SETEX failed for %s: %s", ticket_id, e)
            return False

    def queue(
        self,
        pipe: Any,
        ticket_id: "str | int",
        features: dict,
    ) -> None:
        """
        Queue a SETEX onto an existing pipeline. Caller must call pipe.execute().
        pipe: redis.client.Pipeline — typed as Any for runtime compat across redis-py versions.
        NumpyEncoder handles numpy.float64 from TargetEncoder — no TypeError.
        Serialization errors logged at ERROR; entry silently skipped (pipeline continues).
        """
        try:
            payload = json.dumps(features, cls=NumpyEncoder)
        except (TypeError, ValueError) as e:
            logger.error(
                "Serialization failed for ticket %s: %s. "
                "Add unhandled type to NumpyEncoder.",
                ticket_id, e,
            )
            return
        pipe.setex(build_redis_key(ticket_id), REDIS_FEATURE_TTL, payload)

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(
        self,
        ticket_id: "str | int",
        redis_client: Optional[_redis.Redis] = None,
    ) -> Optional[dict]:
        """
        Returns feature dict on cache hit, None on cache miss or connection error.
        Distinguishes Redis connection errors (warning) from data corruption (error + self-heal).
        """
        client = self.get_client(redis_client)
        if client is None:
            return None

        try:
            val = client.get(build_redis_key(ticket_id))
        except _redis.RedisError as e:
            logger.warning("Redis GET failed for %s: %s", ticket_id, e)
            return None

        if val is None:
            return None                    # normal cache miss

        try:
            return json.loads(val)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Corrupted value = write path bug, not a connection issue
            key = build_redis_key(ticket_id)
            logger.error("Corrupted cache at %s: %s. Deleting corrupt entry.", key, e)
            try:
                client.delete(key)         # self-heal — remove corrupt entry
            except _redis.RedisError:
                pass                       # delete failed; TTL will expire naturally
            return None


# ── Module-level default instance + backward-compatible functions ─────────────
# Tests override: import src.features.feature_store as fs; fs._default_store = RedisFeatureStore(url="redis://mock")

_default_store = RedisFeatureStore()


def read_ticket_features(
    ticket_id: "str | int",
    redis_client: Optional[_redis.Redis] = None,
) -> Optional[dict]:
    return _default_store.read(ticket_id, redis_client)


def write_ticket_features(
    ticket_id: "str | int",
    features: dict,
    redis_client: Optional[_redis.Redis] = None,
) -> bool:
    return _default_store.write(ticket_id, features, redis_client)


def queue_ticket_features(
    pipe: Any,
    ticket_id: "str | int",
    features: dict,
) -> None:
    _default_store.queue(pipe, ticket_id, features)


def _get_client(
    redis_client: Optional[_redis.Redis] = None,
) -> Optional[_redis.Redis]:
    """Module-level wrapper — notebook Cell 8 uses this to share the default pool with FastAPI."""
    return _default_store.get_client(redis_client)
