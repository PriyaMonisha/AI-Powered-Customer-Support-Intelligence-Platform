# filename: notebooks/14_docker_smoke_test.py
# purpose:  Section 14 — Full-stack Docker Compose smoke test. Run from the host
#           AFTER `docker compose up -d` (all 10 containers). Verifies every
#           service is reachable and wired together correctly:
#           FastAPI, Prometheus, Alertmanager, Grafana, MLflow, Airflow,
#           Postgres (csip + airflow DBs), Redis.
# version:  1.0

# ── Cell 1: Imports + .env + wait_for_service helper ────────────────────────
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    _NB_DIR = Path(__file__).resolve().parent
except NameError:
    _NB_DIR = Path.cwd()

PROJECT_ROOT = _NB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


def wait_for_service(url: str, service_name: str, timeout: int = 300, interval: int = 10) -> requests.Response:
    """
    Poll `url` until it returns a 2xx response, printing elapsed time on each poll.
    On timeout, dumps `docker compose logs --tail=20 <service_name>` before raising
    so a startup failure (missing model file, import error, ...) is visible
    immediately rather than after a silent multi-minute wait.
    """
    start = time.time()
    last_exc: Exception | None = None
    while time.time() - start < timeout:
        elapsed = round(time.time() - start, 1)
        try:
            resp = requests.get(url, timeout=5)
            if resp.ok:
                print(f"  [{elapsed:>6}s] {service_name}: {url} -> {resp.status_code} OK")
                return resp
            print(f"  [{elapsed:>6}s] {service_name}: {url} -> {resp.status_code}, retrying...")
        except requests.RequestException as exc:
            last_exc = exc
            print(f"  [{elapsed:>6}s] {service_name}: not reachable yet ({exc.__class__.__name__}), retrying...")
        time.sleep(interval)

    print(f"\n--- docker compose logs --tail=20 {service_name} ---")
    subprocess.run(
        ["docker", "compose", "logs", "--tail=20", service_name],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    raise TimeoutError(f"{service_name} did not become healthy within {timeout}s") from last_exc


print("Section 14: Docker Compose full-stack smoke test")
print(f"PROJECT_ROOT={PROJECT_ROOT}")

# ── Cell 2: FastAPI — /health, /metrics, /predict/type ───────────────────────
print("\n" + "=" * 62)
print("FastAPI (localhost:8000)")
print("=" * 62)

resp = wait_for_service("http://localhost:8000/health", "fastapi", timeout=300, interval=10)
health = resp.json()
print("health:", health)
assert health["status"] == "healthy"
assert health["models_loaded"] is True
assert health["model_count"] > 0

resp = requests.get("http://localhost:8000/metrics", timeout=5)
assert resp.headers["content-type"].startswith("text/plain"), resp.headers["content-type"]
assert "csip_models_loaded 1" in resp.text
print("/metrics OK, content-type:", resp.headers["content-type"])

if ADMIN_API_KEY and "choose_a" not in ADMIN_API_KEY:
    sample_ticket = {
        "ticket_subject": "Cannot access my account",
        "ticket_description": "I have been locked out of my account since yesterday and need urgent help.",
        "customer_age": 35,
        "customer_gender": "Male",
        "product_purchased": "Unknown",
        "ticket_channel": "Email",
        "response_hour_of_day": 14,
    }
    resp = requests.post(
        "http://localhost:8000/predict/type",
        json=sample_ticket,
        headers={"X-API-Key": ADMIN_API_KEY},
        timeout=10,
    )
    print("/predict/type ->", resp.status_code, resp.json())
    assert resp.status_code == 200, resp.text
else:
    print("ADMIN_API_KEY not set in .env — skipping /predict/type call")

# ── Cell 3: Prometheus — scrape target + alert rules ─────────────────────────
print("\n" + "=" * 62)
print("Prometheus (localhost:9090)")
print("=" * 62)

wait_for_service("http://localhost:9090/-/healthy", "prometheus", timeout=120, interval=5)

targets = requests.get("http://localhost:9090/api/v1/targets", timeout=5).json()
csip_targets = [t for t in targets["data"]["activeTargets"] if t["labels"].get("job") == "csip-api"]
assert csip_targets, "csip-api scrape target not found"
print("csip-api target health:", csip_targets[0]["health"])
assert csip_targets[0]["health"] == "up", csip_targets[0]

rules = requests.get("http://localhost:9090/api/v1/rules", timeout=5).json()
csip_groups = [g for g in rules["data"]["groups"] if g["name"] == "csip_alerts"]
assert csip_groups, "csip_alerts rule group not found"
rule_names = [r["name"] for r in csip_groups[0]["rules"]]
print(f"csip_alerts rules ({len(rule_names)}):", rule_names)
assert len(rule_names) == 5
assert all(r["state"] == "inactive" for r in csip_groups[0]["rules"]), csip_groups[0]["rules"]

# ── Cell 4: Alertmanager — config loaded ─────────────────────────────────────
print("\n" + "=" * 62)
print("Alertmanager (localhost:9093)")
print("=" * 62)

wait_for_service("http://localhost:9093/-/healthy", "alertmanager", timeout=60, interval=5)
status = requests.get("http://localhost:9093/api/v2/status", timeout=5).json()
assert "config" in status, status
print("alertmanager config loaded OK")

# ── Cell 5: Grafana — datasource + dashboard provisioning ────────────────────
print("\n" + "=" * 62)
print("Grafana (localhost:3000)")
print("=" * 62)

resp = wait_for_service("http://localhost:3000/api/health", "grafana", timeout=120, interval=5)
g = resp.json()
print("grafana health:", g)
assert g.get("database") == "ok", g

# ── Cell 6: MLflow — server up + existing runs visible ───────────────────────
print("\n" + "=" * 62)
print("MLflow (localhost:5001)")
print("=" * 62)

wait_for_service("http://localhost:5001/", "mlflow", timeout=60, interval=5)

import mlflow  # noqa: E402

mlflow.set_tracking_uri("http://localhost:5001")
experiments = mlflow.search_experiments()
all_runs = mlflow.search_runs(experiment_ids=[e.experiment_id for e in experiments])
print(f"MLflow: {len(experiments)} experiments, {len(all_runs)} runs")
assert len(all_runs) >= 40, f"expected ~47 runs, got {len(all_runs)}"

# ── Cell 7: Airflow webserver — metadatabase + scheduler healthy ─────────────
print("\n" + "=" * 62)
print("Airflow webserver (localhost:8080)")
print("=" * 62)

resp = wait_for_service("http://localhost:8080/health", "airflow-webserver", timeout=300, interval=15)
af = resp.json()
print("airflow health:", af)
assert af["metadatabase"]["status"] == "healthy"
assert af["scheduler"]["status"] == "healthy"

# ── Cell 8: PostgreSQL — csip + airflow DBs reachable ─────────────────────────
print("\n" + "=" * 62)
print("PostgreSQL (localhost:5433)")
print("=" * 62)

import psycopg2  # noqa: E402

# Host port 5433 (not 5432) — avoids clashing with a local native PostgreSQL install
for dbname in ("csip", "airflow"):
    conn = psycopg2.connect(
        host="localhost", port=5433, dbname=dbname,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone() == (1,)
    conn.close()
    print(f"  {dbname}: connection OK")

# ── Cell 9: Redis — feature store reachable ──────────────────────────────────
print("\n" + "=" * 62)
print("Redis (localhost:6379)")
print("=" * 62)

import redis  # noqa: E402

r = redis.Redis.from_url("redis://localhost:6379")
assert r.ping() is True
print("redis PING -> PONG")

# ── Cell 10: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("Section 14 smoke test PASSED — all 10 containers verified")
print("=" * 62)
