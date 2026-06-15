# CSIP Runbook — Prometheus Alert Rules

This runbook covers the 5 alert rules defined in
[monitoring/configs/alert_rules.yml](monitoring/configs/alert_rules.yml) (group `csip_alerts`),
evaluated by Prometheus (http://localhost:9090/alerts) and routed by Alertmanager
(http://localhost:9093) per [monitoring/configs/alertmanager.yml](monitoring/configs/alertmanager.yml).

## Alert routing

| Severity | Receiver | Repeat interval |
|---|---|---|
| `critical` | `slack-notifications` (placeholder webhook — see alertmanager.yml header) | 1h |
| `warning` / `info` | `default` (Alertmanager UI only, not forwarded) | 4h |

Only `ModelsNotLoaded` is `critical`. The other four are `warning`/`info` and are visible in
the Alertmanager UI but not forwarded anywhere until a real Slack webhook is configured.

Airflow's `csip_retrain` and `csip_drift_monitor` DAGs have their own, independent
notification path via `CSIP_ALERT_WEBHOOK` (`_send_notification_stub` in
[airflow/dags/csip_retrain.py](airflow/dags/csip_retrain.py) and
[airflow/dags/csip_drift_monitor.py](airflow/dags/csip_drift_monitor.py)) — set that env var in
`.env` to receive DAG-level alerts independent of Prometheus/Alertmanager.

---

## 1. HighPredictionErrorRate

**Severity:** `warning` · **For:** 2m

**Meaning:** Over the last 5 minutes, more than 5% of `/predict/*` or `/explain/*` requests for
a given `task` (`type`, `priority`, `resolution`, `explain_priority`) resulted in an error,
recorded via `csip_prediction_errors_total`. Guarded so it only fires when that task has
nonzero traffic (`sum by (task) (rate(csip_predictions_total[5m])) > 0`).

**Diagnosis:**
1. Identify the failing task from the `task` label on the alert.
2. Break down by error type: `sum by (task, error_type) (rate(csip_prediction_errors_total[5m]))`
   in Prometheus (http://localhost:9090/graph).
3. Check FastAPI logs for stack traces: `docker compose logs --tail=200 fastapi`.
4. Common causes: malformed request payloads (validation errors — usually client-side, not
   actionable here), a stale/corrupt model artifact after a bad `/admin/reload`, or a
   downstream dependency (Redis/Postgres) timing out inside the feature pipeline.

**Remediation:**
- If caused by a bad `/admin/reload`: re-run `POST /admin/reload` with `ADMIN_API_KEY` — it
  smoke-tests new models with `dummy17`/`dummy13` before swapping, so a second reload from
  good artifacts on disk should restore service.
- If caused by a dependency outage (Redis/Postgres down): `docker compose ps` to find the
  unhealthy container, then `docker compose up -d <service>`. Recall from Section 14: a
  recreated `postgres` container can leave `airflow-scheduler` in a crash loop — restart it
  manually if needed (`docker compose up -d airflow-scheduler`).
- If error_type is consistently a validation/4xx error from one client: this is a client-side
  integration bug, not a service incident — no API-side action needed.

**Escalation:** If the error rate persists after a model reload and dependency check, treat as
a code regression — check `git log` for recent changes to `api/routers/<task>.py` or
`src/features/inference.py` and consider rolling back the FastAPI image
(`docker compose build fastapi` from a prior commit).

---

## 2. HighPredictionLatencyP95

**Severity:** `warning` · **For:** 2m

**Meaning:** The 95th-percentile latency for a `task` (from `csip_prediction_latency_seconds`)
has exceeded 0.5s — the project's documented `< 500ms` API latency target — sustained over 5
minutes, again only when that task has nonzero traffic.

**Diagnosis:**
1. Identify the slow `task` from the alert label.
2. Check `docker stats` (or `docker compose top fastapi`) for CPU/memory pressure on the
   `fastapi` container — `mem_limit: 3g` in `docker-compose.yml` can be exhausted if multiple
   models + SHAP explainer are loaded under concurrent load.
3. For `task="explain_priority"`: SHAP `TreeExplainer` calls are inherently the slowest of the
   4 endpoints — check whether the alert is isolated to this task before treating it as a
   regression.
4. Check Redis health (`docker compose exec redis redis-cli ping`) — if the feature store is
   down, every request falls through to live feature computation (slower, but still
   functionally correct per the graceful-degrade design).

**Remediation:**
- Transient spike under load: usually self-resolves; confirm via the Grafana dashboard
  (http://localhost:3000) latency panel that p95 is trending back down.
- Sustained: restart the `fastapi` container (`docker compose restart fastapi`) to clear any
  memory fragmentation from repeated `/admin/reload` calls, then re-check.
- If Redis is down and explains the slowdown: bring it back with
  `docker compose up -d redis` — no code change needed, the feature store reconnects
  automatically on the next request.

**Escalation:** If p95 stays above 500ms with healthy CPU/memory and Redis up, this likely
needs a code-level profiling pass on `src/features/inference.py` (the per-request feature
pipeline: `build_inference_row` → `add_text_meta_features` → `TabularEncoder.transform`) —
file as a follow-up, not an emergency.

---

## 3. FeatureDriftDetected

**Severity:** `warning` · **For:** 0m (fires immediately)

**Meaning:** `csip_drift_detected == 1` — on the last `/admin/drift-check` run (manual, or via
the daily `csip_drift_monitor` Airflow DAG at 03:00 UTC), at least one of the 17 tabular
features had a PSI ≥ `DRIFT_PSI_THRESHOLD = 0.10` against the Section 6 training baseline
(`artifacts/drift/baseline.json`). Fires with no sustain window because this gauge only
changes on an explicit, infrequent check — there's nothing to "flap".

**Diagnosis:**
1. Open the Dash "Drift Monitoring" page (http://localhost:8050/drift) for a live PSI bar
   chart per feature, or query `csip_feature_drift_psi` by `feature` label directly in
   Prometheus.
2. Identify which feature(s) exceeded 0.10 and by how much — PSI between 0.10–0.25 is
   "moderate" drift, > 0.25 is "major".
3. Cross-reference with recent data: has there been a real distributional shift in production
   tickets (e.g., a marketing campaign changing `days_since_purchase` or
   `product_purchased` mix), or is this a data-quality issue (nulls, a broken upstream feed)?

**Remediation:**
- If the drift is a genuine, expected distribution shift: this is informational — the
  `csip_retrain` DAG (weekly, Sundays 02:00 UTC) will retrain on fresh data and its 3-branch
  guard (promote/skip/alert) will decide whether to promote new models based on evaluation
  metrics, not drift alone.
- If the drift indicates a data-quality bug (e.g., a feature suddenly all-null or
  out-of-range): fix the upstream ETL (`airflow/dags/csip_etl.py`, `src/data/clean.py`) before
  the next retrain — retraining on bad data would only encode the bug into the next model.
- To re-baseline after a confirmed, accepted shift: regenerate
  `artifacts/drift/baseline.json` (the `regen_baseline` task in `csip_retrain` does this
  automatically on a successful promotion).

**Escalation:** Given `Ticket Type`/`Resolution Time` are already at the noise floor (see
[README.md — Known Limitations](README.md#️-known-limitations-read-this-first)), drift on
those targets' features is unlikely to change served-model quality meaningfully. Drift on
`Ticket Priority` features (the one task with real signal, XGB val F1=0.27863) is the
higher-priority case to investigate.

---

## 4. ModelsNotLoaded

**Severity:** `critical` · **For:** 2m

**Meaning:** `csip_models_loaded == 0` for over 2 minutes — the FastAPI service is up (health
checks pass at the container level) but `app.state.ready` is `False`, so all `/predict/*` and
`/explain/*` endpoints will return 503. Since the [api/main.py](api/main.py) `lifespan`
rewrite, a failed model load no longer crashes the container — it starts in this "not ready"
state instead.

**Diagnosis:**
1. `curl http://localhost:8000/health` — confirm `status: "loading"` (or check for a 503).
2. `docker compose logs --tail=100 fastapi` — the `lifespan` function logs
   `"CSIP API startup failed — model loading/verification raised"` with a full traceback on
   failure.
3. Common causes: a model artifact under `models/` is missing, corrupted, or fails the
   `_verify_models()` smoke test (`dummy17`/`dummy13` inputs) in
   [api/deps.py](api/deps.py); or `models/` bind mount isn't populated (fresh clone without
   running the training notebooks).

**Remediation:**
- Missing/corrupt artifact: re-run the relevant training notebook
  (`notebooks/05_baseline.py` through `notebooks/07_regression.py`, or
  `notebooks/10_mlflow.py` for registry-sourced artifacts) to regenerate it under `models/`.
- Bind-mount issue (Docker): confirm `./models` exists and is populated on the host, then
  `docker compose restart fastapi` — the `lifespan` re-runs `_load_all_models()` on container
  start.
- After fixing artifacts on disk without a restart: `POST /admin/reload` with
  `ADMIN_API_KEY` re-runs the same load + verify + atomic swap path.

**Escalation:** This is the only `critical`-severity alert (routed to `slack-notifications`,
1h repeat) — it means the API is effectively down for all prediction traffic. If artifacts are
present and verified-good but loading still fails, check for a dependency version mismatch
(e.g., a `scikit-learn`/`joblib` version drift between training and serving environments —
compare `requirements.txt` pins against what produced the `.pkl` files).

---

## 5. DriftCheckStale

**Severity:** `info` · **For:** 1m

**Meaning:** `csip_drift_last_check_timestamp` is more than 24 hours old (and not the `-1`
"never checked since startup" sentinel, which is deliberately excluded so a freshly-booted API
doesn't immediately alert). This means neither a manual `POST /admin/drift-check` nor the daily
`csip_drift_monitor` Airflow DAG (03:00 UTC) has run successfully in over a day.

**Diagnosis:**
1. Check the Airflow UI (http://localhost:8080) for `csip_drift_monitor` — is it paused, or
   has the latest run failed?
2. `docker compose ps airflow-scheduler` — confirm the scheduler container is healthy (recall
   the Section 14 finding: a recreated `postgres` container can leave the scheduler in a crash
   loop requiring a manual restart).
3. If the DAG looks fine, check whether `/admin/drift-check` itself is erroring —
   `docker compose logs fastapi | grep drift`.

**Remediation:**
- Paused DAG: unpause `csip_drift_monitor` in the Airflow UI (or
  `docker compose exec airflow-webserver airflow dags unpause csip_drift_monitor`).
- Crashed scheduler: `docker compose up -d airflow-scheduler`.
- For an immediate refresh without waiting for the next schedule: `POST /admin/drift-check`
  with `ADMIN_API_KEY` (also gated behind `DASH_ADMIN_ENABLED` as a manual button on the Dash
  Drift page).

**Escalation:** Informational only — does not indicate a serving outage. If it persists for
multiple days alongside `csip_etl` also failing, treat as an Airflow infrastructure issue
(check Postgres `airflow` database health) rather than a drift-monitoring-specific bug.
