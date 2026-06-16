# Customer Support Intelligence Platform (CSIP)

> **8,469 support tickets · 3 ML tasks · DistilBERT + classical ML · Full MLOps stack**

[![CI](https://github.com/PriyaMonisha/AI-Powered-Customer-Support-Intelligence-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/PriyaMonisha/AI-Powered-Customer-Support-Intelligence-Platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Dash](https://img.shields.io/badge/Plotly%20Dash-2.17-3F4F75?style=flat&logo=plotly&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=flat&logo=mlflow&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE?style=flat&logo=apacheairflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

An end-to-end ML system that automates **ticket type classification**, **priority prediction**,
and **resolution-time estimation** for SaaS/e-commerce customer support operations. Covers the
full ML lifecycle: data cleaning → feature engineering → classical ML + transformer fine-tuning →
serving → monitoring/drift → orchestration → interactive dashboard.

**Engineering highlights:** PSI-based drift detection (Evidently + Prometheus) · SHAP
explainability · MLflow model registry with Production/Challenger aliases · Optuna HPO
(FAST_MODE smoke-test vs. FULL_MODE) · 4 Airflow DAGs (ETL, drift monitor, retrain, model
report) · 55 automated tests · GitHub Actions CI · 10-container Docker Compose stack.

---

## What This Project Does

| Layer | What's Built |
|---|---|
| **Data Pipeline** | 8,469-row Kaggle ticket dataset → 8-step cleaning pipeline → Pandera schema validation → chunked Postgres upsert |
| **Feature Engineering** | `TabularEncoder` (Ordinal + multiclass TargetEncoder), TF-IDF (9,307-term vocab), VADER sentiment + meta-features, Redis feature store (graceful degrade if unavailable) |
| **Classical ML** | RF / XGBoost / LightGBM for Type + Priority + Resolution, tuned via Optuna (`FAST_MODE` smoke-test vs. `FULL_MODE`), `class_weight="balanced"` |
| **Transformer Fine-Tuning** | DistilBERT fine-tuned on subject+description (Google Colab T4, 5 epochs) for Ticket Type |
| **Explainability** | SHAP `TreeExplainer` for the Priority model — top-feature waterfalls, beeswarm plots, and DistilBERT token-level attention heatmaps |
| **Clustering** | K-Means (K=2, silhouette=0.1573, Davies-Bouldin=2.1902) + PCA + t-SNE customer segmentation |
| **Experiment Tracking** | MLflow — 50+ logged runs, 3 registered models with Production/Challenger aliases |
| **Serving** | FastAPI — 7 endpoints, `X-API-Key` auth (admin + read-scoped), `/admin/reload` hot-swap, Prometheus metrics |
| **Monitoring** | Prometheus + Grafana + Alertmanager + Evidently/PSI drift detection, 5 alert rules |
| **Orchestration** | Airflow — 4 DAGs: daily ETL, daily drift monitor, weekly retrain (PSI-gated per task, 3-branch promote/skip/alert), weekly model report |
| **Dashboard** | Plotly Dash — 6 pages, each `def layout()` so bind-mounted artifact edits appear with zero restart |
| **Tests / CI** | 55 pytest tests (auth, dashboard, drift math), GitHub Actions CI, zero skips |
| **Deployment** | `docker-compose.yml` — 10 services across a single bridge network |

---

## Architecture

```
Kaggle Dataset (8,469 tickets)
        │
        ▼
Cleaning + Pandera Validation            Docker Compose (Serving & Ops)
        │                                 ──────────────────────────────────────────────
        ▼                            ┌─────────────────────────────────────────┐
Feature Engineering                  │  Dash (port 8050)                       │
(TF-IDF, TabularEncoder, VADER)      │  6 pages — Live Predictions calls       │
        │                            │  FastAPI via CSIP_API_KEY               │
        ▼                            └────────────────┬────────────────────────┘
Train: RF / XGB / LightGBM                            │ HTTP (X-API-Key)
  + Optuna HPO + DistilBERT (Colab) ┌──────────────────▼──────────────────────┐
        │                           │  FastAPI (port 8000)                    │
        ▼                           │  /predict/{type,priority,resolution}    │
MLflow Registry ───── models ───────│  /explain/priority  /health  /metrics   │
(Production/Challenger aliases)     │  /admin/{reload,drift-check}            │
                                     └───┬─────────────┬─────────────┬────────┘
                                         │             │             │
                                 ┌───────▼───┐  ┌──────▼──────┐ ┌────▼─────┐
                                 │ Postgres  │  │ Prometheus  │ │  Redis   │
                                 │ (csip,    │  │ + alert_    │ │ feature  │
                                 │ airflow)  │  │   rules     │ │  store   │
                                 └─────┬─────┘  └──────┬──────┘ └──────────┘
                                       │               │
                                ┌──────▼──────┐ ┌──────▼───────┐
                                │  Airflow    │ │ Alertmanager │
                                │  4 DAGs     │ │  + Grafana   │
                                └─────────────┘ └──────────────┘
```

---

## Dashboard Pages

| Page | Key Content |
|---|---|
| **0 — Overview** | KPI cards sourced live from `artifacts/reports/model_registry.json` + Section 10 MLflow consolidation notes |
| **1 — EDA Gallery** | 14 EDA charts (class balance, SLA breach rates, sentiment distribution, etc.) |
| **2 — Leaderboard** | Live MLflow leaderboard via `dcc.Interval` + `mlflow_client.get_leaderboard()`, with a "Live (MLflow)" / "Static fallback" badge |
| **3 — Live Predictions** | Submits a ticket to all 4 prediction/explain endpoints in parallel (`ThreadPoolExecutor`); surfaces the Ticket Type `model_status` warning prominently |
| **4 — Drift Monitoring** | 3-state UI: static fallback (Prometheus unreachable) → "no check yet" → live PSI bar chart vs. `DRIFT_PSI_THRESHOLD` |
| **5 — Clustering & Explainability** | K-Means/PCA/t-SNE segmentation charts + SHAP beeswarm/waterfall galleries + DistilBERT attention heatmaps + fairness breakdown by Gender/Age/Channel |

---

## Tech Stack

```
Python 3.11           pandas 2.2.2          scikit-learn 1.5.2
XGBoost 2.0.3         LightGBM 4.3.0        PyTorch 2.3.0 (CPU)
transformers 4.41.2   spaCy 3.7.5           NLTK / VADER
MLflow 2.13.2         Optuna 3.6.1          SHAP 0.45.1
FastAPI 0.111.0       Uvicorn 0.29.0        Pydantic 2.7.1
Dash 2.17.1           dash-bootstrap-components   Plotly 5.22.0
Matplotlib / Seaborn  Pandera 0.20.3        Evidently 0.4.30
Prometheus client     Redis 5.0.4           PostgreSQL (psycopg2)
Apache Airflow 2.9.3  Docker Compose        pytest 8.2.2
```

---

## Project Structure

```
Customer Support Intelligence Platform/
├── api/
│   ├── main.py                # FastAPI app — lifespan model loading, router wiring
│   ├── deps.py                  # X-API-Key auth (admin + read-scoped), model loading
│   ├── schemas.py                 # Pydantic request/response models
│   ├── metrics.py                   # Prometheus metric definitions (no FastAPI imports)
│   └── routers/
│       ├── health.py                  # GET /health
│       ├── monitoring.py               # GET /metrics
│       ├── classify.py                  # POST /predict/type, /predict/priority
│       ├── regress.py                    # POST /predict/resolution
│       ├── explain.py                     # POST /explain/priority (SHAP)
│       └── admin.py                        # POST /admin/reload, /admin/drift-check
├── dash_app/
│   ├── app.py                  # Dash entrypoint — sidebar nav, /charts/<file> route
│   ├── pages/                   # 6 pages (00_overview .. 05_clustering_shap)
│   └── utils/                    # api_client, mlflow_client, data_sources, charts
├── src/
│   ├── data/                     # clean.py, validate.py (Pandera), etl.py
│   ├── features/                  # tabular_features.py, text_features.py,
│   │                                 feature_store.py (Redis), inference.py
│   ├── models/                     # baseline.py, advanced_classifier.py,
│   │                                 advanced_regressor.py, bilstm.py
│   ├── monitoring/                  # drift.py (PSI)
│   └── utils/                        # helpers.py (NumpyEncoder, etc.)
├── notebooks/                    # 01-15 .py scripts — one per pipeline section
├── airflow/dags/                  # csip_etl, csip_drift_monitor, csip_retrain,
│                                     csip_model_report (26 tasks total)
├── monitoring/
│   ├── configs/                     # prometheus.yml, alert_rules.yml, alertmanager.yml
│   └── grafana/provisioning/          # datasource + dashboard JSON
├── docker/
│   ├── airflow/ dash/ mlflow/ postgres/   # per-service Dockerfiles
├── artifacts/
│   ├── reports/                     # model_registry.json, confidence_thresholds.json
│   ├── metrics/                      # per-section metrics JSON (Sections 5-13)
│   ├── charts/ drift/ tmp/             # gitignored — regenerated by notebooks/DAGs
├── models/                        # trained .pkl/.pt artifacts (gitignored)
├── data/                           # raw/ (gitignored), processed/
├── tests/                          # test_api_auth.py, test_dash_app.py, test_drift.py
├── config.py                       # FAST_MODE, RANDOM_STATE=42, paths, thresholds
├── docker-compose.yml              # 10-service orchestration
├── Dockerfile                      # FastAPI image (CPU-only torch)
├── requirements.txt / requirements-ci.txt
└── .github/workflows/ci.yml        # pytest + flake8 on every push
```

---

## Quick Start

### Verify the code — no model weights needed

```bash
git clone https://github.com/PriyaMonisha/AI-Powered-Customer-Support-Intelligence-Platform.git
cd "Customer Support Intelligence Platform"
python -m venv venv
venv\Scripts\activate              # Windows  |  source venv/bin/activate (Linux/Mac)
python -m pip install -r requirements-ci.txt
pytest tests/ -v                   # 55 tests — no model artifacts required
```

### Option A — Full stack via Docker Compose

```bash
cp .env.example .env               # Windows: copy .env.example .env
# Fill in ADMIN_API_KEY, CSIP_API_KEY (any random strings), and DB/Grafana/Airflow
# passwords. AIRFLOW__CORE__FERNET_KEY / WEBSERVER__SECRET_KEY can be generated via
# scripts/generate_env.py.

docker compose up --build -d
```

| Service | URL | Purpose |
|---|---|---|
| Dash dashboard | http://localhost:8050 | Interactive 6-page dashboard |
| FastAPI | http://localhost:8000 | Model inference API (`/docs` for Swagger UI) |
| MLflow | http://localhost:5001 | Experiment tracking + model registry |
| Prometheus | http://localhost:9090 | Metrics collector |
| Alertmanager | http://localhost:9093 | Alert routing |
| Grafana | http://localhost:3000 | Dashboards (`GRAFANA_ADMIN_USER`/`PASSWORD` from `.env`) |
| Airflow | http://localhost:8080 | 4 DAGs (`_AIRFLOW_WWW_USER_*` from `.env`) |
| Postgres | localhost:5433 | `csip` + `airflow` databases |

```bash
docker compose logs -f fastapi     # stream API logs
docker compose ps                  # check service health
docker compose down                # stop all services
```

### Option B — FastAPI only (local development)

```bash
python -m pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Reproduce Training

`notebooks/01_*.py` through `notebooks/13_*.py` run in the terminal (`python notebooks/0X.py`)
with `FAST_MODE = True` in `config.py` for a fast smoke-test, or `False` for the full run.
`notebooks/09_distilbert.py` requires a GPU — convert with
`jupytext --to notebook notebooks/09_distilbert.py` and run on Google Colab T4.

---

## API Endpoints

```bash
# Ticket type classification (LGBM) — note model_status/reliability_note in the response
curl -X POST http://localhost:8000/predict/type \
  -H "X-API-Key: $CSIP_API_KEY" -H "Content-Type: application/json" \
  -d '{"ticket_subject": "Order issue", "ticket_description": "My order arrived damaged"}'

# Priority classification (XGB) + SHAP explanation
curl -X POST http://localhost:8000/predict/priority -H "X-API-Key: $CSIP_API_KEY" ...
curl -X POST http://localhost:8000/explain/priority -H "X-API-Key: $CSIP_API_KEY" ...
```

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | none | Service health, model count, uptime |
| GET | `/metrics` | none | Prometheus metrics (`text/plain`) |
| POST | `/predict/type` | `CSIP_API_KEY` | Ticket Type classification (LightGBM) |
| POST | `/predict/priority` | `CSIP_API_KEY` | Ticket Priority classification (XGBoost) |
| POST | `/predict/resolution` | `CSIP_API_KEY` | Resolution-time regression (Random Forest, hours) |
| POST | `/explain/priority` | `CSIP_API_KEY` | SHAP top-feature explanation for the priority prediction |
| POST | `/admin/reload` | `ADMIN_API_KEY` | Hot-swap reload of all model artifacts (rate-limited) |
| POST | `/admin/drift-check` | `ADMIN_API_KEY` | On-demand PSI drift check vs. training baseline |

`CSIP_API_KEY` is a read-scoped key (accepted alongside the superset `ADMIN_API_KEY`) used by
the Dash app, so the dashboard container never holds admin credentials. API docs at
**http://localhost:8000/docs**.

---

## Tests

```bash
pytest tests/ -v                           # 55 tests
pytest tests/ --cov=api --cov=src          # with coverage
```

| File | Tests | Coverage |
|---|---|---|
| `test_api_auth.py` | 11 | `X-API-Key` auth — admin key, read-scoped key, missing/invalid key on all protected routers |
| `test_dash_app.py` | 28 | Page registry, all 6 `layout()`s render with zero data files (3-tier fallback), `/charts/<file>` traversal safety, api/mlflow client mocks |
| `test_drift.py` | 16 | PSI categorical/continuous math, baseline loading, `check_drift()` |

CI runs on every push via GitHub Actions (`.github/workflows/ci.yml`): installs
`requirements-ci.txt` (full `requirements.txt` minus the torch/transformers/HF stack, which
nothing in `tests/` imports), runs flake8 across all production code, then the full test suite —
**55 passed, 0 skips**.

---

## Production Readiness

| Category | What's Implemented |
|---|---|
| **Security** | PII (`Customer Name`/`Email`) dropped in Step 2 of cleaning, before any model sees it · `X-API-Key` via `secrets.compare_digest` · two-tier keys (`ADMIN_API_KEY` superset, `CSIP_API_KEY` read-only) · no secrets in VCS, `.env.example` documents every required var |
| **Performance** | Models loaded once at startup via `lifespan` + `run_in_executor` (non-blocking event loop) · Redis feature cache with 24h TTL and graceful degrade if unavailable |
| **Reliability** | Startup model loading wrapped in try/except → `/health` reports `"loading"`/503 instead of a crash loop · health checks on all 10 Docker services · 55 automated tests |
| **MLOps** | MLflow registry with Production/Challenger aliases (50+ runs) · Optuna HPO with `FAST_MODE`/`FULL_MODE` split · `/admin/reload` hot-swaps models with a pre-swap smoke test · per-task PSI drift gating before weekly retrains |
| **Observability** | Prometheus counters/histograms/gauges (predictions, latency, errors, confidence, models loaded, drift) · Grafana 6-panel dashboard · Alertmanager with a `slack-notifications` receiver scaffold for critical alerts |
| **Orchestration** | 4 Airflow DAGs — daily ETL, daily drift monitor, weekly retrain (PSI-gated per task, 3-branch promote/skip/alert guard), weekly MLflow leaderboard report |
| **Reproducibility** | `RANDOM_STATE=42` everywhere · atomic (`tempfile`+`shutil.move`) writes for all JSON artifacts · `torch.save(state_dict)` / `weights_only=True` |

---

## Key Design Decisions

- **`/predict/type` ships a `model_status` + `reliability_note` field** — rather than hide a
  model at the statistical noise floor, the API is explicit about its reliability, and the Dash
  app surfaces it as a warning banner. See *Results & Key Findings* below.
- **Redis is optional** — `RedisFeatureStore` degrades gracefully (`except redis.RedisError:
  return None`) and never raises; cache misses fall through to live feature computation.
- **`lifespan` over `@app.on_event`** — model loading runs in `run_in_executor` so the event
  loop stays responsive during startup; loading failures set `app.state.ready = False` instead
  of crashing the container.
- **PSI over KS-test for drift** — the training baseline stores percentile/frequency summaries,
  not raw reference rows, so drift is computed via Population Stability Index
  (`DRIFT_PSI_THRESHOLD = 0.10`) rather than a true KS test.
- **`def layout()` per Dash page** — every page re-reads its data on navigation, so bind-mounted
  artifact edits (e.g. `model_registry.json`) appear with zero container restart.
- **Two-tier API keys** — `CSIP_API_KEY` (read-only: `/predict/*`, `/explain/*`) is a strict
  subset of `ADMIN_API_KEY` (also `/admin/*`), so the Dash container never holds admin
  credentials even though it calls the prediction endpoints directly.
- **`FAST_MODE` convention** — every training notebook starts with `FAST_MODE = True`
  (3-trial Optuna smoke test) for fast iteration; registry numbers were produced under
  `FAST_MODE`, and DistilBERT (Section 9) was the one full (`FAST_MODE=False`) run.
- **Training in Colab, served via classical ML** — DistilBERT was fine-tuned on a T4 GPU but,
  since its test F1 (0.1954) ties the served LightGBM model's val F1 (0.1997), the much
  cheaper/faster classical model remains in `/predict/type` — a deliberate cost/accuracy
  tradeoff, not an oversight.
- **Per-task PSI drift gating in the retrain DAG** — `ShortCircuitOperator` checks PSI for each
  task's feature group (text / tabular / resolution features) independently before triggering
  retrain; fail-open by design (unreadable metrics file → always retrain).

---

## Results & Key Findings

Four independent experiments — TF-IDF term analysis, classical ML (RF/XGB/LightGBM + Optuna
HPO), full DistilBERT fine-tuning (Colab T4, 5 epochs), and majority-class sanity checks — all
converge on the same result for `Ticket Type` and `Resolution Time`: no learnable signal above
the statistical noise floor. The most probable explanation is that these two label columns are
synthetic or near-randomly assigned in the source Kaggle dataset. `Ticket Priority` is the only
task with a measurable signal.

| Task | Best model | Metric | Best result | Dummy baseline |
|---|---|---|---|---|
| Ticket Type (5-class) | LightGBM | val F1-macro | **0.1997** | 0.0685 (most-frequent) |
| Ticket Type (5-class) | DistilBERT (fine-tuned, 5 epochs) | test F1-macro | **0.1954** | — |
| Ticket Priority (4-class) | XGBoost | val F1-macro | **0.2625** | 0.1027 (most-frequent) |
| Resolution Time (regression) | Random Forest | val RMSE / R² | 7.09 hrs / **−0.01 to −0.05** | RMSE 7.06 (mean predictor — marginally better) |

Random chance for balanced 5-class classification is F1-macro ≈ 0.20; for 4-class, ≈ 0.25.
`Ticket Type` real models are statistically tied with random chance, and the resolution-time
models are statistically tied with predicting the training mean. `Ticket Priority` shows a small
but consistent lift above its baseline — SHAP identifies `days_since_purchase` as the dominant
feature.

**Engineering response:** `POST /predict/type` returns `model_status: "below_quality_bar"` and
a `reliability_note` in every response. The Dash "Live Predictions" page surfaces this as a
visible warning banner. The `auto_route` / `flag_for_review` API fields are preserved in the
contract but explicitly flagged — they should not drive automated ticket routing until a model
clears a defensible F1 threshold. This is a deliberate, documented decision.

---

## Dataset

| Item | Value |
|---|---|
| Source | Kaggle Customer Support Ticket Dataset |
| Rows | 8,469 (raw and cleaned) |
| Columns | 17 raw → 21 after cleaning/feature derivation |
| Ticket Type classes (5) | Billing inquiry · Cancellation request · Product inquiry · Refund request · Technical issue |
| Ticket Priority classes (4) | Low · Medium · High · Critical |
| Train / Val / Test split | 5,928 / 847 / 1,694 (70/10/20, stratified) |
| Regression subset (closed tickets) | 1,931 / 276 / 562 |
| TF-IDF vocabulary | 9,307 terms (min_df=2 on train) |

---

## Author

**Priya Monisha** · [GitHub](https://github.com/PriyaMonisha)
