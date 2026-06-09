# filename: api/metrics.py
# purpose:  Standalone Prometheus metric definitions — no FastAPI imports (avoids circular import)
# version:  1.0

from prometheus_client import Counter, Gauge, Histogram

csip_predictions_total = Counter(
    "csip_predictions_total", "Total predictions by task", ["task"]
)

csip_prediction_latency = Histogram(
    "csip_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["task"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

csip_prediction_errors_total = Counter(
    "csip_prediction_errors_total", "Errors by task and error type", ["task", "error_type"]
)

csip_prediction_confidence = Histogram(
    "csip_prediction_confidence",
    "Max softmax probability per prediction",
    ["task"],
    buckets=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

csip_models_loaded = Gauge(
    "csip_models_loaded", "1 if all models loaded and verified, 0 otherwise"
)

# --- Section 12: drift monitoring -------------------------------------------------

csip_feature_drift_psi = Gauge(
    "csip_feature_drift_psi", "Population Stability Index per feature vs. training baseline", ["feature"]
)

csip_drift_detected = Gauge(
    "csip_drift_detected", "1 if any feature's PSI exceeds DRIFT_PSI_THRESHOLD, 0 otherwise"
)

csip_drift_last_check_timestamp = Gauge(
    "csip_drift_last_check_timestamp",
    "Unix timestamp of the last /admin/drift-check run (-1 = never checked)",
)


def register_feature_gauges(feature_names: list[str]) -> None:
    """
    Pre-initializes drift gauges at startup (called once from the lifespan, after
    tabular_columns.json loads).

    Why this matters:
    - Prevents "missing series" gaps in Grafana before the first drift check runs.
    - -1 is a "never checked" sentinel for the timestamp gauge — 0 would render in
      Grafana as "Jan 1 1970", which reads as a bug rather than "not yet run".
    - Pre-registering the exact label set keeps Prometheus's label cardinality bounded
      and known up front (cheap to control at write time, expensive to fix after
      malformed values have been ingested into the TSDB).
    """
    for name in feature_names:
        csip_feature_drift_psi.labels(feature=name).set(0.0)
    csip_drift_detected.set(0)
    csip_drift_last_check_timestamp.set(-1)
