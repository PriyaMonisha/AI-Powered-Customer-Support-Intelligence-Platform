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
