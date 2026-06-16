# filename: notebooks/phase_c_01_dummy_baselines.py
# purpose:  Phase C — Dummy/majority-class baseline comparison.
#           Logs DummyClassifier + DummyRegressor to MLflow so every leaderboard
#           has an explicit naive lower bound alongside the real models.
# version:  1.0

FAST_MODE = True  # kept for project consistency; all ops here are < 1 second

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from config import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

# Use local mlruns/ — Docker MLflow server may not be running during terminal runs
_MLRUNS_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(_MLRUNS_DIR.as_uri())
logger.info("MLflow tracking URI: %s", _MLRUNS_DIR.as_uri())

# ── Cell 1: Load pre-split feature arrays ────────────────────────────────────
FEAT_DIR = PROJECT_ROOT / "data" / "processed" / "features"
METRICS_OUT = PROJECT_ROOT / "artifacts" / "metrics" / "phase_c_baseline_metrics.json"

X_train = np.load(FEAT_DIR / "X_train_tabular.npy")
X_val   = np.load(FEAT_DIR / "X_val_tabular.npy")
X_test  = np.load(FEAT_DIR / "X_test_tabular.npy")

y_train_type = np.load(FEAT_DIR / "y_train_type.npy")
y_val_type   = np.load(FEAT_DIR / "y_val_type.npy")
y_test_type  = np.load(FEAT_DIR / "y_test_type.npy")

y_train_prio = np.load(FEAT_DIR / "y_train_prio.npy")
y_val_prio   = np.load(FEAT_DIR / "y_val_prio.npy")
y_test_prio  = np.load(FEAT_DIR / "y_test_prio.npy")

y_train_reg  = np.load(FEAT_DIR / "y_train_reg.npy")
y_val_reg    = np.load(FEAT_DIR / "y_val_reg.npy")
y_test_reg   = np.load(FEAT_DIR / "y_test_reg.npy")

label_maps = json.load(open(FEAT_DIR / "label_maps.json"))

logger.info(
    "Splits loaded — train: %d  val: %d  test: %d",
    len(y_train_type), len(y_val_type), len(y_test_type),
)


# ── Cell 2: Metric helpers matching existing MLflow schema exactly ────────────
def _clf_metrics(prefix: str, y_true, y_pred, y_prob=None) -> dict:
    m = {
        f"{prefix}_f1_macro":    round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 6),
        f"{prefix}_f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        f"{prefix}_accuracy":    round(accuracy_score(y_true, y_pred), 6),
        f"{prefix}_roc_auc":     None,
    }
    if y_prob is not None:
        try:
            m[f"{prefix}_roc_auc"] = round(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"), 6
            )
        except ValueError:
            pass  # leave None
    return m


def _reg_metrics(prefix: str, y_true, y_pred) -> dict:
    y_pred_c = np.clip(y_pred, 0, None)
    y_true_c = np.clip(y_true, 0, None)
    rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_c) - np.log1p(y_true_c)) ** 2)))
    # MAPE: add small epsilon to true values to avoid divide-by-zero on any zero-hour tickets
    mape = float(mean_absolute_percentage_error(y_true + 1e-6, y_pred))
    return {
        f"{prefix}_rmse":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        f"{prefix}_mae":   round(float(mean_absolute_error(y_true, y_pred)), 6),
        f"{prefix}_r2":    round(float(r2_score(y_true, y_pred)), 6),
        f"{prefix}_mape":  round(mape, 6),
        f"{prefix}_rmsle": round(rmsle, 6),
    }


def _mlflow_log(experiment: str, run_name: str, params: dict, metrics: dict) -> str:
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        # MLflow cannot log None — filter out null metrics
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
    return run.info.run_id


# ── Cell 3: Ticket Type — DummyClassifier(strategy='most_frequent') ───────────
logger.info("=== Ticket Type Dummy ===")
type_clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
type_clf.fit(X_train, y_train_type)

# Document what the classifier learned
majority_label = int(type_clf.classes_[np.argmax(type_clf.class_prior_)])
majority_name  = label_maps["ticket_type"][str(majority_label)]
majority_freq  = float(np.mean(y_train_type == majority_label))
logger.info(
    "Majority class: %s (label %d)  %.1f%% of train set",
    majority_name, majority_label, majority_freq * 100,
)
logger.info(
    "NOTE: DummyClassifier predicts '%s' for EVERY sample. "
    "F1-macro is near zero for all other classes by design — "
    "this is the naive lower bound, not a modelling failure.",
    majority_name,
)

val_type_m  = _clf_metrics("val",  y_val_type,  type_clf.predict(X_val),
                            type_clf.predict_proba(X_val))
test_type_m = _clf_metrics("test", y_test_type, type_clf.predict(X_test),
                            type_clf.predict_proba(X_test))
type_metrics = {**val_type_m, **test_type_m}
logger.info(
    "Type dummy  val_f1_macro=%.4f  test_f1_macro=%.4f  val_roc_auc=%s",
    type_metrics["val_f1_macro"], type_metrics["test_f1_macro"],
    f"{type_metrics['val_roc_auc']:.4f}" if type_metrics["val_roc_auc"] else "n/a",
)

type_run_id = _mlflow_log(
    "csip-baseline-classifiers",
    "dummy_ticket_type",
    {
        "task": "ticket_type", "algorithm": "DummyClassifier",
        "strategy": "most_frequent", "majority_class": majority_name,
        "majority_class_freq": f"{majority_freq:.6f}",
        "class_weight": "n/a", "fast_mode": str(FAST_MODE),
    },
    type_metrics,
)
logger.info("MLflow run ID: %s", type_run_id)


# ── Cell 4: Ticket Priority — DummyClassifier(strategy='most_frequent') ────────
logger.info("=== Ticket Priority Dummy ===")
prio_clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
prio_clf.fit(X_train, y_train_prio)

maj_prio_label = int(prio_clf.classes_[np.argmax(prio_clf.class_prior_)])
maj_prio_name  = label_maps["ticket_priority"][str(maj_prio_label)]
maj_prio_freq  = float(np.mean(y_train_prio == maj_prio_label))
logger.info(
    "Majority class: %s (label %d)  %.1f%% of train set",
    maj_prio_name, maj_prio_label, maj_prio_freq * 100,
)

val_prio_m  = _clf_metrics("val",  y_val_prio,  prio_clf.predict(X_val),
                            prio_clf.predict_proba(X_val))
test_prio_m = _clf_metrics("test", y_test_prio, prio_clf.predict(X_test),
                            prio_clf.predict_proba(X_test))
prio_metrics = {**val_prio_m, **test_prio_m}
logger.info(
    "Priority dummy  val_f1_macro=%.4f  test_f1_macro=%.4f",
    prio_metrics["val_f1_macro"], prio_metrics["test_f1_macro"],
)

prio_run_id = _mlflow_log(
    "csip-baseline-classifiers",
    "dummy_ticket_priority",
    {
        "task": "ticket_priority", "algorithm": "DummyClassifier",
        "strategy": "most_frequent", "majority_class": maj_prio_name,
        "majority_class_freq": f"{maj_prio_freq:.6f}",
        "class_weight": "n/a", "fast_mode": str(FAST_MODE),
    },
    prio_metrics,
)
logger.info("MLflow run ID: %s", prio_run_id)


# ── Cell 5: Resolution Time — DummyRegressor(strategy='mean') ─────────────────
logger.info("=== Resolution Time Dummy ===")
# Regression uses only closed tickets — filter NaN targets (open tickets have NaN hours_to_resolve)
train_mask = ~np.isnan(y_train_reg)
val_mask   = ~np.isnan(y_val_reg)
test_mask  = ~np.isnan(y_test_reg)
logger.info(
    "Regression rows (closed tickets): train=%d  val=%d  test=%d",
    train_mask.sum(), val_mask.sum(), test_mask.sum(),
)

reg_dummy = DummyRegressor(strategy="mean")
reg_dummy.fit(X_train[train_mask], y_train_reg[train_mask])

train_mean_hours = float(reg_dummy.constant_[0])
logger.info(
    "Train mean hours_to_resolve = %.2f h  "
    "(DummyRegressor predicts this for EVERY sample; R2=0 by definition)",
    train_mean_hours,
)

val_reg_m  = _reg_metrics("val",  y_val_reg[val_mask],   reg_dummy.predict(X_val[val_mask]))
test_reg_m = _reg_metrics("test", y_test_reg[test_mask], reg_dummy.predict(X_test[test_mask]))
reg_metrics = {**val_reg_m, **test_reg_m}
logger.info(
    "Regressor dummy  val_rmse=%.4f  val_r2=%.4f  val_mae=%.4f",
    reg_metrics["val_rmse"], reg_metrics["val_r2"], reg_metrics["val_mae"],
)

mlflow.set_experiment("csip-regression-models")
with mlflow.start_run(run_name="dummy_resolution") as reg_run:
    mlflow.log_params({
        "task": "resolution", "algorithm": "DummyRegressor",
        "strategy": "mean", "train_mean_hours": f"{train_mean_hours:.6f}",
        "fast_mode": str(FAST_MODE),
    })
    mlflow.log_metrics(reg_metrics)
reg_run_id = reg_run.info.run_id
logger.info("MLflow run ID: %s", reg_run_id)


# ── Cell 6: Write JSON artifact ───────────────────────────────────────────────
output = {
    "section": "phase_c_01",
    "description": (
        "Dummy/majority-class baselines — naive lower bound for each task. "
        "DummyClassifier(strategy='most_frequent') predicts one class for ALL samples; "
        "F1-macro is near zero for all non-majority classes by design. "
        "DummyRegressor(strategy='mean') always predicts the training mean; R2=0 by definition."
    ),
    "generated_at": datetime.now().isoformat(),
    "ticket_type": {
        "algorithm": "DummyClassifier(strategy=most_frequent)",
        "majority_class": majority_name,
        "majority_class_freq": round(majority_freq, 6),
        **{k: round(v, 6) for k, v in type_metrics.items() if v is not None},
        "mlflow_run_id": type_run_id,
        "mlflow_experiment": "csip-baseline-classifiers",
    },
    "ticket_priority": {
        "algorithm": "DummyClassifier(strategy=most_frequent)",
        "majority_class": maj_prio_name,
        "majority_class_freq": round(maj_prio_freq, 6),
        **{k: round(v, 6) for k, v in prio_metrics.items() if v is not None},
        "mlflow_run_id": prio_run_id,
        "mlflow_experiment": "csip-baseline-classifiers",
    },
    "resolution_time": {
        "algorithm": "DummyRegressor(strategy=mean)",
        "train_mean_hours": round(train_mean_hours, 6),
        **reg_metrics,
        "mlflow_run_id": reg_run_id,
        "mlflow_experiment": "csip-regression-models",
    },
}

METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = METRICS_OUT.with_suffix(".json.tmp")
tmp.write_text(json.dumps(output, indent=2))
shutil.move(str(tmp), str(METRICS_OUT))
logger.info("Written: %s", METRICS_OUT)


# ── Cell 7: Summary ───────────────────────────────────────────────────────────
print("\n=== DUMMY BASELINE SUMMARY ===")
print(f"{'Task':<22} {'Val F1-macro':>14} {'Test F1-macro':>14} {'Val RMSE':>10}")
print("-" * 64)
print(f"{'Ticket Type (5-class)':<22} {type_metrics['val_f1_macro']:14.4f} "
      f"{type_metrics['test_f1_macro']:14.4f} {'n/a':>10}")
print(f"{'Ticket Priority (4-class)':<22} {prio_metrics['val_f1_macro']:14.4f} "
      f"{prio_metrics['test_f1_macro']:14.4f} {'n/a':>10}")
print(f"{'Resolution Time':<22} {'n/a':>14} {'n/a':>14} {reg_metrics['val_rmse']:10.4f}")
print("-" * 64)
print(f"\nType majority class      : '{majority_name}' ({majority_freq:.1%} of train)")
print(f"Priority majority class  : '{maj_prio_name}' ({maj_prio_freq:.1%} of train)")
print(f"Resolution training mean : {train_mean_hours:.1f} hours")
print(f"\nMLflow experiment csip-baseline-classifiers: 2 new runs")
print(f"MLflow experiment csip-regression-models   : 1 new run")
print(f"Artifact: {METRICS_OUT}")
