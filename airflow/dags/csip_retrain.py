# filename: airflow/dags/csip_retrain.py
# purpose:  DAG 3 — Weekly model retraining with 3-branch promotion guard.
#           Trains LGBM type classifier, XGB priority classifier, RF regressor.
#           Branch: promote (F1 +0.02) | skip (within bounds) | alert (F1 -0.05).
#           Drift baseline regenerated atomically after successful promotion.
# version:  1.0

import logging
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.exceptions import AirflowSkipException  # module-level: cross_project_ml.md rule
from airflow.models import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Config constants imported at module level — config.py is stdlib-only, safe to import early
from config import PROMOTE_THRESHOLD, REGRESSION_THRESHOLD  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "csip-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

_TMP_DIR = _PROJECT_ROOT / "artifacts" / "tmp"

# ---------------------------------------------------------------------------
# Notification stub — shared by alert_regression
# ---------------------------------------------------------------------------

def _send_notification_stub(subject: str, body: str) -> None:
    """Send Slack notification if CSIP_ALERT_WEBHOOK is set; otherwise log only."""
    webhook = os.getenv("CSIP_ALERT_WEBHOOK", "")
    if webhook:
        import json as _json
        payload = _json.dumps({"text": f"*{subject}*\n{body}"}).encode()
        req = urllib.request.Request(
            webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        logger.info("Alert sent to webhook: %s", subject)
    else:
        logger.warning("No CSIP_ALERT_WEBHOOK configured — alert logged only: %s", subject)


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _load_training_data(**context) -> None:
    """
    Clean up stale temp retrain artifacts (C-4 fix), then load feature arrays
    and champion F1 scores from model_registry.json for comparison.
    """
    import json
    import numpy as np
    from config import FEATURES_DIR, MODEL_REGISTRY_PATH

    # C-4: clear stale temp models from any previous failed run
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in _TMP_DIR.glob("*_retrain.pkl"):
        stale.unlink()
        logger.info("Cleared stale temp artifact: %s", stale.name)

    # Verify feature arrays exist
    for split in ("train", "val", "test"):
        p = FEATURES_DIR / f"X_{split}_tabular.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing feature array: {p}. Run csip_etl first.")

    # Load champion metrics from model registry
    if not MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Model registry not found: {MODEL_REGISTRY_PATH}")
    with open(str(MODEL_REGISTRY_PATH)) as fh:
        registry: dict = json.load(fh)

    champion_type_f1    = float(registry["ticket_type"]["val_f1_macro"])
    champion_priority_f1 = float(registry["ticket_priority"]["val_f1_macro"])
    mean_champion_f1    = float((champion_type_f1 + champion_priority_f1) / 2)

    context["ti"].xcom_push(key="champion_metrics", value={
        "type_f1":          champion_type_f1,
        "priority_f1":      champion_priority_f1,
        "mean_champion_f1": mean_champion_f1,
    })
    logger.info(
        "Champion metrics loaded: type_f1=%.4f priority_f1=%.4f mean=%.4f",
        champion_type_f1, champion_priority_f1, mean_champion_f1,
    )


def _retrain_type_clf(**context) -> None:
    """
    Optuna LightGBM training for Ticket Type classification.
    In-memory study (storage=None) avoids SQLite lock contention with parallel tasks.
    Saves temp artifact to artifacts/tmp/lgbm_type_retrain.pkl.
    """
    import numpy as np
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from config import FEATURES_DIR, FAST_MODE, FAST_N_TRIALS, FAST_CV_FOLDS, FULL_N_TRIALS, FULL_CV_FOLDS, RANDOM_STATE
    from src.models.advanced_classifier import AdvancedClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train = np.load(str(FEATURES_DIR / "X_train_tabular.npy"))
    y_train = np.load(str(FEATURES_DIR / "y_train_type.npy"))
    X_val   = np.load(str(FEATURES_DIR / "X_val_tabular.npy"))
    y_val   = np.load(str(FEATURES_DIR / "y_val_type.npy"))

    n_trials = FAST_N_TRIALS if FAST_MODE else FULL_N_TRIALS
    n_folds  = FAST_CV_FOLDS if FAST_MODE else FULL_CV_FOLDS

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 200),
            "max_depth":       trial.suggest_int("max_depth", 4, 10),
            "num_leaves":      trial.suggest_int("num_leaves", 10, 50),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "class_weight":    "balanced",
            "random_state":    RANDOM_STATE,
            "n_jobs":          -1,
            "verbose":         -1,
        }
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        for tr_idx, vl_idx in kf.split(X_train, y_train):
            clf = LGBMClassifier(**params)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            pred = clf.predict(X_train[vl_idx])
            fold_scores.append(f1_score(y_train[vl_idx], pred, average="macro", zero_division=0))
            trial.report(float(np.mean(fold_scores)), len(fold_scores) - 1)
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_type_retrain",
        storage=None,  # in-memory — no SQLite contention with parallel tasks
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**study.best_params, "class_weight": "balanced",
                   "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    clf = AdvancedClassifier(
        model_name="lgbm",
        task="ticket_type_retrain",
        model=model,
        feature_schema="tabular_only",
        n_features_in=X_train.shape[1],
        best_params=best_params,
    )
    clf.save(_TMP_DIR / "lgbm_type_retrain.pkl")

    val_f1 = float(f1_score(y_val, model.predict(X_val), average="macro", zero_division=0))
    context["ti"].xcom_push(key="type_clf_val_f1", value=val_f1)
    logger.info("lgbm_type_retrain: val_f1_macro=%.4f (best_trial=%.4f)", val_f1, study.best_value)


def _retrain_priority_clf(**context) -> None:
    """
    Optuna XGBoost training for Ticket Priority classification.
    In-memory study. Saves temp artifact to artifacts/tmp/xgb_priority_retrain.pkl.
    """
    import numpy as np
    import optuna
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    from config import FEATURES_DIR, FAST_MODE, FAST_N_TRIALS, FAST_CV_FOLDS, FULL_N_TRIALS, FULL_CV_FOLDS, RANDOM_STATE
    from src.models.advanced_classifier import AdvancedClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train = np.load(str(FEATURES_DIR / "X_train_tabular.npy"))
    y_train = np.load(str(FEATURES_DIR / "y_train_prio.npy"))
    X_val   = np.load(str(FEATURES_DIR / "X_val_tabular.npy"))
    y_val   = np.load(str(FEATURES_DIR / "y_val_prio.npy"))

    n_trials = FAST_N_TRIALS if FAST_MODE else FULL_N_TRIALS
    n_folds  = FAST_CV_FOLDS if FAST_MODE else FULL_CV_FOLDS

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 200),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric":       "mlogloss",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "verbosity":         0,
        }
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        for tr_idx, vl_idx in kf.split(X_train, y_train):
            sw = compute_sample_weight("balanced", y_train[tr_idx])
            clf = XGBClassifier(**params)
            clf.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=sw)
            pred = clf.predict(X_train[vl_idx])
            fold_scores.append(f1_score(y_train[vl_idx], pred, average="macro", zero_division=0))
            trial.report(float(np.mean(fold_scores)), len(fold_scores) - 1)
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_priority_retrain",
        storage=None,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**study.best_params, "eval_metric": "mlogloss",
                   "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0}
    sw_full = compute_sample_weight("balanced", y_train)
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, sample_weight=sw_full)

    clf = AdvancedClassifier(
        model_name="xgb",
        task="ticket_priority_retrain",
        model=model,
        feature_schema="tabular_only",
        n_features_in=X_train.shape[1],
        best_params=best_params,
    )
    clf.save(_TMP_DIR / "xgb_priority_retrain.pkl")

    val_f1 = float(f1_score(y_val, model.predict(X_val), average="macro", zero_division=0))
    context["ti"].xcom_push(key="priority_clf_val_f1", value=val_f1)
    logger.info("xgb_priority_retrain: val_f1_macro=%.4f (best_trial=%.4f)", val_f1, study.best_value)


def _retrain_regressor(**context) -> None:
    """
    Optuna RandomForest training for hours_to_resolve regression.
    Operates on closed tickets only (non-NaN y_train_reg).
    Applies regressor_keep_mask.npy to reduce from 17 to 13 features.
    In-memory study. Saves temp artifact to artifacts/tmp/rf_regressor_retrain.pkl.
    """
    import numpy as np
    import optuna
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    from config import FEATURES_DIR, MODELS_DIR, FAST_MODE, FAST_N_TRIALS, FAST_CV_FOLDS, FULL_N_TRIALS, FULL_CV_FOLDS, RANDOM_STATE, REGRESSOR_KEEP_MASK_PATH
    from src.models.advanced_regressor import AdvancedRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train_full = np.load(str(FEATURES_DIR / "X_train_tabular.npy"))
    y_train_raw  = np.load(str(FEATURES_DIR / "y_train_reg.npy"))
    X_val_full   = np.load(str(FEATURES_DIR / "X_val_tabular.npy"))
    y_val_raw    = np.load(str(FEATURES_DIR / "y_val_reg.npy"))

    # Apply feature keep mask (17 → 13 regression features)
    keep_mask = np.load(str(REGRESSOR_KEEP_MASK_PATH))
    X_train_13 = X_train_full[:, keep_mask]
    X_val_13   = X_val_full[:, keep_mask]

    # Closed tickets only
    train_mask = ~np.isnan(y_train_raw)
    val_mask   = ~np.isnan(y_val_raw)
    X_reg   = X_train_13[train_mask]
    y_reg   = y_train_raw[train_mask]
    X_val_r = X_val_13[val_mask]
    y_val_r = y_val_raw[val_mask]

    n_trials = FAST_N_TRIALS if FAST_MODE else FULL_N_TRIALS
    n_folds  = FAST_CV_FOLDS if FAST_MODE else FULL_CV_FOLDS

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 50, 200),
            "max_depth":      trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":   trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state":   RANDOM_STATE,
            "n_jobs":         -1,
        }
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        for tr_idx, vl_idx in kf.split(X_reg):
            reg = RandomForestRegressor(**params)
            reg.fit(X_reg[tr_idx], y_reg[tr_idx])
            pred = reg.predict(X_reg[vl_idx])
            rmse = float(np.sqrt(mean_squared_error(y_reg[vl_idx], pred)))
            fold_scores.append(rmse)
            trial.report(float(np.mean(fold_scores)), len(fold_scores) - 1)
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        study_name="rf_regressor_retrain",
        storage=None,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**study.best_params, "random_state": RANDOM_STATE, "n_jobs": -1}
    model = RandomForestRegressor(**best_params)
    model.fit(X_reg, y_reg)

    reg = AdvancedRegressor(
        model_name="rf",
        task="hours_to_resolve_retrain",
        model=model,
        feature_schema="tabular_only_13",
        n_features_in=X_reg.shape[1],
        best_params=best_params,
        log_transform=False,
    )
    reg.save(_TMP_DIR / "rf_regressor_retrain.pkl")

    val_rmse = float(np.sqrt(mean_squared_error(y_val_r, model.predict(X_val_r))))
    context["ti"].xcom_push(key="regressor_val_rmse", value=val_rmse)
    logger.info("rf_regressor_retrain: val_rmse=%.4f (best_trial=%.4f)", val_rmse, study.best_value)


def _evaluate_new_models(**context) -> None:
    """
    Aggregate val metrics from the three parallel retrain tasks.
    mean_new_f1 = (type_f1 + priority_f1) / 2.
    Regressor RMSE tracked separately — does not gate promotion (monitored via DAG 4).
    All XCom values cast to float() to prevent np.float64 JSON serialization errors.
    """
    ti = context["ti"]

    type_f1      = ti.xcom_pull(task_ids="retrain_type_clf",      key="type_clf_val_f1")
    priority_f1  = ti.xcom_pull(task_ids="retrain_priority_clf",  key="priority_clf_val_f1")
    regressor_rmse = ti.xcom_pull(task_ids="retrain_regressor",   key="regressor_val_rmse")
    champion_m   = ti.xcom_pull(task_ids="load_training_data",    key="champion_metrics")

    if None in (type_f1, priority_f1, regressor_rmse, champion_m):
        missing = [
            n for n, v in [("type_f1", type_f1), ("priority_f1", priority_f1),
                            ("regressor_rmse", regressor_rmse), ("champion_m", champion_m)]
            if v is None
        ]
        raise ValueError(
            f"XCom missing for: {missing} — task(s) may have failed. Check task logs."
        )

    mean_new_f1 = float((float(type_f1) + float(priority_f1)) / 2)

    eval_metrics = {
        "mean_new_f1":      mean_new_f1,
        "mean_champion_f1": float(champion_m["mean_champion_f1"]),
        "type_f1":          float(type_f1),
        "priority_f1":      float(priority_f1),
        "regressor_rmse":   float(regressor_rmse),
    }
    ti.xcom_push(key="eval_metrics", value=eval_metrics)

    delta = mean_new_f1 - float(champion_m["mean_champion_f1"])
    logger.info(
        "Evaluation: mean_new_f1=%.4f mean_champion_f1=%.4f delta=%.4f",
        mean_new_f1, float(champion_m["mean_champion_f1"]), delta,
    )


def _decide_branch(**context) -> str:
    """
    3-branch guard: compares mean_new_f1 against champion_f1.
    Branch thresholds (from config.py):
      PROMOTE_THRESHOLD    = 0.02  (+2% improvement → promote)
      REGRESSION_THRESHOLD = 0.05  (-5% regression → alert)
    """
    metrics = context["ti"].xcom_pull(task_ids="evaluate_new_models", key="eval_metrics")
    if metrics is None:
        raise ValueError(
            "evaluate_new_models XCom is None — task may have failed. Check task logs."
        )

    delta = metrics["mean_new_f1"] - metrics["mean_champion_f1"]
    logger.info(
        "Promotion decision: delta=%.4f | PROMOTE_THRESHOLD=%.2f | REGRESSION_THRESHOLD=%.2f",
        delta, PROMOTE_THRESHOLD, REGRESSION_THRESHOLD,
    )

    if delta >= PROMOTE_THRESHOLD:
        logger.info("Decision: PROMOTE (delta=%.4f >= %.2f)", delta, PROMOTE_THRESHOLD)
        return "promote_models"
    elif delta >= -REGRESSION_THRESHOLD:
        logger.info(
            "Decision: SKIP (delta=%.4f within [%.2f, %.2f))",
            delta, -REGRESSION_THRESHOLD, PROMOTE_THRESHOLD,
        )
        return "skip_retrain"
    else:
        logger.info("Decision: ALERT (delta=%.4f < -%.2f)", delta, REGRESSION_THRESHOLD)
        return "alert_regression"


def _promote_models(**context) -> None:
    """
    Promote new models to production paths using shutil.move (cross-device safe — C-1).
    Pre-flight: all 3 temp PKLs must exist before touching any production path (C-4 fix).
    Registry: backup → atomic write (C-5 fix).
    Post-promote: assert TABULAR_ENCODER_PATH exists (CE-4 fix).
    """
    import json
    import shutil
    from config import (
        LGBM_TYPE_PATH, XGB_PRIORITY_PATH, RF_REGRESSOR_PATH,
        MODEL_REGISTRY_PATH, TABULAR_ENCODER_PATH,
    )

    promotions = [
        (_TMP_DIR / "lgbm_type_retrain.pkl",    LGBM_TYPE_PATH),
        (_TMP_DIR / "xgb_priority_retrain.pkl", XGB_PRIORITY_PATH),
        (_TMP_DIR / "rf_regressor_retrain.pkl", RF_REGRESSOR_PATH),
    ]

    # Pre-flight: verify ALL temp files exist before touching any production path
    missing = [str(src) for src, _ in promotions if not src.exists()]
    if missing:
        raise FileNotFoundError(
            f"Temp model(s) missing — aborting promotion. "
            f"Production models untouched. Missing: {missing}"
        )

    # Promote: shutil.move handles cross-device moves (Docker volume → host path)
    for src, dst in promotions:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        logger.info("Promoted: %s → %s", src.name, dst)

    # CE-4: assert TABULAR_ENCODER_PATH exists (unchanged by this promotion, but verify)
    if not Path(str(TABULAR_ENCODER_PATH)).exists():
        raise RuntimeError(
            f"TABULAR_ENCODER_PATH missing after promotion: {TABULAR_ENCODER_PATH}. "
            "Downstream regenerate_drift_baseline would use stale encoder."
        )

    # C-5: backup + atomic registry write
    metrics = context["ti"].xcom_pull(task_ids="evaluate_new_models", key="eval_metrics")
    backup_path = MODEL_REGISTRY_PATH.with_suffix(".json.bak")
    shutil.copy2(str(MODEL_REGISTRY_PATH), str(backup_path))

    with open(str(MODEL_REGISTRY_PATH)) as fh:
        registry: dict = json.load(fh)

    registry["ticket_type"]["val_f1_macro"]     = round(metrics["type_f1"], 6)
    registry["ticket_type"]["algo"]             = "lgbm"
    registry["ticket_priority"]["val_f1_macro"] = round(metrics["priority_f1"], 6)
    registry["ticket_priority"]["algo"]         = "xgb"
    registry["last_retrain"]                    = context["ds"]

    tmp_path = MODEL_REGISTRY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(registry, indent=2))
    shutil.move(str(tmp_path), str(MODEL_REGISTRY_PATH))
    logger.info(
        "Models promoted and registry updated: type_f1=%.4f priority_f1=%.4f",
        metrics["type_f1"], metrics["priority_f1"],
    )


def _regenerate_drift_baseline(**context) -> None:
    """
    Regenerate PSI drift baseline from X_train_tabular.npy using the existing
    production encoder. Runs AFTER promote_models (trigger_rule=ALL_SUCCESS).

    Uses pre-encoded X_train_tabular.npy directly — no re-encoding needed since
    TabularEncoder is not refitted during retraining (refit occurs in the training
    pipeline only, not in this DAG).
    Atomic write: .json.tmp → shutil.move.
    """
    import json
    import shutil
    from datetime import datetime, timezone

    import numpy as np
    from config import DRIFT_BASELINE_PATH, DRIFT_DIR, FEATURES_DIR

    X_train = np.load(str(FEATURES_DIR / "X_train_tabular.npy"))
    with open(str(FEATURES_DIR / "tabular_columns.json")) as fh:
        columns: list = json.load(fh)

    n_train = len(X_train)
    stats: dict = {}

    for i, col in enumerate(columns):
        vals = X_train[:, i]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue

        feat_stats = {
            "mean": round(float(np.mean(vals)), 6),
            "std":  round(float(np.std(vals)), 6),
            "min":  round(float(np.min(vals)), 6),
            "max":  round(float(np.max(vals)), 6),
            "p25":  round(float(np.percentile(vals, 25)), 6),
            "p50":  round(float(np.percentile(vals, 50)), 6),
            "p75":  round(float(np.percentile(vals, 75)), 6),
        }

        # Categorical heuristic: ≤ 25 unique values (ordinal-encoded + binary features)
        unique_vals = np.unique(vals)
        if len(unique_vals) <= 25:
            freq = {
                str(float(v)): round(float(np.sum(vals == v) / len(vals)), 6)
                for v in unique_vals
            }
            feat_stats["value_frequencies"] = freq

        stats[col] = feat_stats

    baseline = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "n_train":        n_train,
        "feature_names":  columns,
        "stats":          stats,
    }

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DRIFT_BASELINE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(baseline, indent=2))
    shutil.move(str(tmp_path), str(DRIFT_BASELINE_PATH))
    logger.info(
        "Drift baseline regenerated: %d features, %d train samples → %s",
        len(columns), n_train, DRIFT_BASELINE_PATH,
    )


def _skip_retrain(**context) -> None:
    """
    Log skip reason and return normally (SUCCESS — not AirflowSkipException).
    BranchPythonOperator already marks unselected branches as Skipped; raising
    AirflowSkipException here would cause confusing double-skip state in the UI.
    """
    metrics = context["ti"].xcom_pull(task_ids="evaluate_new_models", key="eval_metrics")
    if metrics:
        delta = metrics["mean_new_f1"] - metrics["mean_champion_f1"]
        logger.info(
            "Retrain SKIPPED: delta=%.4f is within acceptable range [%.2f, +%.2f). "
            "Current champion models retained.",
            delta, -REGRESSION_THRESHOLD, PROMOTE_THRESHOLD,
        )
    else:
        logger.info("Retrain SKIPPED. Champion models retained.")


def _alert_regression(**context) -> None:
    """Log CRITICAL F1 regression and send notification."""
    metrics = context["ti"].xcom_pull(task_ids="evaluate_new_models", key="eval_metrics")
    if metrics:
        delta = metrics["mean_new_f1"] - metrics["mean_champion_f1"]
        logger.critical(
            "MODEL REGRESSION DETECTED | new_f1=%.4f champion_f1=%.4f delta=%.4f "
            "(<= -%.2f threshold). New models NOT promoted.",
            metrics["mean_new_f1"], metrics["mean_champion_f1"],
            delta, REGRESSION_THRESHOLD,
        )
        _send_notification_stub(
            subject="[CSIP] Model Regression Detected",
            body=(
                f"New mean_f1={metrics['mean_new_f1']:.4f} vs "
                f"champion={metrics['mean_champion_f1']:.4f} | delta={delta:.4f}\n"
                f"Type: {metrics['type_f1']:.4f} | Priority: {metrics['priority_f1']:.4f}"
            ),
        )
    else:
        logger.critical("MODEL REGRESSION DETECTED — evaluate_new_models XCom unavailable.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="csip_retrain",
    default_args=_DEFAULT_ARGS,
    description="Weekly model retraining with 3-branch guard: promote | skip | alert",
    schedule_interval="0 2 * * 0",    # 02:00 UTC every Sunday
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,                 # prevent concurrent promotions/registry writes
    tags=["csip", "training", "mlops"],
) as dag:

    load_training_data = PythonOperator(
        task_id="load_training_data",
        python_callable=_load_training_data,
    )

    retrain_type_clf = PythonOperator(
        task_id="retrain_type_clf",
        python_callable=_retrain_type_clf,
    )

    retrain_priority_clf = PythonOperator(
        task_id="retrain_priority_clf",
        python_callable=_retrain_priority_clf,
    )

    retrain_regressor = PythonOperator(
        task_id="retrain_regressor",
        python_callable=_retrain_regressor,
    )

    evaluate_new_models = PythonOperator(
        task_id="evaluate_new_models",
        python_callable=_evaluate_new_models,
    )

    promotion_branch = BranchPythonOperator(
        task_id="promotion_branch",
        python_callable=_decide_branch,
    )

    promote_models = PythonOperator(
        task_id="promote_models",
        python_callable=_promote_models,
    )

    regenerate_drift_baseline = PythonOperator(
        task_id="regenerate_drift_baseline",
        python_callable=_regenerate_drift_baseline,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # explicit: only if promote_models succeeded
    )

    skip_retrain = PythonOperator(
        task_id="skip_retrain",
        python_callable=_skip_retrain,
    )

    alert_regression = PythonOperator(
        task_id="alert_regression",
        python_callable=_alert_regression,
    )

    # Task dependency graph
    load_training_data >> [retrain_type_clf, retrain_priority_clf, retrain_regressor]
    [retrain_type_clf, retrain_priority_clf, retrain_regressor] >> evaluate_new_models
    evaluate_new_models >> promotion_branch
    promotion_branch >> [promote_models, skip_retrain, alert_regression]
    promote_models >> regenerate_drift_baseline
