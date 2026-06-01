# filename: notebooks/06_advanced_ml.py
# purpose:  Section 6 — RF, XGBoost, LightGBM + Optuna + confidence thresholds + drift baseline
# version:  1.0

# %% Cell 1 — FAST_MODE (must be first line of code)
FAST_MODE = True

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-trial output

N_TRIALS = 3   if FAST_MODE else 25   # FAST_MODE: smoke test only (below MedianPruner minimum)
N_FOLDS  = 3   if FAST_MODE else 5

# %% Cell 2 — Imports + PROJECT_ROOT
import datetime
import gc
import json
import logging
import sys
import warnings
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    CONFIDENCE_THRESHOLDS_PATH,
    DRIFT_BASELINE_PATH,
    FEATURES_DIR,
    FAST_MODE as CFG_FAST_MODE,
    LE_PRIORITY_PATH,
    LE_TYPE_PATH,
    LGBM_PRIORITY_PATH,
    LGBM_TYPE_PATH,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    RF_PRIORITY_PATH,
    RF_TYPE_PATH,
    XGB_PRIORITY_PATH,
    XGB_TYPE_PATH,
)
from src.models.advanced_classifier import AdvancedClassifier
from src.models.baseline import plot_confusion_matrix
from src.utils.helpers import NumpyEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("section6")
logger.info("FAST_MODE=%s  N_TRIALS=%d  N_FOLDS=%d", FAST_MODE, N_TRIALS, N_FOLDS)

# %% Cell 3 — Load feature arrays + metadata
logger.info("Loading feature artifacts from %s ...", FEATURES_DIR)

X_tab_train = np.load(FEATURES_DIR / "X_train_tabular.npy")
X_tab_val   = np.load(FEATURES_DIR / "X_val_tabular.npy")
X_tab_test  = np.load(FEATURES_DIR / "X_test_tabular.npy")

y_train_type = np.load(FEATURES_DIR / "y_train_type.npy")
y_val_type   = np.load(FEATURES_DIR / "y_val_type.npy")
y_test_type  = np.load(FEATURES_DIR / "y_test_type.npy")

y_train_prio = np.load(FEATURES_DIR / "y_train_prio.npy")
y_val_prio   = np.load(FEATURES_DIR / "y_val_prio.npy")
y_test_prio  = np.load(FEATURES_DIR / "y_test_prio.npy")

# Feature column names — load from S4 artifact (never reconstruct from encoder internals)
with open(FEATURES_DIR / "tabular_columns.json") as fh:
    tab_col_names: list[str] = json.load(fh)

# Label encoders — load for class_names only (not for training)
le_type = joblib.load(LE_TYPE_PATH)
le_prio = joblib.load(LE_PRIORITY_PATH)
assert isinstance(le_type, LabelEncoder) and hasattr(le_type, "classes_")
assert isinstance(le_prio, LabelEncoder) and hasattr(le_prio, "classes_")
type_class_names = [str(c) for c in le_type.classes_]
prio_class_names = [str(c) for c in le_prio.classes_]

# %% Cell 4 — Validate shapes
for split, xt, yt_t, yt_p in [
    ("train", X_tab_train, y_train_type, y_train_prio),
    ("val",   X_tab_val,   y_val_type,   y_val_prio),
    ("test",  X_tab_test,  y_test_type,  y_test_prio),
]:
    assert xt.shape[0] == yt_t.shape[0] == yt_p.shape[0], \
        f"{split}: row count mismatch X={xt.shape[0]} y_type={yt_t.shape[0]} y_prio={yt_p.shape[0]}"
    logger.info("%s: X_tab=%s  y_type=%s  y_prio=%s", split, xt.shape, yt_t.shape, yt_p.shape)

assert len(tab_col_names) == X_tab_train.shape[1], \
    f"tabular_columns.json has {len(tab_col_names)} names but X has {X_tab_train.shape[1]} cols"
logger.info("Tabular columns (%d): %s", len(tab_col_names), tab_col_names)
logger.info("Type classes (%d):     %s", len(type_class_names), type_class_names)
logger.info("Priority classes (%d): %s", len(prio_class_names), prio_class_names)

# %% Cell 5 — MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("csip-advanced-classifiers")
logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)

# %% Cell 6 — Core abstractions: _instantiate, param builders, make_objective,
#             run_optuna_mlflow, calibrate_thresholds

def _instantiate(spec: dict) -> object:
    """Build a fresh, unfitted estimator from a spec dict."""
    algo   = spec["algo"]
    params = spec["params"]
    if algo == "rf":
        return RandomForestClassifier(
            **params, class_weight="balanced", random_state=RANDOM_STATE
        )
    elif algo == "xgb":
        return XGBClassifier(
            **params,
            n_jobs=1,               # deterministic
            tree_method="hist",     # deterministic on CPU
            eval_metric="mlogloss",
            verbosity=0,
            random_state=RANDOM_STATE,
        )
    elif algo == "lgbm":
        return LGBMClassifier(
            **params, class_weight="balanced",
            verbosity=-1, random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown algo: {algo!r}")


# ── Hyperparameter spaces ─────────────────────────────────────────────────────

def rf_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def lgbm_params(trial: optuna.Trial) -> dict:
    max_depth  = trial.suggest_int("max_depth", 4, 15)   # min=4 → 2^4-1=15 ≥ num_leaves low=10
    max_leaves = min(100, 2 ** max_depth - 1)   # enforce num_leaves < 2^max_depth
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
        "max_depth":         max_depth,
        "num_leaves":        trial.suggest_int("num_leaves", 10, max_leaves),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }


# ── Optuna objective factory ──────────────────────────────────────────────────

def make_objective(X, y, algo: str, n_folds: int, build_params_fn, random_state: int = 42):
    """Returns an Optuna objective function that reports running mean to MedianPruner."""
    X_safe = np.array(X, copy=True)   # defensive copy — 5928x17 ~ 800KB
    y_safe = np.array(y, copy=True)
    _rs    = random_state              # captured in closure — not a mutable global ref

    def objective(trial: optuna.Trial) -> float:
        params = build_params_fn(trial)
        spec   = {"algo": algo, "params": params}
        skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=_rs)
        fold_scores: list[float] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_safe, y_safe)):
            X_tr, X_vf = X_safe[tr_idx], X_safe[val_idx]
            y_tr, y_vf = y_safe[tr_idx], y_safe[val_idx]

            model = _instantiate(spec)
            if algo == "xgb":
                # XGBoost has no class_weight param — compute on fold subset, not full set
                sw = compute_sample_weight("balanced", y_tr)
                model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)

            score = f1_score(y_vf, model.predict(X_vf), average="macro", zero_division=0)
            fold_scores.append(score)

            # Report running mean — stable + comparable across trials at same step
            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ── Main training + logging wrapper ──────────────────────────────────────────

def run_optuna_mlflow(
    task: str,
    algo: str,
    build_params_fn,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    class_names: list[str],
    save_path: Path,
    n_trials: int,
    n_folds: int,
    random_state: int,
) -> tuple[AdvancedClassifier, optuna.Study, float]:
    """
    Returns (clf, study, val_f1_macro):
      clf      — AdvancedClassifier fitted on FULL X_train with study.best_params
      study    — completed Optuna study (for trial history artifact)
      val_f1   — val_f1_macro of the final refitted model (used for best-model selection)
    """
    logger.info("Starting Optuna: algo=%s task=%s n_trials=%d n_folds=%d",
                algo, task, n_trials, n_folds)

    # 1. Run Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        make_objective(X_train, y_train, algo, n_folds, build_params_fn, random_state),
        n_trials=n_trials,
    )
    logger.info("Optuna done: best_value=%.4f  best_params=%s",
                study.best_value, study.best_params)

    # 2. Refit on FULL X_train with best hyperparameters
    spec        = {"algo": algo, "params": study.best_params}
    final_model = _instantiate(spec)
    if algo == "xgb":
        # Compute sample_weight on full training set for final refit
        sw = compute_sample_weight("balanced", y_train)
        final_model.fit(X_train, y_train, sample_weight=sw)
    else:
        final_model.fit(X_train, y_train)

    # 3. Wrap — set n_features_in EXPLICITLY (do not call train(), which would re-fit)
    clf = AdvancedClassifier(
        model_name=algo,
        task=task,
        model=final_model,
        feature_schema="tabular_only",
        n_features_in=X_train.shape[1],   # explicit — FastAPI input validation depends on this
        best_params=study.best_params,
    )

    # 4. Evaluate on val and test
    val_metrics  = clf.evaluate(X_val,  y_val,  "val")
    test_metrics = clf.evaluate(X_test, y_test, "test")
    all_metrics  = {**val_metrics, **test_metrics}
    val_f1 = val_metrics.get("val_f1_macro") or -1.0

    # 5. Log to MLflow
    run_name = f"{algo}_{task.replace('ticket_', '')}"   # e.g. "lgbm_type"
    with mlflow.start_run(run_name=run_name):
        # Namespace hyperparams with hp_ to avoid key collision with fixed params
        mlflow_params: dict[str, str] = {
            "algo":           algo,
            "task":           task,
            "feature_schema": "tabular_only",
            "n_features_in":  str(clf.n_features_in),
            "n_trials":       str(n_trials),
            "n_folds":        str(n_folds),
            "fast_mode":      str(FAST_MODE),
        }
        mlflow_params.update({f"hp_{k}": str(v) for k, v in study.best_params.items()})
        mlflow.log_params(mlflow_params)

        # Filter None before log_metrics — MLflow rejects None/NaN
        safe_metrics = {k: v for k, v in all_metrics.items() if v is not None}
        mlflow.log_metrics(safe_metrics)

        # Native MLflow flavor per algo
        sample_dense = X_val[:5]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = infer_signature(sample_dense, clf.model.predict(sample_dense))
        if algo == "rf":
            mlflow.sklearn.log_model(clf.model, "model", signature=sig,
                                     input_example=sample_dense[:2])
        elif algo == "xgb":
            mlflow.xgboost.log_model(clf.model, "model", signature=sig,
                                     input_example=sample_dense[:2])
        elif algo == "lgbm":
            mlflow.lightgbm.log_model(clf.model, "model", signature=sig,
                                      input_example=sample_dense[:2])

        # Confusion matrix on test set
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        png_path = CHARTS_DIR / f"cm_{algo}_{task.replace('ticket_', '')}.png"
        plot_confusion_matrix(
            y_test, clf.model.predict(X_test),
            class_names=class_names,
            title=f"Confusion Matrix -- {algo.upper()} / {task.replace('_', ' ').title()}",
            save_path=png_path,
        )
        mlflow.log_artifact(str(png_path))

        # Trial history — log then delete temp file
        trial_csv = Path("trial_history_tmp.csv")
        study.trials_dataframe().to_csv(trial_csv, index=False)
        mlflow.log_artifact(str(trial_csv))
        trial_csv.unlink(missing_ok=True)

        logger.info(
            "Run %s OK: val_f1_macro=%.4f  val_roc_auc=%s",
            run_name, val_f1,
            f"{val_metrics.get('val_roc_auc'):.4f}"
            if val_metrics.get("val_roc_auc") is not None else "n/a",
        )

    # 6. Save fitted model
    clf.save(save_path)

    return clf, study, float(val_f1)


# ── Confidence threshold calibration ─────────────────────────────────────────

def calibrate_thresholds(
    model,
    X_val, y_val,
    X_test, y_test,
    auto_route_target: float = 0.85,
    flag_target: float = 0.70,
) -> tuple[dict | None, dict | None, pd.DataFrame]:
    """
    Select thresholds on val, verify on test.
    Returns (best_auto, best_flag, sweep_df).
    """
    probs_val  = model.predict_proba(X_val)
    probs_test = model.predict_proba(X_test)

    rows = []
    # Start at 0.20 — realistic floor for 4-5 class models where max_proba ~ 1/n_classes
    for t in np.linspace(0.20, 1.00, 17):
        t = round(float(t), 2)
        mv = probs_val.max(axis=1) >= t
        mt = probs_test.max(axis=1) >= t
        if mv.sum() == 0:
            continue
        prec_val  = float((probs_val[mv].argmax(axis=1) == y_val[mv]).mean())
        cov_val   = float(mv.mean())
        prec_test = float((probs_test[mt].argmax(axis=1) == y_test[mt]).mean()) if mt.sum() else 0.0
        cov_test  = float(mt.mean()) if mt.sum() else 0.0
        rows.append({
            "threshold":      t,
            "precision_val":  round(prec_val, 6),
            "coverage_val":   round(cov_val, 6),
            "precision_test": round(prec_test, 6),
            "coverage_test":  round(cov_test, 6),
            "precision_gap":  round(abs(prec_val - prec_test), 6),
        })

    if not rows:
        logger.warning("calibrate_thresholds: no val samples exceed minimum threshold -- model confidence too low")
        empty_df = pd.DataFrame(columns=["threshold", "precision_val", "coverage_val",
                                         "precision_test", "coverage_test", "precision_gap"])
        return None, None, empty_df

    df = pd.DataFrame(rows)

    # Auto-route: lowest threshold achieving auto_route_target on val
    auto_cands = df[df["precision_val"] >= auto_route_target]
    best_auto  = auto_cands.iloc[0].to_dict() if not auto_cands.empty else None
    if best_auto and best_auto["precision_gap"] > 0.05:
        logger.warning(
            "auto-route threshold %.2f: val/test precision gap %.3f > 0.05",
            best_auto["threshold"], best_auto["precision_gap"]
        )

    # Flag-review: lowest threshold achieving flag_target, below auto-route threshold
    flag_cands = df[df["precision_val"] >= flag_target]
    if best_auto is not None:
        flag_cands = flag_cands[flag_cands["threshold"] < best_auto["threshold"]]
    best_flag = flag_cands.iloc[0].to_dict() if not flag_cands.empty else None

    return best_auto, best_flag, df


def _threshold_note(best_auto: dict | None) -> str:
    if best_auto is None:
        return "auto-route threshold not achievable at 85% precision"
    gap = best_auto["precision_gap"]
    if gap <= 0.05:
        return f"gap {gap:.3f} < 0.05 -- threshold validated on test set"
    return f"WARNING: gap {gap:.3f} > 0.05 -- threshold not fully validated"


# %% Cell 7 — RF — Ticket Type
logger.info("=" * 60)
logger.info("RF / Ticket Type")
logger.info("=" * 60)
clf_rf_type, study_rf_type, val_f1_rf_type = run_optuna_mlflow(
    task="ticket_type",
    algo="rf",
    build_params_fn=rf_params,
    X_train=X_tab_train, y_train=y_train_type,
    X_val=X_tab_val,     y_val=y_val_type,
    X_test=X_tab_test,   y_test=y_test_type,
    class_names=type_class_names,
    save_path=RF_TYPE_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 8 — RF — Ticket Priority
logger.info("=" * 60)
logger.info("RF / Ticket Priority")
logger.info("=" * 60)
clf_rf_prio, study_rf_prio, val_f1_rf_prio = run_optuna_mlflow(
    task="ticket_priority",
    algo="rf",
    build_params_fn=rf_params,
    X_train=X_tab_train, y_train=y_train_prio,
    X_val=X_tab_val,     y_val=y_val_prio,
    X_test=X_tab_test,   y_test=y_test_prio,
    class_names=prio_class_names,
    save_path=RF_PRIORITY_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 9 — XGBoost — Ticket Type
logger.info("=" * 60)
logger.info("XGBoost / Ticket Type")
logger.info("=" * 60)
clf_xgb_type, study_xgb_type, val_f1_xgb_type = run_optuna_mlflow(
    task="ticket_type",
    algo="xgb",
    build_params_fn=xgb_params,
    X_train=X_tab_train, y_train=y_train_type,
    X_val=X_tab_val,     y_val=y_val_type,
    X_test=X_tab_test,   y_test=y_test_type,
    class_names=type_class_names,
    save_path=XGB_TYPE_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 10 — XGBoost — Ticket Priority
logger.info("=" * 60)
logger.info("XGBoost / Ticket Priority")
logger.info("=" * 60)
clf_xgb_prio, study_xgb_prio, val_f1_xgb_prio = run_optuna_mlflow(
    task="ticket_priority",
    algo="xgb",
    build_params_fn=xgb_params,
    X_train=X_tab_train, y_train=y_train_prio,
    X_val=X_tab_val,     y_val=y_val_prio,
    X_test=X_tab_test,   y_test=y_test_prio,
    class_names=prio_class_names,
    save_path=XGB_PRIORITY_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 11 — LightGBM — Ticket Type
logger.info("=" * 60)
logger.info("LightGBM / Ticket Type")
logger.info("=" * 60)
clf_lgbm_type, study_lgbm_type, val_f1_lgbm_type = run_optuna_mlflow(
    task="ticket_type",
    algo="lgbm",
    build_params_fn=lgbm_params,
    X_train=X_tab_train, y_train=y_train_type,
    X_val=X_tab_val,     y_val=y_val_type,
    X_test=X_tab_test,   y_test=y_test_type,
    class_names=type_class_names,
    save_path=LGBM_TYPE_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 12 — LightGBM — Ticket Priority
logger.info("=" * 60)
logger.info("LightGBM / Ticket Priority")
logger.info("=" * 60)
clf_lgbm_prio, study_lgbm_prio, val_f1_lgbm_prio = run_optuna_mlflow(
    task="ticket_priority",
    algo="lgbm",
    build_params_fn=lgbm_params,
    X_train=X_tab_train, y_train=y_train_prio,
    X_val=X_tab_val,     y_val=y_val_prio,
    X_test=X_tab_test,   y_test=y_test_prio,
    class_names=prio_class_names,
    save_path=LGBM_PRIORITY_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 13 — Free Optuna study objects + garbage collect
del study_rf_type, study_rf_prio
del study_xgb_type, study_xgb_prio
del study_lgbm_type, study_lgbm_prio
gc.collect()
logger.info("Optuna study objects freed")

# %% Cell 14 — Best model selection + confidence threshold calibration
logger.info("=" * 60)
logger.info("Best-model selection + confidence threshold calibration")
logger.info("=" * 60)

all_results: dict[str, tuple[AdvancedClassifier, float]] = {
    "rf_type":      (clf_rf_type,   val_f1_rf_type),
    "rf_priority":  (clf_rf_prio,   val_f1_rf_prio),
    "xgb_type":     (clf_xgb_type,  val_f1_xgb_type),
    "xgb_priority": (clf_xgb_prio,  val_f1_xgb_prio),
    "lgbm_type":    (clf_lgbm_type, val_f1_lgbm_type),
    "lgbm_priority":(clf_lgbm_prio, val_f1_lgbm_prio),
}

type_results = {k: v for k, v in all_results.items() if "type" in k}
prio_results = {k: v for k, v in all_results.items() if "priority" in k}

best_type_key = max(type_results, key=lambda k: type_results[k][1])
best_prio_key = max(prio_results, key=lambda k: prio_results[k][1])
best_type_clf = type_results[best_type_key][0]
best_prio_clf = prio_results[best_prio_key][0]

logger.info("Best type model:     %s (val_f1_macro=%.4f)", best_type_key, type_results[best_type_key][1])
logger.info("Best priority model: %s (val_f1_macro=%.4f)", best_prio_key, prio_results[best_prio_key][1])

# Calibrate thresholds — same best models go into model_registry.json (no split)
best_auto_type, best_flag_type, _ = calibrate_thresholds(
    best_type_clf.model, X_tab_val, y_val_type, X_tab_test, y_test_type
)
best_auto_prio, best_flag_prio, _ = calibrate_thresholds(
    best_prio_clf.model, X_tab_val, y_val_prio, X_tab_test, y_test_prio
)

confidence_output = {
    "ticket_type": {
        "model":       best_type_clf.model_name,
        "auto_route":  best_auto_type,
        "flag_review": best_flag_type,
        "note":        _threshold_note(best_auto_type),
    },
    "ticket_priority": {
        "model":       best_prio_clf.model_name,
        "auto_route":  best_auto_prio,
        "flag_review": best_flag_prio,
        "note":        _threshold_note(best_auto_prio),
    },
}
CONFIDENCE_THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CONFIDENCE_THRESHOLDS_PATH, "w") as fh:
    json.dump(confidence_output, fh, cls=NumpyEncoder, indent=2)
logger.info("Saved %s", CONFIDENCE_THRESHOLDS_PATH)

# %% Cell 15 — Drift baseline from training set tabular features
logger.info("=" * 60)
logger.info("Drift baseline generation")
logger.info("=" * 60)

baseline_df = pd.DataFrame(X_tab_train, columns=tab_col_names)
drift_stats: dict[str, dict] = {}

for col in tab_col_names:
    col_data = baseline_df[col]
    stats: dict = {
        "mean": round(float(col_data.mean()), 6),
        "std":  round(float(col_data.std()),  6),
        "min":  round(float(col_data.min()),  6),
        "max":  round(float(col_data.max()),  6),
        "p25":  round(float(col_data.quantile(0.25)), 6),
        "p50":  round(float(col_data.quantile(0.50)), 6),
        "p75":  round(float(col_data.quantile(0.75)), 6),
    }
    # Low-cardinality columns (ordinal-encoded categoricals): add PSI-compatible
    # frequency distribution — Evidently uses PSI not z-score for categoricals
    if col_data.nunique() <= 10:
        freq = col_data.value_counts(normalize=True).sort_index()
        stats["value_frequencies"] = {
            str(k): round(float(v), 6) for k, v in freq.items()
        }
    drift_stats[col] = stats

drift_baseline = {
    "generated_at":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n_train":       int(X_tab_train.shape[0]),
    "feature_names": tab_col_names,
    "stats":         drift_stats,
}
DRIFT_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(DRIFT_BASELINE_PATH, "w") as fh:
    json.dump(drift_baseline, fh, cls=NumpyEncoder, indent=2)
logger.info("Saved %s", DRIFT_BASELINE_PATH)

# %% Cell 16 — Save section metrics JSON + model registry
logger.info("=" * 60)
logger.info("Saving section metrics and model registry")
logger.info("=" * 60)

metrics_dir = ARTIFACTS_DIR / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
reports_dir = ARTIFACTS_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# Load S5 baselines for vs_baseline computation
s5_path = metrics_dir / "section_05_metrics.json"
s5_type_baseline  = -1.0
s5_prio_baseline  = -1.0
if s5_path.exists():
    with open(s5_path) as fh:
        s5 = json.load(fh)
    # Find best val_f1_macro per task from S5 runs
    for run in s5.get("runs", []):
        if run.get("task") == "ticket_type":
            s5_type_baseline = max(s5_type_baseline, run.get("val_f1_macro") or -1.0)
        elif run.get("task") == "ticket_priority":
            s5_prio_baseline = max(s5_prio_baseline, run.get("val_f1_macro") or -1.0)

# Build per-model metrics — val_f1 scalars come from the final refitted model
# (returned by run_optuna_mlflow), not from the Optuna CV study.best_value.
# Test metrics are already logged to MLflow; this JSON is a convenience cache.
section_06_metrics = {
    "section":      6,
    "fast_mode":    FAST_MODE,
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "models": {
        "rf_type":       {"val_f1_macro": val_f1_rf_type,   "n_trials": N_TRIALS,
                          "best_params": clf_rf_type.best_params},
        "rf_priority":   {"val_f1_macro": val_f1_rf_prio,   "n_trials": N_TRIALS,
                          "best_params": clf_rf_prio.best_params},
        "xgb_type":      {"val_f1_macro": val_f1_xgb_type,  "n_trials": N_TRIALS,
                          "best_params": clf_xgb_type.best_params},
        "xgb_priority":  {"val_f1_macro": val_f1_xgb_prio,  "n_trials": N_TRIALS,
                          "best_params": clf_xgb_prio.best_params},
        "lgbm_type":     {"val_f1_macro": val_f1_lgbm_type, "n_trials": N_TRIALS,
                          "best_params": clf_lgbm_type.best_params},
        "lgbm_priority": {"val_f1_macro": val_f1_lgbm_prio, "n_trials": N_TRIALS,
                          "best_params": clf_lgbm_prio.best_params},
    },
    "best_per_task": {
        "ticket_type": {
            "algo":         best_type_clf.model_name,
            "val_f1_macro": type_results[best_type_key][1],
            "artifact":     str(LGBM_TYPE_PATH.relative_to(PROJECT_ROOT))
                            if best_type_clf.model_name == "lgbm"
                            else str(RF_TYPE_PATH.relative_to(PROJECT_ROOT))
                            if best_type_clf.model_name == "rf"
                            else str(XGB_TYPE_PATH.relative_to(PROJECT_ROOT)),
        },
        "ticket_priority": {
            "algo":         best_prio_clf.model_name,
            "val_f1_macro": prio_results[best_prio_key][1],
            "artifact":     str(LGBM_PRIORITY_PATH.relative_to(PROJECT_ROOT))
                            if best_prio_clf.model_name == "lgbm"
                            else str(RF_PRIORITY_PATH.relative_to(PROJECT_ROOT))
                            if best_prio_clf.model_name == "rf"
                            else str(XGB_PRIORITY_PATH.relative_to(PROJECT_ROOT)),
        },
    },
}

with open(metrics_dir / "section_06_metrics.json", "w") as fh:
    json.dump(section_06_metrics, fh, cls=NumpyEncoder, indent=2)
logger.info("Saved %s", metrics_dir / "section_06_metrics.json")

# Model registry — vs_baseline as float (display formatted only in Cell 17)
def _vs_baseline(val_f1: float, baseline: float) -> float | None:
    if baseline < 0:
        return None
    return round(val_f1 - baseline, 6)

def _artifact_path(clf: AdvancedClassifier) -> str:
    mapping = {
        ("rf",   "ticket_type"):     RF_TYPE_PATH,
        ("rf",   "ticket_priority"): RF_PRIORITY_PATH,
        ("xgb",  "ticket_type"):     XGB_TYPE_PATH,
        ("xgb",  "ticket_priority"): XGB_PRIORITY_PATH,
        ("lgbm", "ticket_type"):     LGBM_TYPE_PATH,
        ("lgbm", "ticket_priority"): LGBM_PRIORITY_PATH,
    }
    p = mapping.get((clf.model_name, clf.task))
    return str(p.relative_to(PROJECT_ROOT)) if p else "unknown"

model_registry = {
    "ticket_type": {
        "section":       6,
        "algo":          best_type_clf.model_name,
        "val_f1_macro":  round(type_results[best_type_key][1], 6),
        "vs_baseline":   _vs_baseline(type_results[best_type_key][1], s5_type_baseline),
        "artifact":      _artifact_path(best_type_clf),
    },
    "ticket_priority": {
        "section":       6,
        "algo":          best_prio_clf.model_name,
        "val_f1_macro":  round(prio_results[best_prio_key][1], 6),
        "vs_baseline":   _vs_baseline(prio_results[best_prio_key][1], s5_prio_baseline),
        "artifact":      _artifact_path(best_prio_clf),
    },
}

with open(reports_dir / "model_registry.json", "w") as fh:
    json.dump(model_registry, fh, cls=NumpyEncoder, indent=2)
logger.info("Saved %s", reports_dir / "model_registry.json")

# %% Cell 17 — Summary (ASCII only — no Unicode, Windows cp1252 safe)
logger.info("=" * 60)
logger.info("SECTION 6 COMPLETE")
logger.info("MLflow experiment: csip-advanced-classifiers (6 runs)")
logger.info("-" * 60)

# Per-task leaderboard
for task_label, results, baseline in [
    ("Ticket Type",     type_results, s5_type_baseline),
    ("Ticket Priority", prio_results, s5_prio_baseline),
]:
    logger.info("%s results:", task_label)
    for key, (clf, f1) in sorted(results.items(), key=lambda x: -x[1][1]):
        delta = f1 - baseline
        vs = (f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}") if baseline >= 0 else "n/a"
        logger.info("  %-16s val_f1_macro=%.4f  vs_baseline=%s", key, f1, vs)

logger.info("-" * 60)

# Coverage floor check
for task_key, res in confidence_output.items():
    auto = res.get("auto_route")
    model_name = res.get("model", "?")
    if auto is None:
        logger.warning(
            "%s (%s): auto-route threshold not achievable at 85%% precision",
            task_key, model_name
        )
        logger.warning("  => Consider lower precision target or wait for DistilBERT (Section 9)")
    elif auto["coverage_val"] < 0.20:
        logger.warning(
            "%s (%s): auto-route coverage %.1f%% < 20%% -- not operationally viable",
            task_key, model_name, auto["coverage_val"] * 100
        )
        logger.warning("  => Consider lower precision target or wait for DistilBERT (Section 9)")
    else:
        logger.info(
            "%s (%s): auto-route threshold=%.2f  precision_val=%.4f  coverage=%.1f%%  gap=%.3f",
            task_key, model_name,
            auto["threshold"], auto["precision_val"],
            auto["coverage_val"] * 100, auto["precision_gap"],
        )

logger.info("-" * 60)
logger.info("Artifacts saved:")
logger.info("  models/rf_type_classifier.pkl")
logger.info("  models/rf_priority_classifier.pkl")
logger.info("  models/xgb_type_classifier.pkl")
logger.info("  models/xgb_priority_classifier.pkl")
logger.info("  models/lgbm_type_classifier.pkl")
logger.info("  models/lgbm_priority_classifier.pkl")
logger.info("  artifacts/reports/confidence_thresholds.json")
logger.info("  artifacts/drift/training_baseline.json")
logger.info("  artifacts/metrics/section_06_metrics.json")
logger.info("  artifacts/reports/model_registry.json")
logger.info("  artifacts/charts/cm_*.png  (6 confusion matrices)")
logger.info("-" * 60)
logger.info("Next step: open http://localhost:5001 to view MLflow runs")
logger.info("Then run: python notebooks/07_regression.py")
logger.info("=" * 60)
