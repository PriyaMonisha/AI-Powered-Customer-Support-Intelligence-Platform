# filename: notebooks/07_regression.py
# purpose:  Section 7 — RF, XGBoost, LightGBM regression on hours_to_resolve (closed tickets)
# version:  1.0

# %% Cell 1 — FAST_MODE (must be first line of code)
FAST_MODE = True

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 5  if FAST_MODE else 30
N_FOLDS  = 3  if FAST_MODE else 5

# %% Cell 2 — Imports + PROJECT_ROOT
import datetime
import gc
import json
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress algo warnings at module level — not inside objective (avoids filter accumulation)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from lightgbm import LGBMRegressor
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from config import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    FAST_MODE as CFG_FAST_MODE,
    FEATURES_DIR,
    LGBM_REGRESSOR_PATH,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    RF_REGRESSOR_PATH,
    XGBOOST_REGRESSOR_PATH,
)
from src.models.advanced_regressor import AdvancedRegressor, _mape_safe
from src.utils.helpers import NumpyEncoder

# Scipy skewness with fallback
try:
    from scipy.stats import skew as _scipy_skew
    def _skewness(arr: np.ndarray) -> float:
        return float(_scipy_skew(arr))
except ImportError:
    # Biased population estimator — acceptable for N>1000; install scipy for unbiased
    def _skewness(arr: np.ndarray) -> float:
        return float(np.mean(((arr - arr.mean()) / arr.std()) ** 3))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("section7")
logger.info("FAST_MODE=%s  N_TRIALS=%d  N_FOLDS=%d", FAST_MODE, N_TRIALS, N_FOLDS)

# %% Cell 3 — Load features + regression targets
logger.info("Loading regression feature artifacts from %s ...", FEATURES_DIR)

X_tab_all = {s: np.load(FEATURES_DIR / f"X_{s}_tabular.npy") for s in ["train", "val", "test"]}
y_reg_all  = {s: np.load(FEATURES_DIR / f"y_{s}_reg.npy")    for s in ["train", "val", "test"]}

with open(FEATURES_DIR / "tabular_columns.json") as fh:
    tab_col_names: list[str] = json.load(fh)

# Shape assertions before masking
for split in ["train", "val", "test"]:
    assert X_tab_all[split].shape[0] == y_reg_all[split].shape[0], (
        f"{split}: row count mismatch X={X_tab_all[split].shape[0]} "
        f"y={y_reg_all[split].shape[0]}"
    )

# Apply NaN mask — closed tickets only
masks = {s: ~np.isnan(y_reg_all[s]) for s in ["train", "val", "test"]}
X     = {s: X_tab_all[s][masks[s]]  for s in ["train", "val", "test"]}
y     = {s: y_reg_all[s][masks[s]]  for s in ["train", "val", "test"]}

logger.info(
    "Closed tickets -- train=%d  val=%d  test=%d",
    X["train"].shape[0], X["val"].shape[0], X["test"].shape[0],
)

# %% Cell 4 — Drop constant columns + target analysis
EXPECTED_CONST = {"is_resolved", "has_first_response", "csat_available", "subject_word_count"}

const_mask   = np.all(X["train"] == X["train"][0], axis=0)
drop_indices = sorted(np.where(const_mask)[0].tolist())
keep_mask    = [i for i in range(len(tab_col_names)) if i not in drop_indices]
reg_col_names = [tab_col_names[i] for i in keep_mask]

detected         = {tab_col_names[i] for i in drop_indices}
unexpected_drops = detected - EXPECTED_CONST
missing_drops    = EXPECTED_CONST - detected

if unexpected_drops:
    logger.warning(
        "Unexpected constant columns dropped: %s -- verify S4 feature engineering",
        unexpected_drops,
    )
if missing_drops:
    logger.warning(
        "Expected constant columns NOT constant: %s -- dataset may have changed",
        missing_drops,
    )

logger.info("Dropped %d constant columns: %s", len(drop_indices), sorted(detected))
logger.info("Regression features (%d): %s", len(reg_col_names), reg_col_names)

X = {s: X[s][:, keep_mask] for s in ["train", "val", "test"]}

# Target distribution — data-driven log transform decision
skewness = _skewness(y["train"])
LOG_TRANSFORM = abs(skewness) > 1.0
logger.info(
    "y_train skewness=%.4f  min=%.4f  max=%.4f  mean=%.4f  median=%.4f  LOG_TRANSFORM=%s",
    skewness, y["train"].min(), y["train"].max(),
    y["train"].mean(), float(np.median(y["train"])), LOG_TRANSFORM,
)

if LOG_TRANSFORM:
    y_fit = {s: np.log1p(y[s]) for s in y}
    logger.info("Targets transformed: log1p applied")
else:
    y_fit = y   # same references — no copy, no transform
    logger.info("Targets NOT transformed (|skewness| <= 1.0)")

# %% Cell 5 — MLflow setup
_mlflow_uri = MLFLOW_TRACKING_URI
try:
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-regression-models")
    logger.info("MLflow connected: %s", _mlflow_uri)
except Exception:
    _mlflow_uri = (PROJECT_ROOT / "mlruns").as_uri()   # file:///C:/... avoids Windows drive-letter parse bug
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-regression-models")
    logger.warning("MLflow server unavailable -- using local file store: %s", _mlflow_uri)

# %% Cell 6 — Training helpers (defined once, called in Cells 7-9)

# ── Visualization constants ───────────────────────────────────────────────────
SCATTER_COLOR = "#2166ac"   # blue — readable on white background
HIST_COLOR    = "#d6604d"   # red-orange — distinguishable from scatter


# ── Context manager for scoped warning suppression inside folds ───────────────
@contextmanager
def _suppress_algo_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
        yield


# ── Estimator factory ─────────────────────────────────────────────────────────
def _instantiate_reg(spec: dict) -> object:
    """Build a fresh, unfitted regressor from a spec dict."""
    algo   = spec["algo"]
    params = spec["params"]
    if algo == "rf":
        return RandomForestRegressor(
            **params, n_jobs=1, random_state=RANDOM_STATE
        )
    elif algo == "xgb":
        return XGBRegressor(
            **params,
            n_jobs=1,
            tree_method="hist",
            eval_metric="rmse",
            verbosity=0,
            random_state=RANDOM_STATE,
        )
    elif algo == "lgbm":
        return LGBMRegressor(**params, verbosity=-1, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown algo: {algo!r}")


# ── Hyperparameter spaces ─────────────────────────────────────────────────────
def rf_reg_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def xgb_reg_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def lgbm_reg_params(trial: optuna.Trial) -> dict:
    max_depth  = trial.suggest_int("max_depth", 4, 15)
    max_leaves = min(100, 2 ** max_depth - 1)   # enforce num_leaves < 2^max_depth
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
        "max_depth":         max_depth,
        "num_leaves":        trial.suggest_int("num_leaves", 10, max_leaves),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }


# ── Optuna objective factory ──────────────────────────────────────────────────
def make_reg_objective(X, y, algo: str, n_folds: int, build_params_fn, random_state: int = 42):
    """Returns an Optuna objective that minimizes RMSE with running-mean reporting."""
    X_safe = np.array(X, copy=True)
    y_safe = np.array(y, copy=True)
    _rs    = random_state

    def objective(trial: optuna.Trial) -> float:
        params = build_params_fn(trial)
        spec   = {"algo": algo, "params": params}
        kf     = KFold(n_splits=n_folds, shuffle=True, random_state=_rs)
        fold_rmses: list[float] = []

        for fold_idx, (tr_idx, vl_idx) in enumerate(kf.split(X_safe)):
            X_tr, X_vf = X_safe[tr_idx], X_safe[vl_idx]
            y_tr, y_vf = y_safe[tr_idx], y_safe[vl_idx]

            model = _instantiate_reg(spec)
            with _suppress_algo_warnings():
                model.fit(X_tr, y_tr)

            rmse = float(mean_squared_error(y_vf, model.predict(X_vf)) ** 0.5)
            fold_rmses.append(rmse)

            # Report running mean — stable + comparable across trials at same step
            trial.report(float(np.mean(fold_rmses)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_rmses))

    return objective


# ── Visualization helpers ─────────────────────────────────────────────────────
def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    algo: str,
    charts_dir: Path,
) -> Path:
    """2-panel residual chart: predicted vs actual scatter + residual histogram."""
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2   = float(r2_score(y_true, y_pred))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Regression Diagnostics -- {algo.upper()} / hours_to_resolve", fontsize=12)

    # Left: predicted vs actual
    ax1.scatter(y_true, y_pred, alpha=0.4, color=SCATTER_COLOR, s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
    ax1.set_xlabel("Actual (hours)")
    ax1.set_ylabel("Predicted (hours)")
    ax1.set_title("Predicted vs Actual")
    ax1.text(
        0.05, 0.92,
        f"RMSE={rmse:.3f}  R2={r2:.3f}",
        transform=ax1.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    ax1.legend(fontsize=8)

    # Right: residual histogram
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=40, color=HIST_COLOR, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Residual (actual - predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")

    plt.tight_layout()
    charts_dir.mkdir(parents=True, exist_ok=True)
    out_path = charts_dir / f"residuals_{algo}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importance(
    model,
    feature_names: list[str],
    algo: str,
    charts_dir: Path,
) -> Path:
    """Horizontal bar chart of feature importances. Annotates zero-importance count."""
    importances = model.feature_importances_
    n_top = min(15, len(feature_names))
    indices = np.argsort(importances)[::-1][:n_top]
    top_names  = [feature_names[i] for i in indices]
    top_imps   = importances[indices]

    zero_count = int((importances == 0).sum())
    title = f"Feature Importances -- {algo.upper()} / hours_to_resolve"
    if zero_count > 0:
        title += f"\n({zero_count}/{len(importances)} features have zero importance)"

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * n_top + 2)))
    ax.barh(range(n_top), top_imps[::-1], color=SCATTER_COLOR)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance (Gini / Gain)")
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    charts_dir.mkdir(parents=True, exist_ok=True)
    out_path = charts_dir / f"feature_importance_regression_{algo}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main training + logging wrapper ──────────────────────────────────────────
def run_optuna_mlflow_reg(
    task: str,
    algo: str,
    build_params_fn,
    X_train, y_train,       # y_train already transformed if LOG_TRANSFORM=True
    X_val,   y_val_raw,     # always raw — evaluate() inverts transform internally
    X_test,  y_test_raw,    # always raw
    log_transform: bool,    # passed to evaluate() + stored on regressor
    feature_names: list[str],
    save_path: Path,
    n_trials: int,
    n_folds: int,
    random_state: int,
) -> tuple["AdvancedRegressor", optuna.Study, dict]:
    """
    Returns (reg, study, metrics_dict):
      reg          — AdvancedRegressor fitted on FULL X_train with study.best_params
      study        — completed Optuna study (for trial history artifact)
      metrics_dict — all val_* and test_* metrics (no re-evaluation needed in caller)
    """
    logger.info(
        "Starting Optuna: algo=%s task=%s n_trials=%d n_folds=%d log_transform=%s",
        algo, task, n_trials, n_folds, log_transform,
    )

    # 1. Run Optuna study — direction=minimize for RMSE
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        make_reg_objective(X_train, y_train, algo, n_folds, build_params_fn, random_state),
        n_trials=n_trials,
    )
    logger.info(
        "Optuna done: best_value(cv_rmse)=%.4f  best_params=%s",
        study.best_value, study.best_params,
    )

    # 2. Refit on FULL X_train with best hyperparameters
    spec        = {"algo": algo, "params": study.best_params}
    final_model = _instantiate_reg(spec)
    with _suppress_algo_warnings():
        final_model.fit(X_train, y_train)

    # 3. Wrap — set n_features_in EXPLICITLY (do not call train(), which would re-fit)
    reg = AdvancedRegressor(
        model_name=algo,
        task=task,
        model=final_model,
        feature_schema="tabular_only",
        n_features_in=X_train.shape[1],
        best_params=study.best_params,
        log_transform=log_transform,
    )

    # 4. Evaluate on val and test (real-space metrics via evaluate(log_transform=...))
    val_metrics  = reg.evaluate(X_val,  y_val_raw,  "val",  log_transform=log_transform)
    test_metrics = reg.evaluate(X_test, y_test_raw, "test", log_transform=log_transform)
    all_metrics  = {**val_metrics, **test_metrics}

    # 5. Log to MLflow
    run_name = f"{algo}_hours_to_resolve"
    with mlflow.start_run(run_name=run_name):
        mlflow_params: dict[str, str] = {
            "algo":           algo,
            "task":           task,
            "feature_schema": "tabular_only",
            "n_features_in":  str(reg.n_features_in),
            "n_trials":       str(n_trials),
            "n_folds":        str(n_folds),
            "fast_mode":      str(FAST_MODE),
            "log_transform":  str(log_transform),
        }
        mlflow_params.update({f"hp_{k}": str(v) for k, v in study.best_params.items()})
        mlflow.log_params(mlflow_params)

        # Filter None before log_metrics — MLflow rejects None values
        safe_metrics = {k: v for k, v in all_metrics.items() if v is not None}
        mlflow.log_metrics(safe_metrics)

        # Native MLflow flavor per algo
        sample_dense = X_val[:5]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = infer_signature(sample_dense, reg.model.predict(sample_dense))
        if algo == "rf":
            mlflow.sklearn.log_model(reg.model, "model", signature=sig,
                                     input_example=sample_dense[:2])
        elif algo == "xgb":
            mlflow.xgboost.log_model(reg.model, "model", signature=sig,
                                     input_example=sample_dense[:2])
        elif algo == "lgbm":
            mlflow.lightgbm.log_model(reg.model, "model", signature=sig,
                                      input_example=sample_dense[:2])

        # Residual chart — predict() applies expm1 if log_transform=True
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        residual_path = plot_residuals(y_test_raw, reg.predict(X_test), algo, CHARTS_DIR)
        mlflow.log_artifact(str(residual_path))

        # Feature importance chart
        fi_path = plot_feature_importance(reg.model, feature_names, algo, CHARTS_DIR)
        mlflow.log_artifact(str(fi_path))

        # Trial history — deterministic path inside artifacts/tmp/ (gitignored)
        tmp_dir = ARTIFACTS_DIR / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        trial_csv = tmp_dir / f"trial_history_reg_{algo}.csv"
        study.trials_dataframe().to_csv(trial_csv, index=False)
        mlflow.log_artifact(str(trial_csv))
        trial_csv.unlink(missing_ok=True)

        logger.info(
            "Run %s OK: val_rmse=%.4f  val_r2=%.4f  test_rmse=%.4f",
            run_name,
            val_metrics.get("val_rmse", float("nan")),
            val_metrics.get("val_r2",   float("nan")),
            test_metrics.get("test_rmse", float("nan")),
        )

    # 6. Save fitted model
    reg.save(save_path)

    return reg, study, all_metrics


# %% Cell 7 — RF Regressor
logger.info("=" * 60)
logger.info("RF Regressor / hours_to_resolve")
logger.info("=" * 60)
reg_rf, study_rf, metrics_rf = run_optuna_mlflow_reg(
    task="hours_to_resolve",
    algo="rf",
    build_params_fn=rf_reg_params,
    X_train=X["train"], y_train=y_fit["train"],
    X_val=X["val"],     y_val_raw=y["val"],
    X_test=X["test"],   y_test_raw=y["test"],
    log_transform=LOG_TRANSFORM,
    feature_names=reg_col_names,
    save_path=RF_REGRESSOR_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 8 — XGBoost Regressor
logger.info("=" * 60)
logger.info("XGBoost Regressor / hours_to_resolve")
logger.info("=" * 60)
reg_xgb, study_xgb, metrics_xgb = run_optuna_mlflow_reg(
    task="hours_to_resolve",
    algo="xgb",
    build_params_fn=xgb_reg_params,
    X_train=X["train"], y_train=y_fit["train"],
    X_val=X["val"],     y_val_raw=y["val"],
    X_test=X["test"],   y_test_raw=y["test"],
    log_transform=LOG_TRANSFORM,
    feature_names=reg_col_names,
    save_path=XGBOOST_REGRESSOR_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 9 — LightGBM Regressor
logger.info("=" * 60)
logger.info("LightGBM Regressor / hours_to_resolve")
logger.info("=" * 60)
reg_lgbm, study_lgbm, metrics_lgbm = run_optuna_mlflow_reg(
    task="hours_to_resolve",
    algo="lgbm",
    build_params_fn=lgbm_reg_params,
    X_train=X["train"], y_train=y_fit["train"],
    X_val=X["val"],     y_val_raw=y["val"],
    X_test=X["test"],   y_test_raw=y["test"],
    log_transform=LOG_TRANSFORM,
    feature_names=reg_col_names,
    save_path=LGBM_REGRESSOR_PATH,
    n_trials=N_TRIALS,
    n_folds=N_FOLDS,
    random_state=RANDOM_STATE,
)

# %% Cell 10 — Free Optuna study objects + garbage collect
del study_rf, study_xgb, study_lgbm
gc.collect()
logger.info("Optuna study objects freed")

# %% Cell 11 — Best model selection + summary table
logger.info("=" * 60)
logger.info("Best-model selection + summary")
logger.info("=" * 60)

all_model_metrics = {"rf": metrics_rf, "xgb": metrics_xgb, "lgbm": metrics_lgbm}
all_regressors    = {"rf": reg_rf,     "xgb": reg_xgb,     "lgbm": reg_lgbm}

best_algo = min(all_model_metrics, key=lambda a: all_model_metrics[a]["val_rmse"])
best_reg  = all_regressors[best_algo]

# ASCII summary table — no Unicode, Windows cp1252 safe
logger.info("%-6s  %-9s  %-9s  %-9s  %-10s  %-9s  %-9s",
            "Model", "val_RMSE", "val_MAE", "val_R2",
            "test_RMSE", "test_MAE", "test_R2")
logger.info("-" * 72)
for algo_key in ["rf", "xgb", "lgbm"]:
    m = all_model_metrics[algo_key]
    marker = " <-- best" if algo_key == best_algo else ""
    logger.info(
        "%-6s  %-9.4f  %-9.4f  %-9.4f  %-10.4f  %-9.4f  %-9.4f%s",
        algo_key,
        m["val_rmse"],  m["val_mae"],  m["val_r2"],
        m["test_rmse"], m["test_mae"], m["test_r2"],
        marker,
    )

logger.info("-" * 72)
logger.info(
    "Best model: %s  val_RMSE=%.4f  test_RMSE=%.4f",
    best_algo,
    all_model_metrics[best_algo]["val_rmse"],
    all_model_metrics[best_algo]["test_rmse"],
)

# %% Cell 12 — Save section_07_metrics.json
logger.info("=" * 60)
logger.info("Saving section_07_metrics.json")
logger.info("=" * 60)

metrics_dir = ARTIFACTS_DIR / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)

_path_map = {
    "rf":   RF_REGRESSOR_PATH,
    "xgb":  XGBOOST_REGRESSOR_PATH,
    "lgbm": LGBM_REGRESSOR_PATH,
}

section_07_metrics = {
    "section":         7,
    "fast_mode":       FAST_MODE,
    "log_transform":   LOG_TRANSFORM,
    "generated_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n_train":         int(X["train"].shape[0]),
    "n_val":           int(X["val"].shape[0]),
    "n_test":          int(X["test"].shape[0]),
    "n_features":      int(X["train"].shape[1]),
    "feature_names":   reg_col_names,
    "dropped_columns": [tab_col_names[i] for i in drop_indices],
    "models": {
        algo_key: {
            **{k: round(v, 6) for k, v in all_model_metrics[algo_key].items()
               if v is not None},
            **{k: None for k, v in all_model_metrics[algo_key].items() if v is None},
            "best_params": all_regressors[algo_key].best_params,
            "n_trials":    N_TRIALS,
        }
        for algo_key in ["rf", "xgb", "lgbm"]
    },
    "best_model": {
        "algo":      best_algo,
        "val_rmse":  round(all_model_metrics[best_algo]["val_rmse"], 6),
        "test_rmse": round(all_model_metrics[best_algo]["test_rmse"], 6),
        "artifact":  str(_path_map[best_algo].relative_to(PROJECT_ROOT)),
    },
}

with open(metrics_dir / "section_07_metrics.json", "w") as fh:
    json.dump(section_07_metrics, fh, cls=NumpyEncoder, indent=2)
logger.info("Saved %s", metrics_dir / "section_07_metrics.json")

# %% Cell 13 — Final summary (ASCII only)
logger.info("=" * 60)
logger.info("SECTION 7 COMPLETE")
logger.info("MLflow experiment: csip-regression-models (3 runs)")
logger.info("-" * 60)
logger.info(
    "Best model: %s  val_RMSE=%.4f  test_RMSE=%.4f  test_R2=%.4f",
    best_algo,
    all_model_metrics[best_algo]["val_rmse"],
    all_model_metrics[best_algo]["test_rmse"],
    all_model_metrics[best_algo]["test_r2"],
)
logger.info("Log transform applied: %s", LOG_TRANSFORM)
logger.info("-" * 60)
logger.info("Artifacts saved:")
logger.info("  models/rf_regressor.pkl")
logger.info("  models/xgboost_regressor.pkl")
logger.info("  models/lgbm_regressor.pkl")
logger.info("  artifacts/metrics/section_07_metrics.json")
logger.info("  artifacts/charts/residuals_rf.png")
logger.info("  artifacts/charts/residuals_xgb.png")
logger.info("  artifacts/charts/residuals_lgbm.png")
logger.info("  artifacts/charts/feature_importance_regression_rf.png")
logger.info("  artifacts/charts/feature_importance_regression_xgb.png")
logger.info("  artifacts/charts/feature_importance_regression_lgbm.png")
logger.info("-" * 60)
logger.info("Next step: python notebooks/08b_clustering.py")
logger.info("=" * 60)