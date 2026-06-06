# filename: notebooks/08c_shap.py
# purpose:  SHAP explainability for LGBM priority classifier + LGBM regressor (Section 8c)
# version:  1.0

FAST_MODE = True   # FIRST LINE

# ---------------------------------------------------------------------------
# Imports + PROJECT_ROOT
# ---------------------------------------------------------------------------
import datetime
import json
import logging
import sys
import time
import warnings
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import mlflow
import numpy as np
import shap

from config import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    FEATURES_DIR,
    LGBM_PRIORITY_PATH,
    LGBM_REGRESSOR_PATH,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REGRESSOR_KEEP_MASK_PATH,
    SHAP_EXPLAINER_PRIORITY_PATH,
    SHAP_EXPLAINER_REGRESSOR_PATH,
)
from src.models.advanced_classifier import AdvancedClassifier
from src.models.advanced_regressor import AdvancedRegressor
from src.utils.helpers import NumpyEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("section8c")
logger.info("FAST_MODE=%s", FAST_MODE)

# ---------------------------------------------------------------------------
# Cell 3 — Load models
# ---------------------------------------------------------------------------
def load_models() -> tuple[AdvancedClassifier, AdvancedRegressor]:
    """Load LGBM priority classifier and LGBM regressor; validate feature counts."""
    clf = AdvancedClassifier.load(LGBM_PRIORITY_PATH)
    reg = AdvancedRegressor.load(LGBM_REGRESSOR_PATH)
    assert clf.n_features_in == 17, f"Priority clf expected 17 features, got {clf.n_features_in}"
    assert reg.n_features_in == 13, f"Regressor expected 13 features, got {reg.n_features_in}"
    logger.info("Loaded clf: %s (n_features=%d)", clf.model_name, clf.n_features_in)
    logger.info("Loaded reg: %s (n_features=%d)", reg.model_name, reg.n_features_in)
    return clf, reg

clf, reg = load_models()

# ---------------------------------------------------------------------------
# Cell 4 — Load and prepare SHAP data
# ---------------------------------------------------------------------------
def load_shap_data(
    features_dir: Path,
    models_dir: Path,
    tab_col_names: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        X_val_full    (847, 17) — for priority classifier, no mask
        prio_col_names list[str]
        X_val_reg     (N_closed, 13) — for regressor (nan-masked + keep_mask)
        y_val_prio    (847,) — for waterfall example selection
        reg_col_names list[str]
    """
    X_val_full = np.load(features_dir / "X_val_tabular.npy")
    y_val_reg  = np.load(features_dir / "y_val_reg.npy")
    y_val_prio = np.load(features_dir / "y_val_prio.npy")

    # Load keep_mask saved by the S7 patch (do NOT recompute)
    keep_mask_path = models_dir / "regressor_keep_mask.npy"
    if not keep_mask_path.exists():
        raise FileNotFoundError(
            f"{keep_mask_path} not found. "
            "Run the Step 0 patch to notebooks/07_regression.py first."
        )
    keep_mask = np.load(keep_mask_path)    # bool array shape (17,)
    assert keep_mask.dtype == bool and keep_mask.shape == (17,), \
        f"Invalid keep_mask: shape={keep_mask.shape} dtype={keep_mask.dtype}"

    closed_mask   = ~np.isnan(y_val_reg)
    X_val_reg     = X_val_full[closed_mask][:, keep_mask]
    reg_col_names = [tab_col_names[i] for i in np.where(keep_mask)[0]]

    logger.info("Priority SHAP: X_val_full=%s", X_val_full.shape)
    logger.info("Regressor SHAP: X_val_reg=%s  (%d closed tickets)", X_val_reg.shape, closed_mask.sum())
    return X_val_full, tab_col_names, X_val_reg, y_val_prio, reg_col_names

with open(FEATURES_DIR / "tabular_columns.json") as fh:
    tab_col_names: list[str] = json.load(fh)

with open(FEATURES_DIR / "label_maps.json") as fh:
    label_maps: dict = json.load(fh)

priority_names = [label_maps["ticket_priority"][str(i)] for i in range(4)]

X_val_full, prio_col_names, X_val_reg, y_val_prio, reg_col_names = load_shap_data(
    FEATURES_DIR, MODELS_DIR, tab_col_names
)

# ---------------------------------------------------------------------------
# Cell 5 — Build SHAP explainers
# ---------------------------------------------------------------------------
# Use default model_output for multi-class LGBM — 'raw' changes the return shape
# and breaks list-of-arrays convention expected by summary_plot
logger.info("Building TreeExplainer for priority classifier ...")
explainer_prio = shap.TreeExplainer(clf.model)

logger.info("Building TreeExplainer for regressor ...")
explainer_reg = shap.TreeExplainer(reg.model)

# ---------------------------------------------------------------------------
# Cell 6 — Compute SHAP values (log progress + timing)
# ---------------------------------------------------------------------------
logger.info("Computing SHAP values for priority classifier (%d samples) ...", len(X_val_full))
t0 = time.time()
shap_values_prio = explainer_prio.shap_values(X_val_full)
logger.info("Priority SHAP done in %.1fs  — shape=%s", time.time() - t0,
            getattr(shap_values_prio, "shape", [getattr(a, "shape", None) for a in shap_values_prio]))

logger.info("Computing SHAP values for regressor (%d samples) ...", len(X_val_reg))
t0 = time.time()
shap_values_reg = explainer_reg.shap_values(X_val_reg)
logger.info("Regressor SHAP done in %.1fs  — shape=%s", time.time() - t0,
            getattr(shap_values_reg, "shape", None))

# Normalise to consistent 3D array (samples, features, classes):
# SHAP 0.45.1 + LGBM returns ndarray (N, F, C); older versions may return list[C] of (N, F).
if isinstance(shap_values_prio, list):
    shap_values_prio = np.stack(shap_values_prio, axis=2)   # list → (N, F, C)
# shap_values_prio is now (847, 17, 4)
# shap_values_reg  is (N_closed, 13) — 2D for regression

# ---------------------------------------------------------------------------
# Cell 7 — Helper: smart waterfall example selection
# ---------------------------------------------------------------------------
def select_waterfall_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    target_class: int,
    target_class_name: str,
) -> tuple[int, int]:
    """
    correct_idx: target-class ticket classified correctly, probability closest to 0.7
    wrong_idx:   target-class ticket misclassified with highest wrong-class confidence
    """
    critical_mask = y_true == target_class
    correct_mask  = (y_pred == target_class) & critical_mask
    wrong_mask    = (y_pred != target_class) & critical_mask

    if correct_mask.sum() > 0:
        probs = y_proba[correct_mask, target_class]
        correct_idx = int(np.where(correct_mask)[0][np.argmin(np.abs(probs - 0.7))])
    else:
        logger.warning("No correct predictions for %s; using first instance", target_class_name)
        correct_idx = int(np.where(critical_mask)[0][0])

    if wrong_mask.sum() > 0:
        wrong_preds   = y_pred[wrong_mask]
        wrong_proba   = y_proba[wrong_mask]
        confidences   = [float(wrong_proba[i, p]) for i, p in enumerate(wrong_preds)]
        wrong_idx     = int(np.where(wrong_mask)[0][np.argmax(confidences)])
    else:
        logger.warning("No misclassifications for %s; using first wrong prediction", target_class_name)
        wrong_idx = int(np.where(y_pred != y_true)[0][0])

    logger.info(
        "Waterfall examples — correct idx=%d (prob=%.3f)  wrong idx=%d (pred=%s conf=%.3f)",
        correct_idx,
        float(y_proba[correct_idx, target_class]),
        wrong_idx,
        priority_names[int(y_pred[wrong_idx])],
        float(y_proba[wrong_idx, int(y_pred[wrong_idx])]),
    )
    return correct_idx, wrong_idx

# Priority predictions for waterfall selection
y_pred_prio  = clf.model.predict(X_val_full)
y_proba_prio = clf.model.predict_proba(X_val_full)

# Critical = class 0 per label_maps.json
CRITICAL_IDX = 0
correct_idx, wrong_idx = select_waterfall_examples(
    y_val_prio, y_pred_prio, y_proba_prio,
    target_class=CRITICAL_IDX, target_class_name="Critical",
)

# ---------------------------------------------------------------------------
# Cell 8 — SHAP charts (6 total; plt.close("all") after each)
# ---------------------------------------------------------------------------
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Mean |SHAP| per feature averaged over all classes — for bar chart and metrics
# shap_values_prio is (847, 17, 4) after normalisation above
mean_abs_shap_prio = np.abs(shap_values_prio).mean(axis=(0, 2))  # (17,)
mean_abs_shap_reg  = np.abs(shap_values_reg).mean(axis=0)         # (13,)

def _shap_beeswarm(shap_vals, X, feature_names, title, save_path: Path) -> Path:
    """Save a SHAP beeswarm (summary) plot."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X, feature_names=feature_names, show=False)
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return save_path

def _shap_bar(shap_vals, X, feature_names, title, save_path: Path) -> Path:
    """Save a SHAP mean-|SHAP| bar plot (pass full shap_values list for multi-class)."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return save_path

def _shap_waterfall(explainer, shap_vals_row, X_row, feature_names, title, save_path: Path) -> Path:
    """Save a SHAP waterfall plot for a single prediction."""
    # expected_value is scalar for regressor, array-of-classes for multi-class classifier
    if hasattr(explainer.expected_value, "__len__"):
        base_val = float(explainer.expected_value[CRITICAL_IDX])
    else:
        base_val = float(explainer.expected_value)
    exp = shap.Explanation(
        values=shap_vals_row,
        base_values=base_val,
        data=X_row,
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False)
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return save_path

# shap_values_prio shape: (847, 17, 4) — index as [:, :, class_idx] for single class

# Chart 1: priority beeswarm — Critical class (index 0)
chart_prio_beeswarm = _shap_beeswarm(
    shap_values_prio[:, :, CRITICAL_IDX], X_val_full, prio_col_names,
    f"SHAP Beeswarm — Priority: {priority_names[CRITICAL_IDX]} class",
    CHARTS_DIR / "shap_priority_beeswarm.png",
)

# Chart 2: priority bar — mean |SHAP| over all 4 classes
# summary_plot with plot_type='bar' expects list of (N, F) arrays (one per class)
shap_values_prio_list = [shap_values_prio[:, :, i] for i in range(shap_values_prio.shape[2])]
chart_prio_bar = _shap_bar(
    shap_values_prio_list, X_val_full, prio_col_names,
    "SHAP Feature Importance — Priority Classifier (all classes)",
    CHARTS_DIR / "shap_priority_bar.png",
)

# Chart 3: waterfall — correctly-classified Critical ticket
chart_wf_correct = _shap_waterfall(
    explainer_prio,
    shap_values_prio[correct_idx, :, CRITICAL_IDX],   # (17,) SHAP values for this sample
    X_val_full[correct_idx],
    prio_col_names,
    f"SHAP Waterfall — Correct Critical Prediction (idx={correct_idx})",
    CHARTS_DIR / "shap_priority_waterfall_correct.png",
)

# Chart 4: waterfall — high-confidence misclassification
chart_wf_wrong = _shap_waterfall(
    explainer_prio,
    shap_values_prio[wrong_idx, :, CRITICAL_IDX],
    X_val_full[wrong_idx],
    prio_col_names,
    f"SHAP Waterfall — Misclassified Critical (pred={priority_names[int(y_pred_prio[wrong_idx])]}, idx={wrong_idx})",
    CHARTS_DIR / "shap_priority_waterfall_wrong.png",
)

# Chart 5: regressor beeswarm
chart_reg_beeswarm = _shap_beeswarm(
    shap_values_reg, X_val_reg, reg_col_names,
    "SHAP Beeswarm — Hours-to-Resolve Regressor",
    CHARTS_DIR / "shap_regressor_beeswarm.png",
)

# Chart 6: regressor bar
chart_reg_bar = _shap_bar(
    shap_values_reg, X_val_reg, reg_col_names,
    "SHAP Feature Importance — Resolution Time Regressor",
    CHARTS_DIR / "shap_regressor_bar.png",
)

logger.info("Saved 6 SHAP charts to %s", CHARTS_DIR)

# ---------------------------------------------------------------------------
# Cell 9 — Save explainers (atomic)
# ---------------------------------------------------------------------------
def _atomic_dump(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    joblib.dump(obj, tmp)
    tmp.replace(path)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
_atomic_dump(explainer_prio, SHAP_EXPLAINER_PRIORITY_PATH)
_atomic_dump(explainer_reg,  SHAP_EXPLAINER_REGRESSOR_PATH)
logger.info("Saved priority explainer → %s", SHAP_EXPLAINER_PRIORITY_PATH)
logger.info("Saved regressor explainer → %s", SHAP_EXPLAINER_REGRESSOR_PATH)

# ---------------------------------------------------------------------------
# Cell 10 — MLflow (experiment "csip-explainability", 2 runs)
# ---------------------------------------------------------------------------
_mlflow_uri = MLFLOW_TRACKING_URI
try:
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-explainability")
    logger.info("MLflow connected: %s", _mlflow_uri)
except Exception:
    _mlflow_uri = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("csip-explainability")
    logger.warning("MLflow server unavailable -- using local file store: %s", _mlflow_uri)

# Run 1: priority classifier
with mlflow.start_run(run_name="shap_lgbm_priority"):
    mlflow.log_params({
        "model":                    clf.model_name,
        "task":                     clf.task,
        "n_samples":                str(X_val_full.shape[0]),
        "n_features":               str(X_val_full.shape[1]),
        "explainer":                "TreeExplainer",
        "waterfall_target_class":   "Critical",
        "waterfall_target_idx":     str(CRITICAL_IDX),
        "waterfall_correct_idx":    str(correct_idx),
        "waterfall_wrong_idx":      str(wrong_idx),
        "waterfall_correct_prob":   f"{float(y_proba_prio[correct_idx, CRITICAL_IDX]):.4f}",
        "waterfall_wrong_pred":     priority_names[int(y_pred_prio[wrong_idx])],
        "fast_mode":                str(FAST_MODE),
    })
    for name, val in zip(prio_col_names, mean_abs_shap_prio):
        mlflow.log_metric(f"mean_abs_shap_{name}", round(float(val), 6))
    for chart in [chart_prio_beeswarm, chart_prio_bar, chart_wf_correct, chart_wf_wrong]:
        mlflow.log_artifact(str(chart))
    logger.info("MLflow run 'shap_lgbm_priority' logged")

# Run 2: regressor
with mlflow.start_run(run_name="shap_lgbm_regressor"):
    mlflow.log_params({
        "model":      reg.model_name,
        "task":       reg.task,
        "n_samples":  str(X_val_reg.shape[0]),
        "n_features": str(X_val_reg.shape[1]),
        "explainer":  "TreeExplainer",
        "fast_mode":  str(FAST_MODE),
    })
    for name, val in zip(reg_col_names, mean_abs_shap_reg):
        mlflow.log_metric(f"mean_abs_shap_{name}", round(float(val), 6))
    for chart in [chart_reg_beeswarm, chart_reg_bar]:
        mlflow.log_artifact(str(chart))
    logger.info("MLflow run 'shap_lgbm_regressor' logged")

# ---------------------------------------------------------------------------
# Cell 11 — Save metrics JSON
# ---------------------------------------------------------------------------
def _top_features(col_names: list[str], mean_abs: np.ndarray) -> list[dict]:
    pairs = sorted(zip(col_names, mean_abs.tolist()), key=lambda x: -x[1])
    return [{"feature": n, "mean_abs_shap": round(float(v), 6)} for n, v in pairs]

section_08c_metrics = {
    "section":       "8c",
    "fast_mode":     FAST_MODE,
    "generated_at":  datetime.datetime.now().isoformat(),
    "priority_explainer": {
        "model":        clf.model_name,
        "task":         clf.task,
        "n_samples":    int(X_val_full.shape[0]),
        "n_features":   int(X_val_full.shape[1]),
        "top_features": _top_features(prio_col_names, mean_abs_shap_prio),
    },
    "regressor_explainer": {
        "model":        reg.model_name,
        "task":         reg.task,
        "n_samples":    int(X_val_reg.shape[0]),
        "n_features":   int(X_val_reg.shape[1]),
        "top_features": _top_features(reg_col_names, mean_abs_shap_reg),
    },
}

metrics_dir = ARTIFACTS_DIR / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
metrics_path = metrics_dir / "section_08c_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(section_08c_metrics, fh, indent=2, cls=NumpyEncoder)
logger.info("Saved metrics → %s", metrics_path)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SECTION 8c COMPLETE — SHAP Explainability")
logger.info("Priority top feature: %s (%.4f)",
            section_08c_metrics["priority_explainer"]["top_features"][0]["feature"],
            section_08c_metrics["priority_explainer"]["top_features"][0]["mean_abs_shap"])
logger.info("Regressor top feature: %s (%.4f)",
            section_08c_metrics["regressor_explainer"]["top_features"][0]["feature"],
            section_08c_metrics["regressor_explainer"]["top_features"][0]["mean_abs_shap"])
logger.info("MLflow experiment: csip-explainability (2 runs)")
logger.info("=" * 60)
