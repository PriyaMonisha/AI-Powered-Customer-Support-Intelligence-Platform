# filename: notebooks/05_baseline.py
# purpose:  Section 5 — Baseline classifiers (LR, NB, DT) + first real MLflow runs
# version:  1.0

# %% Cell 1 — FAST_MODE (must be first line of code)
FAST_MODE    = True
LR_MAX_ITER  = 200  if FAST_MODE else 1000
DT_MAX_DEPTH = 5    if FAST_MODE else 15   # 15 not None — prevents memorization artifact

# %% Cell 2 — Imports + PROJECT_ROOT
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
import mlflow.sklearn
import numpy as np
import scipy.sparse as sp
from mlflow.models.signature import infer_signature
from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from config import (
    ARTIFACTS_DIR,
    BASELINE_PRIORITY_PATH,
    BASELINE_TYPE_PATH,
    CHARTS_DIR,
    FEATURES_DIR,
    LE_PRIORITY_PATH,
    LE_TYPE_PATH,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.models.baseline import BaselineClassifier, plot_confusion_matrix
from src.utils.helpers import NumpyEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("section5")
logger.info("FAST_MODE=%s  LR_MAX_ITER=%d  DT_MAX_DEPTH=%s",
            FAST_MODE, LR_MAX_ITER, DT_MAX_DEPTH)

# %% Cell 3 — Load + validate feature artifacts
logger.info("Loading feature artifacts from %s ...", FEATURES_DIR)

X_tab_train   = np.load(FEATURES_DIR / "X_train_tabular.npy")
X_tab_val     = np.load(FEATURES_DIR / "X_val_tabular.npy")
X_tab_test    = np.load(FEATURES_DIR / "X_test_tabular.npy")

X_tfidf_train = sp.load_npz(str(FEATURES_DIR / "X_train_tfidf.npz"))
X_tfidf_val   = sp.load_npz(str(FEATURES_DIR / "X_val_tfidf.npz"))
X_tfidf_test  = sp.load_npz(str(FEATURES_DIR / "X_test_tfidf.npz"))

y_train_type  = np.load(FEATURES_DIR / "y_train_type.npy")
y_val_type    = np.load(FEATURES_DIR / "y_val_type.npy")
y_test_type   = np.load(FEATURES_DIR / "y_test_type.npy")

y_train_prio  = np.load(FEATURES_DIR / "y_train_prio.npy")
y_val_prio    = np.load(FEATURES_DIR / "y_val_prio.npy")
y_test_prio   = np.load(FEATURES_DIR / "y_test_prio.npy")

# Row-count assertions — catch artifact mismatches before training
for split_name, tab, tfidf, y_t, y_p in [
    ("train", X_tab_train, X_tfidf_train, y_train_type, y_train_prio),
    ("val",   X_tab_val,   X_tfidf_val,   y_val_type,   y_val_prio),
    ("test",  X_tab_test,  X_tfidf_test,  y_test_type,  y_test_prio),
]:
    assert tab.shape[0] == tfidf.shape[0], \
        f"{split_name}: tabular rows {tab.shape[0]} != tfidf rows {tfidf.shape[0]}"
    assert tab.shape[0] == y_t.shape[0], \
        f"{split_name}: X rows {tab.shape[0]} != y_type rows {y_t.shape[0]}"
    assert tab.shape[0] == y_p.shape[0], \
        f"{split_name}: X rows {tab.shape[0]} != y_prio rows {y_p.shape[0]}"
    logger.info("%s: tab=%s  tfidf=%s  y_type=%s  y_prio=%s",
                split_name, tab.shape, tfidf.shape, y_t.shape, y_p.shape)

# Load label encoders (plain sklearn — use joblib.load, not a custom .load() method)
le_type = joblib.load(LE_TYPE_PATH)
le_prio = joblib.load(LE_PRIORITY_PATH)
assert isinstance(le_type, LabelEncoder) and hasattr(le_type, "classes_"), \
    "le_type is not a fitted LabelEncoder"
assert isinstance(le_prio, LabelEncoder) and hasattr(le_prio, "classes_"), \
    "le_prio is not a fitted LabelEncoder"

# str() cast: le.classes_ returns numpy.str_ — convert to Python str for JSON/seaborn safety
type_class_names = [str(c) for c in le_type.classes_]
prio_class_names = [str(c) for c in le_prio.classes_]
logger.info("Type classes (%d):     %s", len(type_class_names), type_class_names)
logger.info("Priority classes (%d): %s", len(prio_class_names), prio_class_names)

# %% Cell 4 — Build feature matrices
logger.info("Building feature matrices ...")

X_combined_train = sp.hstack([sp.csr_matrix(X_tab_train), X_tfidf_train])
X_combined_val   = sp.hstack([sp.csr_matrix(X_tab_val),   X_tfidf_val])
X_combined_test  = sp.hstack([sp.csr_matrix(X_tab_test),  X_tfidf_test])

# Dense conversion for DT (train≈55MB, val≈7.8MB, test≈15.6MB — all safe)
X_dt_train = X_combined_train.toarray()
X_dt_val   = X_combined_val.toarray()
X_dt_test  = X_combined_test.toarray()

# Model-specific matrix lookup keyed by feature_schema
FEATURE_MATRICES: dict[str, tuple] = {
    "combined_sparse": (X_combined_train, X_combined_val, X_combined_test),
    "tfidf_only":      (X_tfidf_train,    X_tfidf_val,    X_tfidf_test),
    "tabular_only":    (X_tab_train,      X_tab_val,      X_tab_test),
    "combined_dense":  (X_dt_train,       X_dt_val,       X_dt_test),
}

TASK_LABELS: dict[str, tuple] = {
    "ticket_type":     (y_train_type, y_val_type, y_test_type),
    "ticket_priority": (y_train_prio, y_val_prio, y_test_prio),
}

logger.info("Combined sparse shape: train=%s  val=%s  test=%s",
            X_combined_train.shape, X_combined_val.shape, X_combined_test.shape)
logger.info("Combined dense shape:  train=%s  val=%s  test=%s",
            X_dt_train.shape, X_dt_val.shape, X_dt_test.shape)

# %% Cell 5 — MLflow setup + accumulators
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("csip-baseline-classifiers")
logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)

all_run_results: list[dict] = []
best_type: tuple[str, float, BaselineClassifier | None] = ("", -1.0, None)
best_prio: tuple[str, float, BaselineClassifier | None] = ("", -1.0, None)

# %% Cell 6 — Ticket Type classifiers (3 MLflow runs)
TYPE_CONFIGS = [
    (
        "lr", "ticket_type", "combined_sparse",
        LogisticRegression(
            class_weight="balanced", max_iter=LR_MAX_ITER,
            random_state=RANDOM_STATE, solver="saga", multi_class="multinomial",
        ),
    ),
    (
        "nb", "ticket_type", "tfidf_only",
        ComplementNB(),
    ),
    (
        "dt", "ticket_type", "combined_dense",
        DecisionTreeClassifier(
            class_weight="balanced", max_depth=DT_MAX_DEPTH,
            random_state=RANDOM_STATE,
        ),
    ),
]

logger.info("=" * 60)
logger.info("TICKET TYPE CLASSIFIERS")
logger.info("=" * 60)

for algo, task, schema, estimator in TYPE_CONFIGS:
    clf = BaselineClassifier(
        model_name=algo, task=task, model=estimator, feature_schema=schema
    )
    X_tr, X_vl, X_te = FEATURE_MATRICES[clf.feature_schema]
    y_tr, y_vl, y_te = TASK_LABELS[task]
    class_names       = type_class_names

    with mlflow.start_run(run_name=f"{algo}_{task}") as run:
        try:
            logger.info("Training %s/%s ...", algo, task)

            # warnings.catch_warnings modifies global state — safe for single-threaded notebook
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clf.train(X_tr, y_tr)

            convergence_w = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
            other_w       = [w for w in caught if not issubclass(w.category, ConvergenceWarning)]
            converged = len(convergence_w) == 0
            if not converged:
                logger.warning("LR %s did not converge in %d iterations", task, LR_MAX_ITER)
            for w in other_w[:3]:
                logger.debug("Training warning (%s): %s", w.category.__name__, w.message)

            val_metrics  = clf.evaluate(X_vl, y_vl, "val")
            test_metrics = clf.evaluate(X_te, y_te, "test")
            all_run_metrics = {**val_metrics, **test_metrics}

            mlflow.log_params({
                "algorithm":              algo,
                "task":                   task,
                "feature_schema":         clf.feature_schema,
                "n_features_in":          str(clf.n_features_in),
                "class_weight":           "balanced" if algo != "nb" else "n/a",
                "fast_mode":              str(FAST_MODE),
                "converged":              str(converged),
                "n_convergence_warnings": str(len(convergence_w)),
                "n_iter_actual":          str(getattr(clf.model, "n_iter_", ["n/a"])[0]),
                "max_depth_requested":    str(DT_MAX_DEPTH) if algo == "dt" else "n/a",
                "max_depth_actual":       (
                    str(clf.model.get_depth())
                    if algo == "dt" and hasattr(clf.model, "tree_")
                    else "n/a"
                ),
                "nb_assumption_note":     "n/a",
            })

            # Filter None before log_metrics — MLflow rejects None/NaN
            safe_metrics = {k: v for k, v in all_run_metrics.items() if v is not None}
            mlflow.log_metrics(safe_metrics)

            # infer_signature — use model's own feature matrix; convert sparse→dense for schema
            # Signature uses dense: JSON payloads are always dense at serve time.
            # LR/DT accept dense input despite training on sparse.
            sample_raw   = X_vl[:5]
            sample_dense = sample_raw.toarray() if sp.issparse(sample_raw) else sample_raw
            signature    = infer_signature(sample_dense, clf.model.predict(sample_raw))
            mlflow.sklearn.log_model(
                clf.model, "model",
                signature=signature,
                input_example=sample_dense[:2],
            )

            CHARTS_DIR.mkdir(parents=True, exist_ok=True)
            png_path = CHARTS_DIR / f"cm_{algo}_{task}.png"
            plot_confusion_matrix(
                y_te, clf.model.predict(X_te),
                class_names=class_names,
                title=f"Confusion Matrix — {algo.upper()} / Ticket Type",
                save_path=png_path,
            )
            mlflow.log_artifact(str(png_path))

            logger.info(
                "Run %s OK: %s/%s  val_f1_macro=%.4f  val_roc_auc=%s",
                run.info.run_id[:8], algo, task,
                val_metrics.get("val_f1_macro") or -1.0,
                f"{val_metrics.get('val_roc_auc'):.4f}" if val_metrics.get("val_roc_auc") is not None else "n/a",
            )

            all_run_results.append({
                "algo": algo, "task": task,
                "feature_schema": clf.feature_schema,
                "n_features_in": clf.n_features_in,
                **all_run_metrics,
            })

            val_f1 = val_metrics.get("val_f1_macro") or -1.0
            if val_f1 > best_type[1]:
                best_type = (algo, val_f1, clf)

        except Exception as exc:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc)[:500])
            logger.error("Run FAILED %s/%s: %s", algo, task, exc)
            raise

# %% Cell 7 — Ticket Priority classifiers (3 MLflow runs)
PRIO_CONFIGS = [
    (
        "lr", "ticket_priority", "tabular_only",
        LogisticRegression(
            class_weight="balanced", max_iter=LR_MAX_ITER,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "nb", "ticket_priority", "tabular_only",
        GaussianNB(),
    ),
    (
        "dt", "ticket_priority", "tabular_only",
        DecisionTreeClassifier(
            class_weight="balanced", max_depth=DT_MAX_DEPTH,
            random_state=RANDOM_STATE,
        ),
    ),
]

logger.info("=" * 60)
logger.info("TICKET PRIORITY CLASSIFIERS")
logger.info("=" * 60)

for algo, task, schema, estimator in PRIO_CONFIGS:
    clf = BaselineClassifier(
        model_name=algo, task=task, model=estimator, feature_schema=schema
    )
    X_tr, X_vl, X_te = FEATURE_MATRICES[clf.feature_schema]
    y_tr, y_vl, y_te = TASK_LABELS[task]
    class_names       = prio_class_names

    with mlflow.start_run(run_name=f"{algo}_{task}") as run:
        try:
            logger.info("Training %s/%s ...", algo, task)

            if algo == "nb":
                logger.warning(
                    "GaussianNB: response_hour_of_day has -1 sentinel -- "
                    "violates Gaussian assumption. ROC-AUC may be underestimated."
                )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clf.train(X_tr, y_tr)

            convergence_w = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
            other_w       = [w for w in caught if not issubclass(w.category, ConvergenceWarning)]
            converged = len(convergence_w) == 0
            if not converged:
                logger.warning("LR %s did not converge in %d iterations", task, LR_MAX_ITER)
            for w in other_w[:3]:
                logger.debug("Training warning (%s): %s", w.category.__name__, w.message)

            val_metrics  = clf.evaluate(X_vl, y_vl, "val")
            test_metrics = clf.evaluate(X_te, y_te, "test")
            all_run_metrics = {**val_metrics, **test_metrics}

            mlflow.log_params({
                "algorithm":              algo,
                "task":                   task,
                "feature_schema":         clf.feature_schema,
                "n_features_in":          str(clf.n_features_in),
                "class_weight":           "balanced" if algo != "nb" else "n/a",
                "fast_mode":              str(FAST_MODE),
                "converged":              str(converged),
                "n_convergence_warnings": str(len(convergence_w)),
                "n_iter_actual":          str(getattr(clf.model, "n_iter_", ["n/a"])[0]),
                "max_depth_requested":    str(DT_MAX_DEPTH) if algo == "dt" else "n/a",
                "max_depth_actual":       (
                    str(clf.model.get_depth())
                    if algo == "dt" and hasattr(clf.model, "tree_")
                    else "n/a"
                ),
                "nb_assumption_note":     (
                    "-1 sentinel in response_hour_of_day violates Gaussian assumption"
                    if algo == "nb"
                    else "n/a"
                ),
            })

            safe_metrics = {k: v for k, v in all_run_metrics.items() if v is not None}
            mlflow.log_metrics(safe_metrics)

            sample_raw   = X_vl[:5]
            sample_dense = sample_raw.toarray() if sp.issparse(sample_raw) else sample_raw
            signature    = infer_signature(sample_dense, clf.model.predict(sample_raw))
            mlflow.sklearn.log_model(
                clf.model, "model",
                signature=signature,
                input_example=sample_dense[:2],
            )

            CHARTS_DIR.mkdir(parents=True, exist_ok=True)
            png_path = CHARTS_DIR / f"cm_{algo}_{task}.png"
            plot_confusion_matrix(
                y_te, clf.model.predict(X_te),
                class_names=class_names,
                title=f"Confusion Matrix — {algo.upper()} / Ticket Priority",
                save_path=png_path,
            )
            mlflow.log_artifact(str(png_path))

            logger.info(
                "Run %s OK: %s/%s  val_f1_macro=%.4f  val_roc_auc=%s",
                run.info.run_id[:8], algo, task,
                val_metrics.get("val_f1_macro") or -1.0,
                f"{val_metrics.get('val_roc_auc'):.4f}" if val_metrics.get("val_roc_auc") is not None else "n/a",
            )

            all_run_results.append({
                "algo": algo, "task": task,
                "feature_schema": clf.feature_schema,
                "n_features_in": clf.n_features_in,
                **all_run_metrics,
            })

            val_f1 = val_metrics.get("val_f1_macro") or -1.0
            if val_f1 > best_prio[1]:
                best_prio = (algo, val_f1, clf)

        except Exception as exc:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc)[:500])
            logger.error("Run FAILED %s/%s: %s", algo, task, exc)
            raise

# %% Cell 8 — Free dense arrays after DT training
# Defensive delete — no NameError if arrays were never created on partial failure
for _name in [
    "X_dt_train", "X_dt_val", "X_dt_test",
    "X_combined_train", "X_combined_val", "X_combined_test",
]:
    try:
        del globals()[_name]
    except KeyError:
        pass
gc.collect()
logger.info("Dense arrays freed")

# %% Cell 9 — Save metrics JSON + best models
# MLflow is the authoritative record. This JSON is a convenience cache.
# Re-running overwrites it — check MLflow for historical runs.
metrics_dir = ARTIFACTS_DIR / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "runs": all_run_results,
    "_notes": {
        "roc_auc_null": "null means metric was not computed (missing classes or no predict_proba)",
        "source_of_truth": "MLflow experiment csip-baseline-classifiers",
    },
}
with open(metrics_dir / "section_05_metrics.json", "w") as fh:
    json.dump(payload, fh, cls=NumpyEncoder, indent=2)
logger.info("Saved %s", metrics_dir / "section_05_metrics.json")

if best_type[2] is not None:
    best_type[2].save(BASELINE_TYPE_PATH)
    logger.info("Best type model saved: %s (val_f1_macro=%.4f)", best_type[0], best_type[1])
else:
    logger.error("No best type model found — all Ticket Type runs failed")

if best_prio[2] is not None:
    best_prio[2].save(BASELINE_PRIORITY_PATH)
    logger.info("Best priority model saved: %s (val_f1_macro=%.4f)", best_prio[0], best_prio[1])
else:
    logger.error("No best priority model found — all Ticket Priority runs failed")

# %% Cell 10 — Summary (ASCII only — no Unicode, Windows cp1252 safe)
logger.info("=" * 60)
logger.info("SECTION 5 COMPLETE")
logger.info("MLflow experiment: csip-baseline-classifiers (%d runs)", len(all_run_results))
logger.info("-" * 60)
logger.info("Best Ticket Type classifier:     %s (val F1-macro=%.4f)",
            best_type[0] or "NONE", best_type[1])
logger.info("Best Ticket Priority classifier: %s (val F1-macro=%.4f)",
            best_prio[0] or "NONE", best_prio[1])
logger.info("-" * 60)
logger.info("Artifacts saved:")
logger.info("  models/baseline_type_classifier.pkl")
logger.info("  models/baseline_priority_classifier.pkl")
logger.info("  artifacts/metrics/section_05_metrics.json")
logger.info("  artifacts/charts/cm_*.png  (6 confusion matrices)")
logger.info("-" * 60)
logger.info("Next step: open http://localhost:5001 to view MLflow runs")
logger.info("Then run: python notebooks/06_advanced_ml.py")
logger.info("=" * 60)
