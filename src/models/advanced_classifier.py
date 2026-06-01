# filename: src/models/advanced_classifier.py
# purpose:  AdvancedClassifier wrapper (RF, XGBoost, LightGBM) with Optuna-compatible
#           feature_schema + n_features_in tracking for safe FastAPI serving.
# version:  1.0

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import ARTIFACTS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass(repr=False, eq=False)
class AdvancedClassifier:
    """
    Thin wrapper around a fitted sklearn-compatible estimator.
    Tracks feature_schema and n_features_in for FastAPI input validation.
    n_features_in must be set explicitly after external fit — train() guards against
    double-fitting an already-fitted model.
    """

    model_name:    str
    task:          str
    model:         BaseEstimator
    feature_schema:str
    n_features_in: int  = 0
    best_params:   dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AdvancedClassifier("
            f"name={self.model_name!r}, task={self.task!r}, "
            f"schema={self.feature_schema!r}, n_features={self.n_features_in})"
        )

    def train(self, X, y, sample_weight=None) -> None:
        """Fit in isolation only. Use run_optuna_mlflow() for Optuna-driven training."""
        if self.n_features_in > 0:
            raise RuntimeError(
                "train() called on already-fitted AdvancedClassifier. "
                "Set n_features_in directly after external fit."
            )
        kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.model.fit(X, y, **kw)
        self.n_features_in = X.shape[1]

    def evaluate(self, X, y, split_name: str) -> dict[str, float | None]:
        """Evaluate on a split. Returns None (not nan) for uncomputable roc_auc — JSON-safe."""
        y_pred = self.model.predict(X)
        metrics: dict[str, float | None] = {
            f"{split_name}_f1_macro":    round(f1_score(y, y_pred, average="macro",    zero_division=0), 6),
            f"{split_name}_f1_weighted": round(f1_score(y, y_pred, average="weighted", zero_division=0), 6),
            f"{split_name}_accuracy":    round(accuracy_score(y, y_pred), 6),
        }

        roc_auc: float | None = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba  = self.model.predict_proba(X)
                present  = set(map(int, np.unique(y)))
                expected = set(range(len(self.model.classes_)))
                if present == expected:
                    roc_auc = round(
                        roc_auc_score(y, y_proba, multi_class="ovr", average="macro"), 6
                    )
                else:
                    logger.warning(
                        "%s %s: roc_auc skipped — missing classes %s",
                        split_name, self.model_name, expected - present,
                    )
            except Exception as exc:
                logger.warning("%s roc_auc failed: %s", self.model_name, exc)
        metrics[f"{split_name}_roc_auc"] = roc_auc
        return metrics

    def save(self, path: Path) -> None:
        """Atomic write: tmp -> replace. Path.replace() overwrites on Windows; rename() does not."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            joblib.dump(self, tmp)
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        logger.info("Saved %s", self)

    @classmethod
    def load(cls, path: Path | str) -> "AdvancedClassifier":
        """Load with path-safety check. Uses resolve() + startswith for Windows path compat."""
        path = Path(path).resolve()
        allowed = [MODELS_DIR.resolve(), ARTIFACTS_DIR.resolve()]
        if not any(str(path).startswith(str(d)) for d in allowed):
            raise PermissionError(f"Refusing to load from untrusted path: {path}")
        if not path.exists():
            raise FileNotFoundError(f"No artifact at {path}")

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        if not hasattr(obj, "feature_schema"):
            raise ValueError(f"Artifact at {path} missing 'feature_schema'. Re-run S6.")
        if obj.n_features_in == 0:
            raise ValueError(f"Artifact at {path} has n_features_in=0. Re-run S6.")
        return obj
