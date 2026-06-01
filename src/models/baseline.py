# filename: src/models/baseline.py
# purpose:  BaselineClassifier wrapper (LR, NB, DT) with MLflow-ready evaluate/save/load
# version:  1.0

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from config import ARTIFACTS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass(repr=False, eq=False)
class BaselineClassifier:
    """Thin wrapper around a sklearn estimator — tracks feature_schema for serving."""

    model_name: str
    task: str
    model: BaseEstimator
    feature_schema: str  # "combined_sparse"|"tfidf_only"|"tabular_only"|"combined_dense"
    n_features_in: int = 0  # set after fit in train()

    def __repr__(self) -> str:
        return (
            f"BaselineClassifier("
            f"name={self.model_name!r}, task={self.task!r}, "
            f"schema={self.feature_schema!r}, n_features={self.n_features_in})"
        )

    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)
        self.n_features_in = X_train.shape[1]

    def evaluate(self, X, y, split_name: str) -> dict[str, float | None]:
        """Evaluate on a split. Returns None (not nan) for missing roc_auc — JSON-safe."""
        y_pred = self.model.predict(X)
        metrics: dict[str, float | None] = {
            f"{split_name}_f1_macro": round(
                f1_score(y, y_pred, average="macro", zero_division=0), 6
            ),
            f"{split_name}_f1_weighted": round(
                f1_score(y, y_pred, average="weighted", zero_division=0), 6
            ),
            f"{split_name}_accuracy": round(accuracy_score(y, y_pred), 6),
        }

        roc_auc: float | None = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(X)
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
        """Atomic write — tmp → rename (same pattern as TextPreprocessor.save)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            joblib.dump(self, tmp)
            tmp.replace(path)  # replace() overwrites atomically on Windows; rename() does not
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        logger.info("Saved %s", self)

    @classmethod
    def load(cls, path: Path) -> "BaselineClassifier":
        """Load with path-safety check (same pattern as TextPreprocessor.load)."""
        path = Path(path).resolve()
        allowed = [Path(MODELS_DIR).resolve(), Path(ARTIFACTS_DIR).resolve()]
        for root in allowed:
            try:
                path.relative_to(root)
                break
            except ValueError:
                continue
        else:
            raise PermissionError(f"Refusing to load from untrusted path: {path}")

        if not path.exists():
            raise FileNotFoundError(f"No artifact at {path}")

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        if not hasattr(obj, "feature_schema"):
            raise ValueError(
                f"Artifact at {path} missing 'feature_schema'. Re-run Section 5."
            )
        if not hasattr(obj, "n_features_in") or obj.n_features_in == 0:
            raise ValueError(
                f"Artifact at {path} has n_features_in=0. Re-run Section 5."
            )
        return obj


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    title: str,
    save_path: Path,
) -> None:
    """Save a Blues confusion matrix PNG. class_names must come from le.classes_."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    # labels= forces (n_classes × n_classes) even if a class has 0 samples in split
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", save_path)
