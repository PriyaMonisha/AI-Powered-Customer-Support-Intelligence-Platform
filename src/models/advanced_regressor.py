# filename: src/models/advanced_regressor.py
# purpose:  AdvancedRegressor wrapper (RF, XGBoost, LightGBM) with Optuna-compatible
#           feature_schema + n_features_in tracking for safe FastAPI serving.
# version:  1.0

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import ARTIFACTS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


def _mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """MAPE excluding zero-valued targets. Returns None if all y_true == 0 (JSON-safe)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


@dataclass(repr=False, eq=False)
class AdvancedRegressor:
    """
    Thin wrapper around a fitted sklearn-compatible regressor.
    Tracks feature_schema, n_features_in, and log_transform for FastAPI input validation
    and inference-time target inversion.
    n_features_in must be set explicitly after external fit — train() guards against
    double-fitting an already-fitted model.
    """

    model_name:     str
    task:           str
    model:          RegressorMixin
    feature_schema: str
    n_features_in:  int  = 0
    best_params:    dict = field(default_factory=dict)
    log_transform:  bool = False   # if True, model predicts log1p(y); predict() applies expm1

    def __repr__(self) -> str:
        return (
            f"AdvancedRegressor("
            f"name={self.model_name!r}, task={self.task!r}, "
            f"schema={self.feature_schema!r}, n_features={self.n_features_in}, "
            f"log_transform={self.log_transform})"
        )

    def train(self, X, y) -> None:
        """Fit in isolation only. Use run_optuna_mlflow_reg() for Optuna-driven training."""
        if self.n_features_in > 0:
            raise RuntimeError(
                "train() called on already-fitted AdvancedRegressor. "
                "Set n_features_in directly after external fit."
            )
        self.model.fit(X, y)
        self.n_features_in = X.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with feature count validation and optional log-space inversion."""
        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"Model '{self.model_name}' expects {self.n_features_in} features "
                f"(constant columns were dropped in S7), got {X.shape[1]}. "
                f"Apply keep_mask before calling predict()."
            )
        pred = self.model.predict(X)
        if self.log_transform:
            pred = np.clip(np.expm1(pred), 0, None)
        return pred

    def evaluate(
        self,
        X: np.ndarray,
        y_true_raw: np.ndarray,
        split_name: str,
        log_transform: bool = False,
    ) -> dict[str, float | None]:
        """
        Evaluate on a split. Always receives raw (non-transformed) y_true.
        If log_transform=True, predictions are inverted (expm1 + clip) before all metrics.
        Returns None (not nan) for MAPE when all y_true == 0 — JSON-safe.
        """
        pred = self.model.predict(X)
        if log_transform:
            pred = np.clip(np.expm1(pred), 0, None)

        rmse  = float(mean_squared_error(y_true_raw, pred) ** 0.5)
        mae   = float(mean_absolute_error(y_true_raw, pred))
        r2    = float(r2_score(y_true_raw, pred))
        mape  = _mape_safe(y_true_raw, pred)
        # RMSLE: both sides in real space after inversion above
        rmsle = float(np.sqrt(np.mean(
            (np.log1p(np.clip(pred, 0, None)) - np.log1p(y_true_raw)) ** 2
        )))

        result: dict[str, float | None] = {
            f"{split_name}_rmse":  round(rmse,  6),
            f"{split_name}_mae":   round(mae,   6),
            f"{split_name}_r2":    round(r2,    6),
            f"{split_name}_mape":  round(mape,  6) if mape is not None else None,
            f"{split_name}_rmsle": round(rmsle, 6),
        }
        return result

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
    def load(cls, path: Path | str) -> "AdvancedRegressor":
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
            raise ValueError(f"Artifact at {path} missing 'feature_schema'. Re-run S7.")
        if obj.n_features_in == 0:
            raise ValueError(f"Artifact at {path} has n_features_in=0. Re-run S7.")
        return obj
