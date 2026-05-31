# filename: src/features/tabular_features.py
# purpose:  Tabular feature encoding for customer support tickets.
#           Explicit feature lists prevent response_hour_of_day from being
#           accidentally encoded as categorical (it's numeric with -1 sentinel).
# version:  1.0

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import TargetEncoder

log = logging.getLogger(__name__)

# ── Feature lists (explicit — no catch-all select_dtypes loops) ──────────────

ORDINAL_FEATURES = [
    "Ticket Channel",    # Email / Chat / Phone / Social media
    "Customer Gender",   # Male / Female / Other
]

TARGET_ENC_FEATURES = [
    "Product Purchased",  # high-cardinality; TargetEncoder with handle_unknown='value'
]

# response_hour_of_day: -1 sentinel = no first response yet (open tickets).
# Kept numeric — XGBoost/LightGBM split on -1 naturally (means "no response").
# DO NOT add to ORDINAL_FEATURES or TARGET_ENC_FEATURES — that conflates
# "no response" with the lowest ordinal rank, which is semantically wrong.
NUMERIC_PASSTHROUGH = [
    "Customer Age",
    "days_since_purchase",
    "response_hour_of_day",   # -1 = no response; numeric sentinel, not a category
    "is_resolved",
    "has_first_response",
    "csat_available",
    # Text meta-features (added in Section 3, passed through here):
    "char_count",
    "word_count",
    "subject_word_count",
    "sentiment_compound",
]

# All tabular columns that go into the feature matrix (order matters for column alignment)
ALL_TABULAR_FEATURES = ORDINAL_FEATURES + TARGET_ENC_FEATURES + NUMERIC_PASSTHROUGH


# ── Encoders ──────────────────────────────────────────────────────────────────

class TabularEncoder:
    """
    Fits OrdinalEncoder + TargetEncoder on training data.
    Transforms any split (train/val/test) consistently.
    Numeric passthrough columns are appended as-is.
    """

    def __init__(self) -> None:
        self.ordinal_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,       # unknown category at inference → -1 (not NaN crash)
            encoded_missing_value=-2,
        )
        self.target_enc = TargetEncoder(
            target_type="multiclass",
            smooth="auto",
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabularEncoder":
        """Fit encoders on training data only. y = Ticket Type (primary target)."""
        self.ordinal_enc.fit(X[ORDINAL_FEATURES])
        self.target_enc.fit(X[TARGET_ENC_FEATURES], y)
        self._fitted = True
        log.info(
            "TabularEncoder fitted. Ordinal: %s | Target: %s | Passthrough: %d cols",
            ORDINAL_FEATURES, TARGET_ENC_FEATURES, len(NUMERIC_PASSTHROUGH)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame — returns encoded DataFrame with consistent column order."""
        assert self._fitted, "Call fit() before transform()"

        # Encode categorical columns
        ord_arr = self.ordinal_enc.transform(X[ORDINAL_FEATURES])
        ord_df  = pd.DataFrame(
            ord_arr,
            columns=[f"{c}_enc" for c in ORDINAL_FEATURES],
            index=X.index,
        )

        tgt_arr = self.target_enc.transform(X[TARGET_ENC_FEATURES])
        tgt_df  = pd.DataFrame(
            tgt_arr,
            columns=[f"{c}_enc" for c in TARGET_ENC_FEATURES],
            index=X.index,
        )

        # Numeric passthrough — only include columns that exist (text features added later)
        present_numeric = [c for c in NUMERIC_PASSTHROUGH if c in X.columns]
        num_df = X[present_numeric].copy()

        result = pd.concat([ord_df, tgt_df, num_df], axis=1)
        log.debug("TabularEncoder.transform: output shape %s", result.shape)
        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        log.info("TabularEncoder saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TabularEncoder":
        enc = joblib.load(path)
        log.info("TabularEncoder loaded from %s", path)
        return enc
