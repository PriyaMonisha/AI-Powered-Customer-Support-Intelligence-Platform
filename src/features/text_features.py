# filename: src/features/text_features.py
# purpose:  Text preprocessing — meta-features (char/word counts, VADER sentiment)
#           and TF-IDF vectorization for ticket descriptions.
# version:  1.0

import logging
import re
from pathlib import Path
from typing import Self

import joblib
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import ARTIFACTS_DIR, MODELS_DIR, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

logger = logging.getLogger(__name__)


def _is_safe_path(p: Path, roots: list[Path]) -> bool:
    """
    Verify p is a proper child of at least one allowed root.
    Uses Path.relative_to() — not str.startswith(), which incorrectly
    accepts /app/models_evil as a child of /app/models.
    Defined at module level so it can be unit-tested independently.
    """
    for root in roots:
        try:
            p.relative_to(root)
            return True
        except ValueError:
            continue
    return False


class TextPreprocessor:
    """
    Text preprocessing for customer support tickets.

    Two-tier TF-IDF design:
      fit_tfidf_exploratory() — full corpus EDA only, never saved.
      fit_tfidf()             — production fit on TRAIN DATA ONLY (Section 4).

    save() raises RuntimeError if only exploratory fit exists — structurally
    impossible to accidentally persist an exploratory vectorizer.

    Not sklearn Pipeline compatible — use standalone.
    """

    VERSION = "1.0.0"
    _KEEP_SHORT_TOKENS = frozenset({"ui", "db", "os", "id", "ip", "ok", "no", "api"})
    # frozenset: immutable + O(1) lookup + signals "do not modify"

    def __init__(self) -> None:
        # NLTK stopwords — offline fallback to minimal set if download fails; never raises
        try:
            from nltk.corpus import stopwords
            self._stop_words = set(stopwords.words("english"))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords
                self._stop_words = set(stopwords.words("english"))
            except Exception as e:
                logger.warning(
                    "NLTK stopwords unavailable (%s). Using minimal fallback.", e
                )
                self._stop_words = {
                    "i", "me", "my", "we", "our", "you", "your", "he", "she",
                    "it", "they", "them", "the", "a", "an", "is", "are", "was",
                    "were", "be", "been", "have", "has", "had", "do", "does",
                    "did", "will", "would", "could", "should", "may", "might",
                    "to", "of", "in", "for", "on", "with", "at", "by", "from",
                    "and", "or", "but", "not", "so", "if", "as", "this", "that",
                }

        # VADER lexicon — raises on failure (no substitute exists)
        try:
            self._sid = SentimentIntensityAnalyzer()
        except LookupError:
            try:
                nltk.download("vader_lexicon", quiet=True)
                self._sid = SentimentIntensityAnalyzer()
            except Exception as e:
                raise RuntimeError(
                    f"VADER lexicon unavailable and download failed: {e}. "
                    "Run: python -c \"import nltk; nltk.download('vader_lexicon')\""
                ) from e

        self._exploratory_vectorizer: TfidfVectorizer | None = None
        self.vectorizer_: TfidfVectorizer | None = None

    # ── Meta-feature computation ──────────────────────────────────────────────

    def add_text_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns NEW DataFrame with 4 appended columns. Input unchanged.

        Columns added:
          char_count        — character count of Ticket Description (raw text)
          word_count        — word count of Ticket Description (raw text)
          subject_word_count — word count of Ticket Subject (raw text)
          sentiment_compound — VADER compound score of Ticket Description (raw text)
                               NaN for null/empty descriptions (not fake 0.0 neutral)
        """
        result = df.copy()
        desc = result["Ticket Description"].fillna("")
        subj = result["Ticket Subject"].fillna("")

        result["char_count"] = desc.str.len()
        result["word_count"] = desc.str.split().str.len().fillna(0).astype(int)
        result["subject_word_count"] = subj.str.split().str.len().fillna(0).astype(int)
        result["sentiment_compound"] = desc.apply(
            lambda t: self._sid.polarity_scores(t)["compound"] if t else float("nan")
        )
        return result

    # ── Text cleaning ─────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Lowercase, keep alphanumeric, remove stopwords. Used as TF-IDF input."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep digits: error codes, model numbers
        tokens = text.split()
        tokens = [
            t for t in tokens
            if t not in self._stop_words and (len(t) > 1 or t in self._KEEP_SHORT_TOKENS)
        ]
        return " ".join(tokens)

    def _clean_series(self, texts: pd.Series) -> pd.Series:
        return texts.fillna("").apply(self.clean_text)

    # ── Vectorizer factory ────────────────────────────────────────────────────

    def _build_vectorizer(self) -> TfidfVectorizer:
        """
        Single source of truth for TfidfVectorizer config.
        Both fit_tfidf_exploratory() and fit_tfidf() call this factory —
        any config change applies to both automatically.

        min_df=2 (integer, not float): term must appear in >= 2 documents.
        Does not scale with dataset size — if retraining DAG train split
        drops below ~200 docs, consider reducing to min_df=1.
        """
        return TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,        # log normalization — better for long support docs
            min_df=2,                 # ignore hapax legomena
            strip_accents="unicode",  # handles accented product names
        )

    # ── Fit methods ───────────────────────────────────────────────────────────

    def fit_tfidf_exploratory(self, texts: pd.Series) -> Self:
        """FOR EDA ONLY. Never save this. Section 4 fits production vectorizer on train split."""
        self._exploratory_vectorizer = self._build_vectorizer()
        self._exploratory_vectorizer.fit(self._clean_series(texts))
        logger.info(
            "Exploratory TF-IDF fitted. Vocab size: %d",
            len(self._exploratory_vectorizer.vocabulary_),
        )
        return self

    def fit_tfidf(self, texts: pd.Series) -> Self:
        """Production fit — call on TRAIN DATA ONLY in Section 4."""
        self.vectorizer_ = self._build_vectorizer()
        self.vectorizer_.fit(self._clean_series(texts))
        logger.info(
            "Production TF-IDF fitted. Vocab size: %d",
            len(self.vectorizer_.vocabulary_),
        )
        return self

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def exploratory_vocab_size(self) -> int | None:
        """Vocabulary size of exploratory vectorizer. None if not yet fitted."""
        if self._exploratory_vectorizer is None:
            return None
        return len(self._exploratory_vectorizer.vocabulary_)

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform_tfidf(self, texts: pd.Series):
        """Transform using production vectorizer. Call fit_tfidf() first."""
        if self.vectorizer_ is None:
            raise RuntimeError(
                "Production TF-IDF vectorizer not fitted. "
                "Call fit_tfidf() on train data (Section 4) before transform_tfidf()."
            )
        return self.vectorizer_.transform(self._clean_series(texts))

    # ── Top-term analysis (EDA only) ──────────────────────────────────────────

    def get_top_terms_overall(self, texts: pd.Series, n: int = 20) -> list[str]:
        """Top N terms by mean TF-IDF score across all documents."""
        if self._exploratory_vectorizer is None:
            raise RuntimeError("Call fit_tfidf_exploratory() first.")
        matrix = self._exploratory_vectorizer.transform(self._clean_series(texts))
        feature_names = self._exploratory_vectorizer.get_feature_names_out()
        mean_scores = np.asarray(matrix.mean(axis=0)).flatten()  # avoid numpy.matrix.argsort() bug
        top_indices = mean_scores.argsort()[::-1][:n]
        return [str(t) for t in feature_names[top_indices]]  # str() → JSON-safe

    def get_top_terms_per_class(
        self,
        df: pd.DataFrame,
        text_col: str,
        target_col: str,
        n: int = 15,
    ) -> dict[str, list[str]]:
        """Top N TF-IDF terms per class label. Requires fit_tfidf_exploratory() first."""
        if self._exploratory_vectorizer is None:
            raise RuntimeError("Call fit_tfidf_exploratory() first.")

        tfidf_matrix = self._exploratory_vectorizer.transform(
            self._clean_series(df[text_col])
        )
        feature_names = self._exploratory_vectorizer.get_feature_names_out()
        result: dict[str, list[str]] = {}

        for cls in sorted(df[target_col].dropna().unique()):
            mask = (df[target_col] == cls).values
            class_matrix = tfidf_matrix[mask]
            mean_scores = np.asarray(class_matrix.mean(axis=0)).flatten()
            top_indices = mean_scores.argsort()[::-1][:n]
            result[cls] = [str(t) for t in feature_names[top_indices]]

        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """
        Save production preprocessor. Raises RuntimeError if only
        exploratory vectorizer is fitted — prevents leaking full-corpus fit.
        """
        if self.vectorizer_ is None:
            raise RuntimeError(
                "No production vectorizer fitted. "
                "Call fit_tfidf() on train data first. "
                "fit_tfidf_exploratory() is NOT saved."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        joblib.dump({"version": self.VERSION, "preprocessor": self}, tmp)
        tmp.rename(path)
        logger.info("TextPreprocessor v%s saved to %s", self.VERSION, path)

    @classmethod
    def load(cls, path: Path) -> "TextPreprocessor":
        """
        Load a saved TextPreprocessor artifact.

        Path safety: loads only from MODELS_DIR or ARTIFACTS_DIR — prevents
        path traversal attacks via /admin/reload endpoint. Uses module-level
        _is_safe_path() which is independently unit-testable.

        Type check: loaded object must be instance of cls or its subclass.
        Loading base TextPreprocessor via subclass call raises TypeError.
        Loading subclass via base class call succeeds (Liskov substitution).
        Version check uses cls.VERSION — subclasses should override VERSION.
        """
        path = Path(path).resolve()
        allowed_roots = [Path(MODELS_DIR).resolve(), Path(ARTIFACTS_DIR).resolve()]

        if not _is_safe_path(path, allowed_roots):
            raise PermissionError(
                f"Refusing to load artifact from untrusted path: {path}. "
                f"Allowed: {[str(r) for r in allowed_roots]}"
            )
        if not path.exists():
            raise FileNotFoundError(f"No preprocessor artifact at {path}")

        payload = joblib.load(path)

        if not isinstance(payload, dict) or "preprocessor" not in payload:
            raise ValueError(f"Malformed artifact at {path}. Expected dict with 'preprocessor' key.")
        if not isinstance(payload["preprocessor"], cls):
            raise TypeError(
                f"Loaded object is {type(payload['preprocessor'])}, expected {cls}. "
                "Artifact may be from incompatible version."
            )
        if payload.get("version") != cls.VERSION:
            logger.warning(
                "Version mismatch: file=%s, current=%s. Proceeding with caution.",
                payload.get("version"),
                cls.VERSION,
            )

        instance = payload["preprocessor"]

        # Reinitialize _sid — VADER lexicon path may differ across machines/containers.
        # Lexicon content is identical; only the file path reference may vary.
        try:
            instance._sid = SentimentIntensityAnalyzer()
        except LookupError:
            logger.warning(
                "VADER lexicon not found on this machine. "
                "_sid not reinitialized — sentiment scoring will fail if called."
            )

        return instance
