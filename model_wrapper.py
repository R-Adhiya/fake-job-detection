"""Wrappers: fraud-optimized threshold + cleaner + inner sklearn pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class FraudDetectionModel:
    """Delegates predict_proba to inner model; predict uses tuned threshold on P(fraud)."""

    def __init__(self, inner: Any, threshold: float = 0.5) -> None:
        self.pipeline = inner
        self.threshold = float(threshold)
        self.classes_ = getattr(inner, "classes_", np.array([0, 1]))

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        classes = np.asarray(self.pipeline.classes_)
        idx = int(np.where(classes == 1)[0][0])
        p = proba[:, idx]
        return (p >= self.threshold).astype(int)


class FullHybridModel:
    """
    Applies a fitted TextCleaner to combined_text, then runs the inner pipeline
    (TF-IDF + metadata + classifier) trained on pre-cleaned text.
    """

    def __init__(self, cleaner: Any, inner: Pipeline) -> None:
        self.cleaner = cleaner
        self.inner = inner
        self.classes_ = getattr(inner, "classes_", np.array([0, 1]))

    def _apply_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        X2 = X.copy()
        raw = X2["combined_text"].values
        X2["combined_text"] = self.cleaner.transform(raw)
        return X2

    def predict_proba(self, X):
        return self.inner.predict_proba(self._apply_clean(X))

    def predict(self, X):
        return self.inner.predict(self._apply_clean(X))
