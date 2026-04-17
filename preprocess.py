"""Text cleaning and dataframe helpers for job posting NLP."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


def _ensure_nltk() -> None:
    import nltk

    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"):
        nltk.download(pkg, quiet=True)


_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.\w{2,}\b", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"#(?:URL|EMAIL|PHONE)_[a-f0-9]+#",
    re.IGNORECASE,
)
_NON_WORD_RE = re.compile(r"[^a-z\s]+")


_lemma_cache: dict[str, str] = {}


def clean_text(raw: str, lemmatizer: WordNetLemmatizer, stops: set[str]) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = str(raw)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = _PLACEHOLDER_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    words = text.split()
    out = []
    for w in words:
        if w in stops or len(w) < 2:
            continue
        if w not in _lemma_cache:
            _lemma_cache[w] = lemmatizer.lemmatize(w)
        out.append(_lemma_cache[w])
    return " ".join(out)


class TextCleaner(BaseEstimator, TransformerMixin):
    """ sklearn-compatible cleaner: HTML strip, URL/email removal, lower, lemma, stopwords. """

    def __init__(self) -> None:
        self._lemmatizer: WordNetLemmatizer | None = None
        self._stops: set[str] | None = None

    def fit(self, X, y=None):  # noqa: ARG002
        global _lemma_cache
        _lemma_cache.clear()
        _ensure_nltk()
        self._lemmatizer = WordNetLemmatizer()
        self._stops = set(stopwords.words("english"))
        return self

    def transform(self, X) -> np.ndarray:
        if self._lemmatizer is None or self._stops is None:
            self.fit(X)
        assert self._lemmatizer is not None and self._stops is not None
        if isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X).ravel()
        cleaned = [clean_text(t, self._lemmatizer, self._stops) for t in X]
        return np.asarray(cleaned, dtype=object)


def fill_missing(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in text_cols:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)
            out[c] = out[c].replace({"nan": "", "None": ""})
    return out


def combine_text_columns(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    parts = []
    for c in text_cols:
        if c not in df.columns:
            continue
        parts.append(df[c].astype(str).fillna(""))
    if not parts:
        raise ValueError("No text columns to combine.")
    stacked = pd.concat(parts, axis=1)
    return stacked.apply(lambda row: " ".join(row.values), axis=1)
