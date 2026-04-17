"""Load model, build inputs, predictions, and explanations."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import METRICS_PATH, MODEL_PATH
from features import frame_from_user_fields
from model_wrapper import FraudDetectionModel, FullHybridModel


def load_pipeline(path: Path | None = None):
    p = path or MODEL_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Model not found at {p}. Run: python train.py")
    return joblib.load(p)


def risk_level(p_fraud: float) -> str:
    if p_fraud >= 0.7:
        return "high"
    if p_fraud >= 0.4:
        return "medium"
    return "low"


def coerce_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, str):
        return frame_from_user_fields(description=X)
    raise TypeError("Expected DataFrame or combined text string.")


def predict_proba_fraud(model, X) -> float:
    df = coerce_frame(X)
    proba = model.predict_proba(df)[0]
    inner = _base_estimator(model)
    classes = np.asarray(inner.classes_)
    idx = int(np.where(classes == 1)[0][0])
    return float(proba[idx])


def predict_label(model, X) -> int:
    df = coerce_frame(X)
    return int(model.predict(df)[0])


def _base_estimator(model):
    """Unwrap FraudDetectionModel / FullHybridModel to sklearn estimator with classes_."""
    if isinstance(model, FraudDetectionModel):
        m = model.pipeline
    else:
        m = model
    if isinstance(m, FullHybridModel):
        return m.inner
    return m


def explain(model, X) -> dict:
    df = coerce_frame(X)
    out: dict = {"top_features": [], "top_metadata": []}

    if isinstance(model, FraudDetectionModel):
        core = model.pipeline
    else:
        core = model

    if isinstance(core, FullHybridModel):
        df_u = core._apply_clean(df)
        return _explain_linear(inner=core.inner, df=df_u)

    return _explain_linear(inner=core, df=df)


def _explain_linear(inner, df: pd.DataFrame) -> dict:
    out: dict = {"top_features": [], "top_metadata": []}
    if not hasattr(inner, "named_steps"):
        return out
    if "preprocess" in inner.named_steps and "clf" in inner.named_steps:
        prep = inner.named_steps["preprocess"]
        clf = inner.named_steps["clf"]
        try:
            Xmat = prep.transform(df)
            coef = clf.coef_.ravel()
            if hasattr(Xmat, "getrow"):
                row = Xmat.getrow(0)
                contrib = row.multiply(coef).toarray().ravel()
            else:
                contrib = np.asarray(Xmat[0] * coef).ravel()
            names = prep.get_feature_names_out()
            pairs = sorted(zip(names, contrib), key=lambda x: abs(x[1]), reverse=True)
            for n, v in pairs:
                if n.startswith("txt__"):
                    term = n.split("__", 1)[-1]
                    out["top_features"].append({"term": term, "weight": float(v)})
                elif n.startswith("bin__") or n.startswith("cat__"):
                    out["top_metadata"].append({"feature": n, "weight": float(v)})
            out["top_features"] = out["top_features"][:15]
            out["top_metadata"] = out["top_metadata"][:12]
            return out
        except (ValueError, AttributeError, TypeError):
            pass
    return _explain_text_only_legacy(inner, df)


def _explain_text_only_legacy(pipe, df: pd.DataFrame) -> dict:
    text = (
        df["combined_text"].iloc[0]
        if "combined_text" in df.columns
        else str(df.iloc[0, 0])
    )
    if "clean" not in pipe.named_steps:
        return {"top_features": [], "top_metadata": []}
    clean = pipe.named_steps["clean"]
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    cleaned = clean.transform([text])
    X = tfidf.transform(cleaned)
    coef = clf.coef_.ravel()
    contrib = X.multiply(coef).tocsr()
    row = contrib.getrow(0)
    nz = row.nonzero()[1]
    if len(nz) == 0:
        return {"top_features": [], "top_metadata": []}
    feats = tfidf.get_feature_names_out()
    pairs = [(feats[i], float(row[0, i])) for i in nz]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return {"top_features": [{"term": t, "weight": w} for t, w in pairs[:15]], "top_metadata": []}


def read_metrics() -> dict | None:
    if not METRICS_PATH.is_file():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
