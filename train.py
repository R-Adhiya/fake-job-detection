"""
Train hybrid fraud detector: TextCleaner fit once on training text, then tune
TF-IDF + metadata + LogisticRegression (fast). Saves FullHybridModel + threshold.
Includes text-only baselines (NB, RF, LR).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config import METRICS_PATH, MODEL_PATH, MODELS_DIR, RANDOM_STATE, resolve_dataset_path
from features import META_BINARY, META_CATEGORICAL, enrich_dataframe, training_columns
from model_wrapper import FraudDetectionModel, FullHybridModel
from preprocess import TextCleaner

TFIDF_BASE = dict(
    ngram_range=(1, 2),
    max_features=3000,
    min_df=1,
    max_df=0.95,
    sublinear_tf=True,
)


def load_frame() -> tuple[pd.DataFrame, pd.Series]:
    path = resolve_dataset_path()
    df = pd.read_csv(path)
    if "fraudulent" not in df.columns:
        raise ValueError("Column 'fraudulent' not found.")
    df = enrich_dataframe(df)
    y = df["fraudulent"].astype(int)
    X = df[training_columns()]
    return X, y


def build_inner_pipeline() -> Pipeline:
    """TF-IDF on already-cleaned combined_text + binary + OHE categoricals."""
    tfidf = TfidfVectorizer(**TFIDF_BASE)
    cat = OneHotEncoder(
        handle_unknown="ignore",
        max_categories=18,
        sparse_output=True,
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("txt", tfidf, "combined_text"),
            ("bin", "passthrough", META_BINARY),
            ("cat", cat, META_CATEGORICAL),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    clf = LogisticRegression(
        class_weight="balanced",
        penalty="l2",
        max_iter=20000,
        random_state=RANDOM_STATE,
        solver="saga",
        tol=1e-3,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def apply_cleaner(cleaner: TextCleaner, X: pd.DataFrame) -> pd.DataFrame:
    o = X.copy()
    o["combined_text"] = cleaner.transform(o["combined_text"].values)
    return o


def optimal_fraud_threshold(y_true: np.ndarray, proba_fraud: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba_fraud >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def build_text_only_baselines() -> dict[str, Pipeline]:
    clean = ("clean", TextCleaner())
    tfidf = ("tfidf", TfidfVectorizer(**TFIDF_BASE))
    lr = Pipeline(
        [
            clean,
            tfidf,
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=8000,
                    random_state=RANDOM_STATE,
                    solver="saga",
                    tol=1e-3,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    nb = Pipeline(
        [
            clean,
            ("tfidf_nb", TfidfVectorizer(**TFIDF_BASE)),
            ("nb", MultinomialNB()),
        ]
    )
    rf = Pipeline(
        [
            clean,
            ("tfidf_rf", TfidfVectorizer(**TFIDF_BASE)),
            ("svd", TruncatedSVD(n_components=300, random_state=RANDOM_STATE)),
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    n_estimators=100,
                    max_depth=32,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {"naive_bayes": nb, "random_forest": rf, "logistic_regression_text_only": lr}


def evaluate(name: str, y_true, y_pred) -> dict:
    out = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fraud": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fraud": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fraud": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_pred))
    except Exception:
        out["roc_auc"] = 0.0
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix [ [TN FP], [FN TP] ]:", out["confusion_matrix"])
    return out


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    cleaner = TextCleaner()
    cleaner.fit(X_train["combined_text"])
    X_tr_c = apply_cleaner(cleaner, X_tr)
    X_val_c = apply_cleaner(cleaner, X_val)
    X_train_c = apply_cleaner(cleaner, X_train)

    inner = build_inner_pipeline()
    param_dist = {
        "preprocess__txt__max_features": [2000, 3000],
        "preprocess__txt__min_df": [1, 2],
        "preprocess__txt__max_df": [0.90, 0.95],
        "preprocess__txt__ngram_range": [(1, 1), (1, 2)],
        "clf__C": loguniform(0.1, 10.0),
        "clf__class_weight": ["balanced", {0: 1, 1: 3}, {0: 1, 1: 5}],
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        inner,
        param_distributions=param_dist,
        n_iter=5,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    print("Tuning inner pipeline (TF-IDF + metadata + LogisticRegression)…")
    search.fit(X_tr_c, y_tr)
    print("Best params:", search.best_params_)
    best_inner = search.best_estimator_
    p_val = best_inner.predict_proba(X_val_c)[:, 1]
    thresh = optimal_fraud_threshold(y_val.values, p_val)
    print(f"Validation-optimal fraud threshold: {thresh:.3f}")

    best_inner.fit(X_train_c, y_train)
    full = FullHybridModel(cleaner, best_inner)
    hybrid_wrapped = FraudDetectionModel(full, threshold=thresh)

    metrics_list: list[dict] = []
    y_pred_h = hybrid_wrapped.predict(X_test)
    metrics_list.append(evaluate("hybrid_lr_tuned_threshold", y_test, y_pred_h))
    metrics_list[-1]["threshold"] = thresh
    metrics_list[-1]["best_cv_f1"] = float(search.best_score_)

    Xt_train = X_train["combined_text"]
    Xt_test = X_test["combined_text"]
    for name, pipe in build_text_only_baselines().items():
        if name == "naive_bayes":
            sw = compute_sample_weight("balanced", y_train)
            pipe.fit(Xt_train, y_train, nb__sample_weight=sw)
        else:
            pipe.fit(Xt_train, y_train)
        metrics_list.append(evaluate(name, y_test, pipe.predict(Xt_test)))

    best = max(metrics_list, key=lambda m: m["f1_fraud"])
    print(f"\nBest model by test F1 (fraud): {best['model']} F1={best['f1_fraud']:.4f}")

    joblib.dump(hybrid_wrapped, MODEL_PATH)

    payload = {
        "best_model": best["model"],
        "saved_artifact": "FraudDetectionModel(FullHybridModel + threshold)",
        "model_path": str(MODEL_PATH),
        "metrics": metrics_list,
        "hybrid_threshold": thresh,
        "randomized_search_best_params": search.best_params_,
        "cv_results_mean_test_score": search.cv_results_["mean_test_score"].tolist(),
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
