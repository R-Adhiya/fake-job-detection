# Bugfix Requirements Document

## Introduction

The Fraud Job Detection System's training pipeline (`train.py`) violates two hard constraints required for the system to run correctly on moderate hardware:

1. `max_features` in TF-IDF is set to 5000 (and the hyperparameter search explores 5000–9000), exceeding the required maximum of 3000.
2. `n_iter` in `RandomizedSearchCV` is set to 22, exceeding the required maximum of 5.

These violations cause excessive memory usage and prolonged training times, preventing the model from running reliably on moderate hardware. The fix must bring both values within bounds without altering any other behavior.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `train.py` is executed THEN the system uses `max_features=5000` in `TFIDF_BASE`, exceeding the allowed maximum of 3000.

1.2 WHEN `RandomizedSearchCV` runs THEN the system uses `n_iter=22`, exceeding the allowed maximum of 5.

1.3 WHEN the hyperparameter search explores `preprocess__txt__max_features` THEN the system samples from `[5000, 7000, 9000]`, all of which exceed the allowed maximum of 3000.

### Expected Behavior (Correct)

2.1 WHEN `train.py` is executed THEN the system SHALL use `max_features` of 3000 or fewer in `TFIDF_BASE`.

2.2 WHEN `RandomizedSearchCV` runs THEN the system SHALL use `n_iter` of 5 or fewer.

2.3 WHEN the hyperparameter search explores `preprocess__txt__max_features` THEN the system SHALL only sample from values that are 3000 or fewer (e.g., `[1000, 2000, 3000]`).

### Unchanged Behavior (Regression Prevention)

3.1 WHEN training completes successfully THEN the system SHALL CONTINUE TO save the trained model to `models/fraud_pipeline.joblib`.

3.2 WHEN training completes successfully THEN the system SHALL CONTINUE TO save metrics to `models/metrics.json`.

3.3 WHEN a valid job posting is submitted to the Flask `/predict` endpoint THEN the system SHALL CONTINUE TO return a prediction label and fraud probability.

3.4 WHEN a valid job posting is analyzed in the Streamlit UI THEN the system SHALL CONTINUE TO display the prediction, probability, and risk level.

3.5 WHEN the hybrid pipeline is built THEN the system SHALL CONTINUE TO use TF-IDF (ngram 1,2), binary passthrough, OneHotEncoder for categoricals, and LogisticRegression with SAGA solver.

3.6 WHEN baseline models are trained THEN the system SHALL CONTINUE TO evaluate Naive Bayes, Random Forest (with SVD), and text-only Logistic Regression.
