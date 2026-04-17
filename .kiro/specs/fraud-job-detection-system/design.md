# Fraud Job Detection System Bugfix Design

## Overview

The training pipeline (`train.py`) violates two hard constraints required for the system to run on moderate hardware:

1. `TFIDF_BASE.max_features` is set to `5000`, exceeding the allowed maximum of `3000`.
2. `RandomizedSearchCV.n_iter` is set to `22`, exceeding the allowed maximum of `5`.
3. The hyperparameter search grid `param_dist["preprocess__txt__max_features"]` contains `[5000, 7000, 9000]`, all exceeding `3000`.

The fix is surgical: update these three literal values in `train.py`. No other logic, pipeline structure, or behavior changes.

## Glossary

- **Bug_Condition (C)*