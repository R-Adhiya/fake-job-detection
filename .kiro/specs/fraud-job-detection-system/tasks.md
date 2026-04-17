# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Fault Condition** - Constraint Violations in train.py
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bugs exist
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the constraint violations exist
  - **Scoped PBT Approach**: Scope the property to the three concrete failing cases (deterministic bugs)
  - Test that `TFIDF_BASE["max_features"]` is <= 3000 (currently 5000 — FAILS)
  - Test that `RandomizedSearchCV` `n_iter` is <= 5 (currently 22 — FAILS)
  - Test that all values in `param_dist["preprocess__txt__max_features"]` are <= 3000 (currently [5000, 7000, 9000] — FAILS)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bugs exist)
  - Document counterexamples: `max_features=5000 > 3000`, `n_iter=22 > 5`, `param_dist values [5000,7000,9000] all > 3000`
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Pipeline Structure and Output Artifacts
  - **IMPORTANT**: Follow observation-first methodology
  - Observe: pipeline uses TF-IDF (ngram 1,2), binary passthrough, OHE categoricals, LogisticRegression with SAGA solver
  - Observe: training saves model to `models/fraud_pipeline.joblib` and metrics to `models/metrics.json`
  - Observe: baseline models (Naive Bayes, Random Forest with SVD, text-only LR) are all built and evaluated
  - Write property-based tests: for all valid job postings, the pipeline structure keys remain unchanged
  - Write property-based tests: output artifact paths remain `MODEL_PATH` and `METRICS_PATH`
  - Verify tests pass on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3. Fix constraint violations in train.py

  - [x] 3.1 Implement the fix
    - Change `TFIDF_BASE` `max_features` from `5000` to `3000`
    - Change `RandomizedSearchCV` `n_iter` from `22` to `5`
    - Change `param_dist["preprocess__txt__max_features"]` from `[5000, 7000, 9000]` to `[1000, 2000, 3000]`
    - _Bug_Condition: isBugCondition(train.py) where TFIDF_BASE.max_features > 3000 OR n_iter > 5 OR any(v > 3000 for v in param_dist["preprocess__txt__max_features"])_
    - _Expected_Behavior: TFIDF_BASE.max_features <= 3000 AND n_iter <= 5 AND all(v <= 3000 for v in param_dist["preprocess__txt__max_features"])_
    - _Preservation: Pipeline structure, artifact paths, baseline models, and all other logic remain unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Constraint Violations Resolved
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms all three constraint values are within bounds
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bugs are fixed)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Pipeline Structure and Output Artifacts
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm pipeline structure, artifact paths, and baseline model logic are unchanged

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
