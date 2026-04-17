---
title: Fraud Job Detection
emoji: 🔍
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
---

# 🔍 Fraud Job Detection — Streamlit App

Detects fraudulent job postings using a hybrid ML model (TF-IDF + metadata + Logistic Regression).

## Features
- Fraud probability gauge chart
- NLP feature importance charts
- Confusion matrix, ROC curve, model comparison
- EDA tab with class distribution, text length, top words

## How to Deploy to Hugging Face Spaces

### Step 1 — Create a new Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - Space name: `fraud-job-detection`
   - License: MIT
   - SDK: **Streamlit**
   - Hardware: CPU basic (free)
3. Click **Create Space**

### Step 2 — Install Git LFS (for the model file)
```bash
git lfs install
```

### Step 3 — Clone your Space and copy files
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/fraud-job-detection
cd fraud-job-detection

# Copy these files from your local project:
# app.py
# streamlit_app.py
# config.py
# preprocess.py
# features.py
# model_wrapper.py
# predict_utils.py
# requirements.txt
# models/fraud_pipeline.joblib   ← large file, needs LFS
# models/metrics.json
# Dataset.csv                    ← optional, for EDA tab
# .streamlit/config.toml
```

### Step 4 — Track model with Git LFS
```bash
git lfs track "models/*.joblib"
git lfs track "*.csv"
git add .gitattributes
```

### Step 5 — Commit and push
```bash
git add .
git commit -m "Deploy fraud job detection app"
git push
```

Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/fraud-job-detection`
