---
title: Fraud Job Detection API
emoji: ⚡
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: gradio_app.py
pinned: false
license: mit
---

# ⚡ Fraud Job Detection — Flask API via Gradio

REST-style fraud detection API wrapped in Gradio for Hugging Face Spaces.

## How to Deploy Flask API to Hugging Face Spaces

HF Spaces does NOT support raw Flask. The solution is to wrap Flask with Gradio,
which HF supports natively. The `gradio_app.py` file exposes the same `/predict`
logic through a Gradio interface.

### Step 1 — Create a new Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - Space name: `fraud-job-detection-api`
   - SDK: **Gradio**
   - Hardware: CPU basic (free)
3. Click **Create Space**

### Step 2 — Clone and copy files
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/fraud-job-detection-api
cd fraud-job-detection-api

# Copy these files:
# gradio_app.py
# config.py
# preprocess.py
# features.py
# model_wrapper.py
# predict_utils.py
# requirements_gradio.txt   ← rename to requirements.txt
# models/fraud_pipeline.joblib
# models/metrics.json
```

### Step 3 — Push
```bash
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes
git add .
git commit -m "Deploy fraud detection API"
git push
```
