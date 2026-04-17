"""
Gradio wrapper for Fraud Job Detection API.
Deployed on Hugging Face Spaces (SDK: gradio).
Replicates the Flask /predict endpoint logic.
"""
from __future__ import annotations

import gradio as gr

from features import frame_from_user_fields
from predict_utils import load_pipeline, predict_label, predict_proba_fraud, risk_level

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_pipeline()
    return _model


def predict(
    title: str,
    description: str,
    requirements: str,
    location: str,
    company_profile: str,
    benefits: str,
    employment_type: str,
    required_experience: str,
    required_education: str,
    industry: str,
    job_function: str,
    telecommuting: bool,
    has_company_logo: bool,
    has_questions: bool,
) -> tuple[str, str, str, str]:
    """Run fraud detection and return prediction results."""
    text_check = " ".join(filter(None, [title, description, requirements, location]))
    if len(text_check.strip()) < 10:
        return "❌ Error", "Please provide more text (at least title + description).", "—", "—"

    try:
        model = get_model()
    except FileNotFoundError as e:
        return "❌ Error", str(e), "—", "—"

    df_row = frame_from_user_fields(
        title=title or "",
        location=location or "",
        department="",
        company_profile=company_profile or "",
        description=description or "",
        requirements=requirements or "",
        benefits=benefits or "",
        telecommuting=1.0 if telecommuting else 0.0,
        has_company_logo=1.0 if has_company_logo else 0.0,
        has_questions=1.0 if has_questions else 0.0,
        employment_type=employment_type or None,
        required_experience=required_experience or None,
        required_education=required_education or None,
        industry=industry or None,
        job_function=job_function or None,
    )

    p_fraud = predict_proba_fraud(model, df_row)
    label   = predict_label(model, df_row)
    risk    = risk_level(p_fraud)

    verdict = "🚨 FRAUDULENT" if label == 1 else "✅ LEGITIMATE"
    prob    = f"{p_fraud:.1%}"
    risk_str = risk.upper()
    detail  = (
        f"Fraud probability: {p_fraud:.4f}\n"
        f"Risk level: {risk_str}\n"
        f"Decision threshold: 0.38 (F1-optimised)"
    )
    return verdict, prob, risk_str, detail


# ── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="Fraud Job Detection API", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Fraud Job Detection API
        **Hybrid ML model** · TF-IDF text + metadata · Logistic Regression · threshold-optimised for fraud F1
        
        Fill in the job posting details and click **Analyze** to detect fraud.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            inp_title       = gr.Textbox(label="Job Title *", placeholder="e.g. Marketing Intern")
            inp_description = gr.Textbox(label="Job Description *", lines=6,
                                         placeholder="Paste the full job description here…")
            inp_requirements = gr.Textbox(label="Requirements", lines=3)
            inp_location    = gr.Textbox(label="Location", placeholder="e.g. New York, US")
            inp_company     = gr.Textbox(label="Company Profile", lines=2)
            inp_benefits    = gr.Textbox(label="Benefits", lines=2)

        with gr.Column(scale=1):
            inp_emp     = gr.Textbox(label="Employment Type", placeholder="e.g. Full-time")
            inp_exp     = gr.Textbox(label="Required Experience", placeholder="e.g. Entry level")
            inp_edu     = gr.Textbox(label="Required Education", placeholder="e.g. Bachelor's")
            inp_ind     = gr.Textbox(label="Industry", placeholder="e.g. Computer Software")
            inp_func    = gr.Textbox(label="Job Function", placeholder="e.g. Sales")
            inp_tc      = gr.Checkbox(label="Telecommuting offered")
            inp_logo    = gr.Checkbox(label="Has company logo")
            inp_quest   = gr.Checkbox(label="Has screening questions")

    analyze_btn = gr.Button("🔍 Analyze Job Posting", variant="primary")

    with gr.Row():
        out_verdict = gr.Textbox(label="Verdict", interactive=False)
        out_prob    = gr.Textbox(label="Fraud Probability", interactive=False)
        out_risk    = gr.Textbox(label="Risk Level", interactive=False)

    out_detail = gr.Textbox(label="Details", interactive=False, lines=4)

    analyze_btn.click(
        fn=predict,
        inputs=[
            inp_title, inp_description, inp_requirements, inp_location,
            inp_company, inp_benefits, inp_emp, inp_exp, inp_edu, inp_ind, inp_func,
            inp_tc, inp_logo, inp_quest,
        ],
        outputs=[out_verdict, out_prob, out_risk, out_detail],
    )

    gr.Markdown(
        """
        ---
        ### API Usage (via HF Inference API)
        ```python
        import requests
        response = requests.post(
            "https://YOUR_USERNAME-fraud-job-detection-api.hf.space/run/predict",
            json={"data": ["Job Title", "Job Description", "", "", "", "", "", "", "", "", "", False, False, False]}
        )
        print(response.json())
        ```
        """
    )

if __name__ == "__main__":
    demo.launch()
