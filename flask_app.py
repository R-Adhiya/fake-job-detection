"""Flask REST API for fake job detection."""
from __future__ import annotations

from flask import Flask, jsonify, request

from features import frame_from_user_fields
from predict_utils import explain, load_pipeline, predict_label, predict_proba_fraud, risk_level

app = Flask(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_pipeline()
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    def fnum(key: str, default: float | None = None) -> float | None:
        v = data.get(key)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    text_probe = " ".join(
        str(data.get(k, "") or "")
        for k in (
            "title",
            "location",
            "department",
            "company_profile",
            "description",
            "requirements",
            "benefits",
        )
    )
    if len(text_probe.strip()) < 10:
        return jsonify({"error": "Provide more text in description or other fields."}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    df_row = frame_from_user_fields(
        title=data.get("title", "") or "",
        location=data.get("location", "") or "",
        department=data.get("department", "") or "",
        company_profile=data.get("company_profile", "") or "",
        description=data.get("description", "") or "",
        requirements=data.get("requirements", "") or "",
        benefits=data.get("benefits", "") or "",
        telecommuting=fnum("telecommuting", None),
        has_company_logo=fnum("has_company_logo", None),
        has_questions=fnum("has_questions", None),
        employment_type=data.get("employment_type") or None,
        required_experience=data.get("required_experience") or None,
        required_education=data.get("required_education") or None,
        industry=data.get("industry") or None,
        job_function=data.get("function") or None,
    )
    p_fraud = predict_proba_fraud(model, df_row)
    label = predict_label(model, df_row)
    body = {
        "label": int(label),
        "probability_fraud": p_fraud,
        "risk_level": risk_level(p_fraud),
        "explanation": explain(model, df_row),
    }
    return jsonify(body)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
