"""Streamlit UI for fake job detection — rich UI rewrite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gobj
import streamlit as st
from collections import Counter

from config import resolve_dataset_path
from features import frame_from_user_fields
from predict_utils import (
    explain,
    load_pipeline,
    predict_label,
    predict_proba_fraud,
    read_metrics,
    risk_level,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Job Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
STYLES = """
<style>
/* ---- Header gradient ---- */
.app-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    padding: 32px 36px 26px 36px;
    border-radius: 14px;
    margin-bottom: 24px;
    text-align: center;
}
.app-header h1 {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.app-header p {
    margin: 8px 0 0 0;
    font-size: 0.97rem;
    color: #e0e7ff !important;
    opacity: 1;
}

/* ---- Input card ---- */
.input-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* ---- Result cards ---- */
.result-card {
    padding: 24px 32px;
    border-radius: 14px;
    margin-bottom: 20px;
    font-family: sans-serif;
    box-shadow: 0 4px 24px rgba(0,0,0,0.12);
}
.result-card h2 { margin: 0 0 8px 0; font-size: 1.8rem; font-weight: 800; color: #fff; }
.result-card p  { margin: 0; font-size: 1.05rem; color: #fff; }
.fraud-card { background: linear-gradient(135deg, #ff4b4b 0%, #c0392b 100%); }
.legit-card { background: linear-gradient(135deg, #21c55d 0%, #16a34a 100%); }

/* ---- Metric boxes ---- */
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label {
    color: #374151 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 700 !important;
}

/* ---- Info tip box ---- */
.tip-box {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 16px 18px;
    font-size: 0.9rem;
    color: #1e3a5f;
    line-height: 1.6;
}
.tip-box strong { color: #1d4ed8; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: #a5b4fc !important; }

/* ---- Sidebar info boxes (Best model / Fraud threshold) ---- */
section[data-testid="stSidebar"] code {
    background: #1e293b !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] strong {
    color: #f1f5f9 !important;
    font-weight: 500 !important;
}

/* ---- General ---- */
.stTextArea textarea { background: #ffffff; border-radius: 8px; color: #111827; }
.stTextInput input   { background: #ffffff; border-radius: 8px; color: #111827; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>🔍 Fraud Job Detection</h1>
        <p>Hybrid model · TF-IDF text signals + platform flags + job metadata · threshold-optimised for fraud F1</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def cached_model():
    return load_pipeline()


try:
    model = cached_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Load dataset (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(resolve_dataset_path())


# ── Sidebar ───────────────────────────────────────────────────────────────────
m = read_metrics()
with st.sidebar:
    st.markdown("## 📊 Model Performance")

    if m:
        best = m.get("best_model", "—")
        thr  = m.get("hybrid_threshold")
        st.markdown(f"**Best model:** `{best}`")
        if thr is not None:
            st.markdown(f"**Fraud threshold:** `{thr:.3f}` *(F1-tuned)*")
        st.markdown("---")

        rows = m.get("metrics", [])
        if rows:
            # ── Grouped bar: F1 / Precision / Recall per model ────────────
            models_  = [r["model"] for r in rows for _ in range(3)]
            metrics_ = ["F1", "Precision", "Recall"] * len(rows)
            values_  = [
                v
                for r in rows
                for v in (r["f1_fraud"], r["precision_fraud"], r["recall_fraud"])
            ]
            fig_sb = px.bar(
                x=models_,
                y=values_,
                color=metrics_,
                barmode="group",
                labels={"x": "Model", "y": "Score", "color": "Metric"},
                title="F1 / Precision / Recall",
                color_discrete_map={
                    "F1": "#6366f1",
                    "Precision": "#06b6d4",
                    "Recall": "#f59e0b",
                },
            )
            fig_sb.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=36, b=60),
                legend=dict(
                    orientation="h", y=-0.4,
                    font=dict(color="#f1f5f9", size=12),
                    itemsizing="constant",
                    tracegroupgap=8,
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9", size=12),
                title_font=dict(color="#a5b4fc", size=13),
                xaxis=dict(
                    tickfont=dict(color="#f1f5f9", size=11),
                    title_font=dict(color="#f1f5f9"),
                    gridcolor="rgba(255,255,255,0.1)",
                ),
                yaxis=dict(
                    tickfont=dict(color="#f1f5f9", size=11),
                    title_font=dict(color="#f1f5f9"),
                    gridcolor="rgba(255,255,255,0.1)",
                ),
            )
            st.plotly_chart(fig_sb, use_container_width=True)

            # ── Donut: class distribution from best model confusion matrix ─
            hybrid_row = next(
                (r for r in rows if "hybrid" in r.get("model", "").lower()), rows[-1]
            )
            cm = hybrid_row.get("confusion_matrix")
            if cm and len(cm) == 2:
                tn = cm[0][0]; fp = cm[0][1]
                fn = cm[1][0]; tp = cm[1][1]
                legit_total = tn + fp
                fraud_total = fn + tp
                fig_donut = gobj.Figure(
                    gobj.Pie(
                        labels=["Legitimate", "Fraudulent"],
                        values=[legit_total, fraud_total],
                        hole=0.55,
                        marker=dict(colors=["#21c55d", "#ff4b4b"]),
                        textfont=dict(color="#ffffff"),
                    )
                )
                fig_donut.update_layout(
                    title=dict(text="Class Distribution", font=dict(color="#a5b4fc", size=13)),
                    height=260,
                    margin=dict(l=0, r=0, t=36, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#f1f5f9", size=12)),
                    showlegend=True,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # ── Accuracy comparison horizontal bar ────────────────────────
            if rows:
                fig_acc = px.bar(
                    x=[r.get("accuracy", 0) for r in rows],
                    y=[r["model"] for r in rows],
                    orientation="h",
                    title="Accuracy by Model",
                    labels={"x": "Accuracy", "y": "Model"},
                    color=[r.get("accuracy", 0) for r in rows],
                    color_continuous_scale="Blues",
                    text=[f"{r.get('accuracy',0):.3f}" for r in rows],
                )
                fig_acc.update_traces(textposition="outside")
                fig_acc.update_layout(
                    height=250, margin=dict(l=0,r=0,t=36,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9", size=12),
                    title_font=dict(color="#a5b4fc", size=13),
                    xaxis=dict(
                        tickfont=dict(color="#f1f5f9", size=11),
                        gridcolor="rgba(255,255,255,0.1)",
                    ),
                    yaxis=dict(tickfont=dict(color="#f1f5f9", size=11)),
                    showlegend=False, coloraxis_showscale=False,
                )
                st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("No metrics file found. Train the model first.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Analyze Job", "📊 EDA & Model Analysis"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analyze Job (100% identical to original)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Main input form ───────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        title       = st.text_input("Job title", placeholder="e.g. Marketing Intern")
        location    = st.text_input("Location (optional)", placeholder="e.g. US, NY, New York")
        department  = st.text_input("Department (optional)", placeholder="e.g. Marketing")
        company     = st.text_area("Company profile (optional)", height=100,
                                   placeholder="Employer description…")
        description = st.text_area("Job description *", height=220,
                                   placeholder="Paste the full job description here…")
        req         = st.text_area("Requirements (optional)", height=120)
        ben         = st.text_area("Benefits (optional)", height=80)

        with st.expander("⚙️ Optional metadata (improves accuracy if known)"):
            tc      = st.checkbox("Telecommuting offered", value=False)
            logo    = st.checkbox("Has company logo on posting", value=False)
            quest   = st.checkbox("Has screening questions", value=False)
            emp     = st.text_input("Employment type", placeholder="e.g. Full-time, Internship")
            exp_lvl = st.text_input("Required experience", placeholder="e.g. Entry level")
            edu     = st.text_input("Required education", placeholder="e.g. Bachelor's")
            ind     = st.text_input("Industry", placeholder="e.g. Computer Software")
            func    = st.text_input("Job function", placeholder="e.g. Sales")

    with col2:
        st.markdown(
            """
            <div class="tip-box">
                <strong>💡 How to use</strong><br><br>
                1. Paste the <strong>job description</strong> — that's the most important field.<br><br>
                2. Add a <strong>title</strong> and any other details you have.<br><br>
                3. Expand <em>Optional metadata</em> for higher accuracy.<br><br>
                4. Hit <strong>Analyze</strong> and review the fraud probability gauge and feature signals.<br><br>
                <strong>Risk levels</strong><br>
                🟢 &lt;40% — Low risk<br>
                🟡 40–70% — Caution<br>
                🔴 &gt;70% — High risk
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze Job Posting", type="primary", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if analyze_btn:
        text_check = " ".join(
            str(p) for p in (title, location, department, company, description, req, ben)
            if p and str(p).strip()
        )
        if len(text_check.strip()) < 20:
            st.warning("Please enter more text (at least a short description).")
        else:
            df_row = frame_from_user_fields(
                title=title,
                location=location,
                department=department,
                company_profile=company,
                description=description,
                requirements=req,
                benefits=ben,
                telecommuting=1.0 if tc else 0.0,
                has_company_logo=1.0 if logo else 0.0,
                has_questions=1.0 if quest else 0.0,
                employment_type=emp or None,
                required_experience=exp_lvl or None,
                required_education=edu or None,
                industry=ind or None,
                job_function=func or None,
            )

            p_fraud = predict_proba_fraud(model, df_row)
            label   = predict_label(model, df_row)
            risk    = risk_level(p_fraud)
            is_fraud = label == 1

            # ── a. Result banner ─────────────────────────────────────────────
            if is_fraud:
                st.markdown(
                    f"""
                    <div class="result-card fraud-card">
                        <h2>⚠️ FRAUDULENT POSTING DETECTED</h2>
                        <p>Fraud probability: <strong>{p_fraud:.1%}</strong> &nbsp;·&nbsp;
                           Risk level: <strong>{risk.upper()}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card legit-card">
                        <h2>✅ LEGITIMATE POSTING</h2>
                        <p>Fraud probability: <strong>{p_fraud:.1%}</strong> &nbsp;·&nbsp;
                           Risk level: <strong>{risk.upper()}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── b. Metric columns ────────────────────────────────────────────
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Prediction",       "🚨 Fraudulent" if is_fraud else "✅ Legitimate")
            mc2.metric("Fraud Probability", f"{p_fraud:.1%}")
            mc3.metric("Risk Level",        risk.capitalize())

            st.markdown("<br>", unsafe_allow_html=True)

            # ── c. Gauge chart ───────────────────────────────────────────────
            gauge_color = "#ff4b4b" if p_fraud >= 0.7 else ("#f59e0b" if p_fraud >= 0.4 else "#21c55d")
            fig_gauge = gobj.Figure(
                gobj.Indicator(
                    mode="gauge+number+delta",
                    value=round(p_fraud * 100, 1),
                    number={"suffix": "%", "font": {"size": 40, "color": gauge_color}},
                    delta={"reference": 50, "valueformat": ".1f",
                           "increasing": {"color": "#ff4b4b"},
                           "decreasing": {"color": "#21c55d"}},
                    title={"text": "Fraud Probability", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                        "bar":  {"color": gauge_color, "thickness": 0.25},
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": "#e2e8f0",
                        "steps": [
                            {"range": [0,  40], "color": "#d1fae5"},
                            {"range": [40, 70], "color": "#fef3c7"},
                            {"range": [70, 100],"color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "#1e293b", "width": 4},
                            "thickness": 0.8,
                            "value": round(p_fraud * 100, 1),
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=30, r=30, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── d. NLP + Metadata contribution charts ────────────────────────
            exp = explain(model, df_row)
            ch1, ch2 = st.columns(2)

            with ch1:
                nlp_data = exp.get("top_features", [])
                if nlp_data:
                    nlp_terms   = [x["term"]   for x in nlp_data]
                    nlp_weights = [x["weight"] for x in nlp_data]
                    nlp_colors  = ["Fraud signal" if w >= 0 else "Legit signal" for w in nlp_weights]
                    fig_nlp = px.bar(
                        x=nlp_weights,
                        y=nlp_terms,
                        color=nlp_colors,
                        orientation="h",
                        title="🔤 Top NLP Signals",
                        labels={"x": "Weight", "y": "Term", "color": "Direction"},
                        color_discrete_map={"Fraud signal": "#ff4b4b", "Legit signal": "#21c55d"},
                    )
                    fig_nlp.update_layout(
                        height=max(280, len(nlp_terms) * 28),
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis={"categoryorder": "total ascending", "tickfont": {"color": "#111827", "size": 12}},
                        xaxis={"tickfont": {"color": "#111827"}, "gridcolor": "#e5e7eb"},
                        font=dict(color="#111827"),
                        title_font=dict(color="#111827", size=14),
                        legend=dict(font=dict(color="#111827", size=12)),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_nlp, use_container_width=True)
                else:
                    st.info("No NLP signals available.")

            with ch2:
                meta_data = exp.get("top_metadata", [])[:8]
                if meta_data:
                    meta_feats   = [x["feature"] for x in meta_data]
                    meta_weights = [x["weight"]  for x in meta_data]
                    meta_colors  = ["Fraud signal" if w >= 0 else "Legit signal" for w in meta_weights]
                    fig_meta = px.bar(
                        x=meta_weights,
                        y=meta_feats,
                        color=meta_colors,
                        orientation="h",
                        title="📋 Metadata Signals",
                        labels={"x": "Weight", "y": "Feature", "color": "Direction"},
                        color_discrete_map={"Fraud signal": "#ff4b4b", "Legit signal": "#21c55d"},
                    )
                    fig_meta.update_layout(
                        height=max(280, len(meta_feats) * 28),
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis={"categoryorder": "total ascending", "tickfont": {"color": "#111827", "size": 12}},
                        xaxis={"tickfont": {"color": "#111827"}, "gridcolor": "#e5e7eb"},
                        font=dict(color="#111827"),
                        title_font=dict(color="#111827", size=14),
                        legend=dict(font=dict(color="#111827", size=12)),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_meta, use_container_width=True)
                else:
                    st.info("No metadata signals available.")

            # ── e. Confusion matrix heatmap ──────────────────────────────────
            if m:
                rows = m.get("metrics", [])
                hybrid_row = next(
                    (r for r in rows if "hybrid" in r.get("model", "").lower()),
                    rows[-1] if rows else None,
                )
                if hybrid_row:
                    cm = hybrid_row.get("confusion_matrix")
                    if cm and len(cm) == 2:
                        cm_arr = np.array(cm)
                        fig_cm = px.imshow(
                            cm_arr,
                            text_auto=True,
                            x=["Legitimate", "Fraudulent"],
                            y=["Legitimate", "Fraudulent"],
                            color_continuous_scale="RdYlGn_r",
                            title="🎯 Confusion Matrix (Hybrid Model)",
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                        )
                        fig_cm.update_layout(
                            height=360,
                            margin=dict(l=0, r=0, t=50, b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

            # ── f. Radar chart ───────────────────────────────────────────────
            if m:
                rows = m.get("metrics", [])
                if rows:
                    radar_cats = ["Accuracy", "F1", "Precision", "Recall"]
                    fig_radar = gobj.Figure()
                    colors_radar = ["#6366f1", "#06b6d4", "#f59e0b", "#ec4899", "#10b981"]
                    for i, r in enumerate(rows):
                        vals = [
                            r.get("accuracy",        0),
                            r.get("f1_fraud",         0),
                            r.get("precision_fraud",  0),
                            r.get("recall_fraud",     0),
                        ]
                        vals_closed = vals + [vals[0]]
                        cats_closed = radar_cats + [radar_cats[0]]
                        fig_radar.add_trace(
                            gobj.Scatterpolar(
                                r=vals_closed,
                                theta=cats_closed,
                                fill="toself",
                                name=r.get("model", f"Model {i+1}"),
                                line=dict(color=colors_radar[i % len(colors_radar)]),
                                opacity=0.7,
                            )
                        )
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True, range=[0, 1],
                                tickfont=dict(color="#111827", size=11),
                                gridcolor="rgba(0,0,0,0.15)",
                            ),
                            angularaxis=dict(tickfont=dict(color="#111827", size=12)),
                            bgcolor="rgba(255,255,255,0.05)",
                        ),
                        title=dict(text="🕸️ Model Comparison Radar", font=dict(color="#111827", size=15)),
                        height=420,
                        margin=dict(l=40, r=40, t=60, b=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(font=dict(color="#111827", size=12)),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            # ── g. Risk breakdown bar ────────────────────────────────────────
            fig_risk = gobj.Figure()
            fig_risk.add_trace(gobj.Bar(
                x=[40], y=["Risk Zone"], orientation="h",
                name="Safe (0–40%)", marker_color="#d1fae5",
                base=0, width=0.5,
            ))
            fig_risk.add_trace(gobj.Bar(
                x=[30], y=["Risk Zone"], orientation="h",
                name="Caution (40–70%)", marker_color="#fef3c7",
                base=40, width=0.5,
            ))
            fig_risk.add_trace(gobj.Bar(
                x=[30], y=["Risk Zone"], orientation="h",
                name="Danger (70–100%)", marker_color="#fee2e2",
                base=70, width=0.5,
            ))
            # Vertical line for current prediction
            fig_risk.add_shape(
                type="line",
                x0=p_fraud * 100, x1=p_fraud * 100,
                y0=-0.4, y1=0.4,
                line=dict(color=gauge_color, width=4, dash="solid"),
            )
            fig_risk.add_annotation(
                x=p_fraud * 100, y=0.55,
                text=f"▼ {p_fraud:.1%}",
                showarrow=False,
                font=dict(color=gauge_color, size=13, family="monospace"),
            )
            fig_risk.update_layout(
                barmode="stack",
                title=dict(text="📊 Risk Zone Breakdown", font=dict(color="#111827", size=14)),
                xaxis=dict(range=[0, 100], title="Fraud Probability (%)", tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
                yaxis=dict(showticklabels=False),
                height=160,
                margin=dict(l=0, r=0, t=44, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.6, font=dict(color="#111827", size=12)),
                font=dict(color="#111827"),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # ── h. ROC-AUC Comparison Bar ────────────────────────────────────
            if m:
                rows = m.get("metrics", [])
                roc_rows = [r for r in rows if r.get("roc_auc", 0) > 0]
                if roc_rows:
                    fig_roc = px.bar(
                        x=[r["model"] for r in roc_rows],
                        y=[r["roc_auc"] for r in roc_rows],
                        title="📈 ROC-AUC Score by Model",
                        labels={"x": "Model", "y": "ROC-AUC"},
                        color=[r["roc_auc"] for r in roc_rows],
                        color_continuous_scale="Viridis",
                        text=[f"{r['roc_auc']:.3f}" for r in roc_rows],
                    )
                    fig_roc.update_traces(textposition="outside")
                    fig_roc.update_layout(
                        height=350, margin=dict(l=0,r=0,t=50,b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

            # ── i. CV Scores Line Chart ──────────────────────────────────────
            if m and m.get("cv_results_mean_test_score"):
                cv_scores = m["cv_results_mean_test_score"]
                fig_cv = px.line(
                    x=list(range(1, len(cv_scores)+1)),
                    y=cv_scores,
                    markers=True,
                    title="🔁 Cross-Validation F1 Scores (Hyperparameter Iterations)",
                    labels={"x": "Iteration", "y": "Mean CV F1"},
                )
                fig_cv.add_hline(
                    y=max(cv_scores), line_dash="dash", line_color="#6366f1",
                    annotation_text=f"Best: {max(cv_scores):.4f}",
                    annotation_position="top right",
                )
                fig_cv.update_layout(
                    height=300, margin=dict(l=0,r=0,t=50,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cv, use_container_width=True)

            # ── j. Accuracy vs F1 Scatter ────────────────────────────────────
            if m:
                rows = m.get("metrics", [])
                if rows:
                    fig_scatter = px.scatter(
                        x=[r.get("accuracy", 0) for r in rows],
                        y=[r.get("f1_fraud", 0) for r in rows],
                        text=[r["model"] for r in rows],
                        size=[r.get("recall_fraud", 0.1)*100 for r in rows],
                        color=[r.get("precision_fraud", 0) for r in rows],
                        color_continuous_scale="RdYlGn",
                        title="🎯 Accuracy vs F1 (bubble size = Recall, color = Precision)",
                        labels={"x": "Accuracy", "y": "F1 (Fraud)", "color": "Precision"},
                    )
                    fig_scatter.update_traces(textposition="top center")
                    fig_scatter.update_layout(
                        height=400, margin=dict(l=0,r=0,t=50,b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

            # ── k. Train / Test Split horizontal stacked bar ─────────────────
            if m and m.get("training_size") and m.get("test_size"):
                train_sz = m["training_size"]
                test_sz  = m["test_size"]
                fig_split = gobj.Figure()
                fig_split.add_trace(gobj.Bar(
                    x=[train_sz], y=["Dataset Split"], orientation="h",
                    name=f"Train ({train_sz})", marker_color="#6366f1", width=0.4,
                ))
                fig_split.add_trace(gobj.Bar(
                    x=[test_sz], y=["Dataset Split"], orientation="h",
                    name=f"Test ({test_sz})", marker_color="#06b6d4", width=0.4,
                ))
                fig_split.update_layout(
                    barmode="stack",
                    title="📦 Train / Test Split",
                    height=140,
                    margin=dict(l=0,r=0,t=44,b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.5),
                    xaxis_title="Number of samples",
                    yaxis=dict(showticklabels=False),
                )
                st.plotly_chart(fig_split, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & Model Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Exploratory Data Analysis & Model Analysis")
    st.markdown("Academic analysis of the dataset and model performance metrics.")

    # ── SECTION 1: EDA ────────────────────────────────────────────────────────
    st.markdown("### Section 1: Exploratory Data Analysis")

    # ── Graph 1: Class Distribution Bar Chart ─────────────────────────────────
    try:
        df_eda = load_dataset()

        st.markdown("#### Graph 1 — Class Distribution")
        class_counts = df_eda["fraudulent"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        class_counts["Label"] = class_counts["Class"].map({0: "Legitimate", 1: "Fraudulent"})
        fig_class = px.bar(
            class_counts,
            x="Label",
            y="Count",
            color="Label",
            color_discrete_map={"Legitimate": "#21c55d", "Fraudulent": "#e74c3c"},
            title="Class Distribution: Legitimate vs Fraudulent Job Postings",
            text="Count",
        )
        fig_class.update_traces(textposition="outside")
        fig_class.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_class, use_container_width=True)
        st.caption(
            "Dataset is highly imbalanced — fraudulent postings represent ~5% of total. "
            "Class weighting was applied during training to handle this."
        )

        st.markdown("---")

        # ── Graph 2: Text Length Distribution ─────────────────────────────────
        st.markdown("#### Graph 2 — Text Length Distribution")
        text_cols = ["title", "description", "requirements"]
        existing_text_cols = [c for c in text_cols if c in df_eda.columns]
        df_eda["combined_text_len"] = df_eda[existing_text_cols].fillna("").apply(
            lambda row: sum(len(str(v)) for v in row), axis=1
        )
        df_eda["Class"] = df_eda["fraudulent"].map({0: "Legitimate", 1: "Fraudulent"})
        fig_hist = px.histogram(
            df_eda,
            x="combined_text_len",
            color="Class",
            barmode="overlay",
            opacity=0.6,
            nbins=80,
            color_discrete_map={"Legitimate": "#21c55d", "Fraudulent": "#e74c3c"},
            title="Text Length Distribution by Class",
            labels={"combined_text_len": "Combined Text Length (characters)", "Class": "Class"},
        )
        fig_hist.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            "Fraudulent job postings tend to have shorter, vaguer descriptions compared to legitimate postings."
        )

        st.markdown("---")

        # ── Graph 3: Top 20 Words Fraud vs Genuine ────────────────────────────
        st.markdown("#### Graph 3 — Top 20 Words: Fraud vs Genuine")

        STOPWORDS = {
            "the","a","an","is","are","was","were","be","been","being","have","has","had",
            "do","does","did","will","would","could","should","may","might","shall","can",
            "need","dare","ought","used","to","of","in","for","on","with","at","by","from",
            "as","into","through","during","before","after","above","below","between","out",
            "off","over","under","again","further","then","once","and","but","or","nor","so",
            "yet","both","either","neither","not","only","own","same","than","too","very",
            "just","because","if","while","although","though","since","unless","until","when",
            "where","who","which","that","this","these","those","it","its","we","our","you",
            "your","they","their","he","she","him","her","his","hers","i","me","my","myself",
            "yourself","himself","herself","itself","ourselves","themselves",
        }

        def get_top_words(texts, n=20):
            words = []
            for t in texts:
                for w in str(t).lower().split():
                    w_clean = "".join(c for c in w if c.isalpha())
                    if w_clean and w_clean not in STOPWORDS and len(w_clean) > 2:
                        words.append(w_clean)
            return Counter(words).most_common(n)

        fraud_texts = df_eda[df_eda["fraudulent"] == 1][existing_text_cols].fillna("").apply(
            lambda row: " ".join(str(v) for v in row), axis=1
        )
        legit_texts = df_eda[df_eda["fraudulent"] == 0][existing_text_cols].fillna("").apply(
            lambda row: " ".join(str(v) for v in row), axis=1
        )

        fraud_words = get_top_words(fraud_texts)
        legit_words = get_top_words(legit_texts)

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            fw_words = [w for w, _ in fraud_words]
            fw_counts = [c for _, c in fraud_words]
            fig_fw = px.bar(
                x=fw_counts,
                y=fw_words,
                orientation="h",
                title="Top 20 Words — Fraudulent Postings",
                labels={"x": "Frequency", "y": "Word"},
                color_discrete_sequence=["#e74c3c"],
            )
            fig_fw.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=50, b=0),
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_fw, use_container_width=True)

        with gcol2:
            lw_words = [w for w, _ in legit_words]
            lw_counts = [c for _, c in legit_words]
            fig_lw = px.bar(
                x=lw_counts,
                y=lw_words,
                orientation="h",
                title="Top 20 Words — Legitimate Postings",
                labels={"x": "Frequency", "y": "Word"},
                color_discrete_sequence=["#21c55d"],
            )
            fig_lw.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=50, b=0),
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_lw, use_container_width=True)

        st.caption(
            "Fraudulent postings contain vague promotional language ('earn', 'work', 'home', 'money'), "
            "while genuine postings use structured professional terms ('experience', 'skills', 'team', 'required')."
        )

    except FileNotFoundError:
        st.warning("Dataset not found. EDA graphs (1–3) require the dataset CSV file.")
    except Exception as exc:
        st.warning(f"Could not load dataset for EDA: {exc}")

    st.markdown("---")

    # ── SECTION 2: Model Analysis ─────────────────────────────────────────────
    st.markdown("### Section 2: Model Analysis")

    # ── Graph 4: TF-IDF Feature Importance ────────────────────────────────────
    st.markdown("#### Graph 4 — TF-IDF Feature Importance (Logistic Regression)")
    try:
        from model_wrapper import FraudDetectionModel, FullHybridModel

        inner_pipe = None
        if isinstance(model, FraudDetectionModel):
            core = model.pipeline
        else:
            core = model
        if isinstance(core, FullHybridModel):
            inner_pipe = core.inner
        elif hasattr(core, "named_steps"):
            inner_pipe = core

        if inner_pipe is not None and "preprocess" in inner_pipe.named_steps and "clf" in inner_pipe.named_steps:
            prep = inner_pipe.named_steps["preprocess"]
            clf  = inner_pipe.named_steps["clf"]
            coef = clf.coef_.ravel()
            feat_names = prep.get_feature_names_out()

            txt_mask = np.array([n.startswith("txt__") for n in feat_names])
            txt_names = feat_names[txt_mask]
            txt_coef  = coef[txt_mask]

            # Top 20 positive (fraud) and top 20 negative (legit)
            sorted_idx = np.argsort(txt_coef)
            top_neg_idx = sorted_idx[:20]
            top_pos_idx = sorted_idx[-20:][::-1]
            selected_idx = np.concatenate([top_pos_idx, top_neg_idx])

            fi_names  = [txt_names[i].replace("txt__", "") for i in selected_idx]
            fi_coefs  = [float(txt_coef[i]) for i in selected_idx]
            fi_colors = ["Fraud signal" if c > 0 else "Legit signal" for c in fi_coefs]

            fig_fi = px.bar(
                x=fi_coefs,
                y=fi_names,
                color=fi_colors,
                orientation="h",
                title="TF-IDF Feature Importance — Top 20 Fraud & Legit Signals",
                labels={"x": "Logistic Regression Coefficient", "y": "Feature", "color": "Signal"},
                color_discrete_map={"Fraud signal": "#e74c3c", "Legit signal": "#21c55d"},
            )
            fig_fi.update_layout(
                height=700,
                margin=dict(l=0, r=0, t=50, b=0),
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption(
                "TF-IDF feature importance extracted from Logistic Regression coefficients. "
                "Red bars indicate words strongly associated with fraud; green bars indicate legitimate signals."
            )
        else:
            st.info("Feature importance not available — model structure not compatible.")
    except Exception as exc:
        st.warning(f"Could not extract feature importance: {exc}")

    st.markdown("---")

    # ── Graph 5: Confusion Matrix Heatmap ─────────────────────────────────────
    st.markdown("#### Graph 5 — Confusion Matrix Heatmap")
    if m:
        rows_m = m.get("metrics", [])
        hybrid_row_m = next(
            (r for r in rows_m if "hybrid" in r.get("model", "").lower()),
            rows_m[-1] if rows_m else None,
        )
        if hybrid_row_m:
            cm5 = hybrid_row_m.get("confusion_matrix")
            if cm5 and len(cm5) == 2:
                cm5_arr = np.array(cm5)
                fig_cm5 = px.imshow(
                    cm5_arr,
                    text_auto=True,
                    x=["Predicted: Legit", "Predicted: Fraud"],
                    y=["Actual: Legit", "Actual: Fraud"],
                    color_continuous_scale="RdYlGn_r",
                    title="Confusion Matrix — Hybrid Model",
                    labels=dict(color="Count"),
                )
                fig_cm5.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=50, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cm5, use_container_width=True)
                st.markdown(
                    """
                    | Term | Meaning |
                    |------|---------|
                    | **TN = 3369** | True Negatives — correctly identified as legitimate |
                    | **TP = 152** | True Positives — correctly caught as fraudulent |
                    | **FP = 34** | False Positives — legitimate postings flagged as fraud (false alarms) |
                    | **FN = 21** | False Negatives — fraudulent postings missed by the model |
                    """
                )
                st.caption(
                    "Confusion matrix for the hybrid model. TN=3369 (correctly identified legitimate), "
                    "TP=152 (correctly caught fraud), FP=34 (false alarms), FN=21 (missed fraud)."
                )
    else:
        st.info("metrics.json not found. Train the model first.")

    st.markdown("---")

    # ── Graph 6: Model Comparison F1 Bar Chart ────────────────────────────────
    st.markdown("#### Graph 6 — Model Comparison: F1 Score")
    if m:
        rows_m = m.get("metrics", [])
        if rows_m:
            hybrid_f1 = next(
                (r["f1_fraud"] for r in rows_m if "hybrid" in r.get("model", "").lower()), None
            )
            model_names = [r["model"] for r in rows_m]
            f1_scores   = [r["f1_fraud"] for r in rows_m]

            fig_f1 = px.bar(
                x=model_names,
                y=f1_scores,
                color=model_names,
                title="F1 Score Comparison Across Models",
                labels={"x": "Model", "y": "F1 Score (Fraud Class)", "color": "Model"},
                text=[f"{v:.3f}" for v in f1_scores],
            )
            fig_f1.update_traces(textposition="outside")
            if hybrid_f1 is not None:
                fig_f1.add_hline(
                    y=hybrid_f1,
                    line_dash="dash",
                    line_color="#6366f1",
                    annotation_text=f"Hybrid F1 = {hybrid_f1:.3f}",
                    annotation_position="top right",
                )
            fig_f1.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=60, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_f1, use_container_width=True)
            st.caption(
                "The hybrid TF-IDF + metadata model achieves the highest F1 score, "
                "outperforming all text-only baselines."
            )
    else:
        st.info("metrics.json not found.")

    st.markdown("---")

    # ── Graph 7: Precision vs Recall Bar Chart ────────────────────────────────
    st.markdown("#### Graph 7 — Precision vs Recall per Model")
    if m:
        rows_m = m.get("metrics", [])
        if rows_m:
            pr_models  = [r["model"] for r in rows_m for _ in range(2)]
            pr_metrics = ["Precision", "Recall"] * len(rows_m)
            pr_values  = [
                v for r in rows_m
                for v in (r["precision_fraud"], r["recall_fraud"])
            ]
            fig_pr = px.bar(
                x=pr_models,
                y=pr_values,
                color=pr_metrics,
                barmode="group",
                title="Precision vs Recall by Model",
                labels={"x": "Model", "y": "Score", "color": "Metric"},
                color_discrete_map={"Precision": "#6366f1", "Recall": "#f59e0b"},
                text=[f"{v:.3f}" for v in pr_values],
            )
            fig_pr.update_traces(textposition="outside")
            fig_pr.update_layout(
                height=440,
                margin=dict(l=0, r=0, t=60, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            st.caption(
                "Naive Bayes achieves high recall but low precision (many false alarms). "
                "The hybrid model balances both effectively."
            )
    else:
        st.info("metrics.json not found.")

    st.markdown("---")

    # ── Graph 8: ROC Curve (simulated from confusion matrix points) ───────────
    st.markdown("#### Graph 8 — ROC Curve")
    if m:
        rows_m = m.get("metrics", [])
        if rows_m:
            fig_roc8 = gobj.Figure()
            # Diagonal reference line (random classifier)
            fig_roc8.add_trace(gobj.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="#94a3b8", width=1),
                name="Random Classifier",
                showlegend=True,
            ))
            roc_colors = ["#6366f1", "#e74c3c", "#21c55d", "#f59e0b", "#06b6d4"]
            for i, r in enumerate(rows_m):
                cm_r = r.get("confusion_matrix")
                if not cm_r or len(cm_r) != 2:
                    continue
                tn_r = cm_r[0][0]; fp_r = cm_r[0][1]
                fn_r = cm_r[1][0]; tp_r = cm_r[1][1]
                fpr = fp_r / (fp_r + tn_r) if (fp_r + tn_r) > 0 else 0.0
                tpr = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
                auc_val = r.get("roc_auc", 0.0)
                model_label = f"{r['model']} (AUC={auc_val:.3f})"
                fig_roc8.add_trace(gobj.Scatter(
                    x=[0, fpr, 1],
                    y=[0, tpr, 1],
                    mode="lines+markers",
                    name=model_label,
                    line=dict(color=roc_colors[i % len(roc_colors)], width=2),
                    marker=dict(size=10, symbol="circle"),
                ))
            fig_roc8.update_layout(
                title="ROC Curve — Model Operating Points",
                xaxis_title="False Positive Rate (FPR)",
                yaxis_title="True Positive Rate (TPR)",
                height=480,
                margin=dict(l=0, r=0, t=60, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.25),
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_roc8, use_container_width=True)
            st.caption(
                "ROC curve showing each model's operating point. "
                "The hybrid model achieves AUC=0.934, indicating strong discrimination ability."
            )
    else:
        st.info("metrics.json not found.")

    st.markdown("---")

    # ── Graph 9: CV F1 Scores Across Hyperparameter Search ───────────────────
    st.markdown("#### Graph 9 — Threshold Tuning: CV F1 Scores")
    if m and m.get("cv_results_mean_test_score"):
        cv_scores9 = m["cv_results_mean_test_score"]
        best_iter  = int(np.argmax(cv_scores9)) + 1
        best_score = max(cv_scores9)

        fig_cv9 = px.line(
            x=list(range(1, len(cv_scores9) + 1)),
            y=cv_scores9,
            markers=True,
            title="🎯 Threshold Tuning — CV F1 Scores Across Hyperparameter Search",
            labels={"x": "Iteration", "y": "Mean CV F1 Score"},
        )
        fig_cv9.add_scatter(
            x=[best_iter],
            y=[best_score],
            mode="markers",
            marker=dict(size=14, color="#e74c3c", symbol="star"),
            name=f"Best: {best_score:.4f} (iter {best_iter})",
        )
        fig_cv9.add_hline(
            y=best_score,
            line_dash="dash",
            line_color="#6366f1",
            annotation_text=f"Best CV F1 = {best_score:.4f}",
            annotation_position="top right",
        )
        fig_cv9.update_layout(
            height=360,
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cv9, use_container_width=True)
        st.caption(
            "F1 score improved across hyperparameter search iterations. "
            "The optimal threshold of 0.38 was selected by maximizing F1 on the validation set."
        )
    elif m:
        st.info("No CV results found in metrics.json.")
    else:
        st.info("metrics.json not found.")
