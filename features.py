"""Tabular + text feature table for hybrid fraud detection."""
from __future__ import annotations

import pandas as pd

from config import TEXT_COLUMNS
from preprocess import combine_text_columns, fill_missing

META_BINARY = ["telecommuting", "has_company_logo", "has_questions"]
META_CATEGORICAL = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined_text; normalize binary and categorical columns."""
    out = df.copy()
    out = fill_missing(out, TEXT_COLUMNS)
    out["combined_text"] = combine_text_columns(
        out, [c for c in TEXT_COLUMNS if c in out.columns]
    )
    for c in META_BINARY:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(float)
        else:
            out[c] = 0.0
    for c in META_CATEGORICAL:
        if c in out.columns:
            out[c] = out[c].fillna("Unknown").astype(str).str.strip()
            out[c] = out[c].replace("", "Unknown")
        else:
            out[c] = "Unknown"
    return out


def training_columns() -> list[str]:
    return ["combined_text", *META_BINARY, *META_CATEGORICAL]


def frame_from_user_fields(
    *,
    title: str = "",
    location: str = "",
    department: str = "",
    company_profile: str = "",
    description: str = "",
    requirements: str = "",
    benefits: str = "",
    telecommuting: float | None = None,
    has_company_logo: float | None = None,
    has_questions: float | None = None,
    employment_type: str | None = None,
    required_experience: str | None = None,
    required_education: str | None = None,
    industry: str | None = None,
    job_function: str | None = None,
) -> pd.DataFrame:
    """Single-row DataFrame aligned with training (unknown meta → neutral defaults)."""
    parts = {
        "title": title or "",
        "location": location or "",
        "department": department or "",
        "company_profile": company_profile or "",
        "description": description or "",
        "requirements": requirements or "",
        "benefits": benefits or "",
    }
    row = {**parts}
    row["telecommuting"] = 0.0 if telecommuting is None else float(telecommuting)
    row["has_company_logo"] = 0.0 if has_company_logo is None else float(has_company_logo)
    row["has_questions"] = 0.0 if has_questions is None else float(has_questions)
    row["employment_type"] = employment_type or "Unknown"
    row["required_experience"] = required_experience or "Unknown"
    row["required_education"] = required_education or "Unknown"
    row["industry"] = industry or "Unknown"
    row["function"] = job_function or "Unknown"
    df = pd.DataFrame([row])
    return enrich_dataframe(df)[training_columns()]
