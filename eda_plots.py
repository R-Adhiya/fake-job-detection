"""
Generate EDA figures for the report: class balance, text length, heatmap,
employment vs fraud, WordClouds (legit vs fraud). Run: python eda_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import TEXT_COLUMNS, resolve_dataset_path
from preprocess import combine_text_columns, fill_missing

FIG_DIR = Path(__file__).resolve().parent / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    path = resolve_dataset_path()
    df = pd.read_csv(path)
    df = fill_missing(df, TEXT_COLUMNS)
    df["combined_text"] = combine_text_columns(df, [c for c in TEXT_COLUMNS if c in df.columns])
    df["text_len"] = df["combined_text"].str.len().clip(upper=50000)

    # 1) Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df["fraudulent"].value_counts().sort_index()
    labs = ["Legitimate (0)", "Fraudulent (1)"]
    ax.bar(labs, [vc.get(0, 0), vc.get(1, 0)], color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Count")
    ax.set_title("Class distribution")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png", dpi=150)
    plt.close()

    # 2) Text length distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        subset = df.loc[df["fraudulent"] == label, "text_len"]
        sns.kdeplot(subset, fill=True, ax=ax, color=color, alpha=0.4, label=f"fraudulent={label}")
    ax.set_xlabel("Combined text length (characters)")
    ax.set_title("Text length distribution by class")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_text_length_kde.png", dpi=150)
    plt.close()

    # 3) Correlation heatmap (numeric / binary columns)
    num_cols = [
        c
        for c in (
            "telecommuting",
            "has_company_logo",
            "has_questions",
            "fraudulent",
        )
        if c in df.columns
    ]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation heatmap (numeric fields)")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=150)
        plt.close()

    # 4) Employment type vs fraud (normalized stacked bar)
    if "employment_type" in df.columns:
        ct = pd.crosstab(df["employment_type"].fillna("Unknown"), df["fraudulent"], normalize="index")
        fig, ax = plt.subplots(figsize=(10, 5))
        ct.plot(kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"])
        ax.set_ylabel("Proportion")
        ax.set_xlabel("Employment type")
        ax.set_title("Fraud rate by employment type (row-normalized)")
        ax.legend(["Legit", "Fraud"])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "04_employment_vs_fraud.png", dpi=150)
        plt.close()

    # 5) WordClouds (sample to keep runtime reasonable)
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud not installed; skip WordClouds. pip install wordcloud")
        print(f"Done. Figures saved under {FIG_DIR}")
        return

    def wc_for(mask: pd.Series, name: str) -> None:
        text = " ".join(df.loc[mask, "combined_text"].sample(min(3000, mask.sum()), random_state=42))
        wc = WordCloud(width=1200, height=600, background_color="white", max_words=200).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(name)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"05_wordcloud_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

    wc_for(df["fraudulent"] == 0, "Legitimate")
    wc_for(df["fraudulent"] == 1, "Fraudulent")

    print(f"Done. Figures saved under {FIG_DIR}")


if __name__ == "__main__":
    main()
