"""Paths and constants for the fake job detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_CANDIDATES = ("DataSet.csv", "Dataset.csv", "dataset.csv")


def resolve_dataset_path() -> Path:
    for name in DATA_CANDIDATES:
        p = ROOT / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No dataset found in {ROOT}. Expected one of: {', '.join(DATA_CANDIDATES)}"
    )


MODEL_PATH = MODELS_DIR / "fraud_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

TEXT_COLUMNS = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

RANDOM_STATE = 42
