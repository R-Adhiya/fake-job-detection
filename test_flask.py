"""Smoke-test Flask API without starting a server (uses Flask test client)."""
from __future__ import annotations

import json

from flask_app import app


def main() -> None:
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    print("GET /health", r.get_json())

    payload = {
        "title": "Urgent work from home",
        "description": "Send fee via western union to receive starter kit. Guaranteed income.",
        "company_profile": "",
        "requirements": "",
        "benefits": "",
    }
    r = client.post("/predict", json=payload)
    print("POST /predict", r.status_code, json.dumps(r.get_json(), indent=2))


if __name__ == "__main__":
    main()
