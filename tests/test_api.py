"""
Tests for the FastAPI risk-scoring service (src/api/main.py).
Uses FastAPI's TestClient to call the app in-process, without needing
a live running server.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest

if not os.path.exists("data/processed/graph.pt") or not os.path.exists("data/processed/gcn_weights.pt"):
    pytest.skip(
        "graph.pt or gcn_weights.pt not found — run src/graph/builder.py and src/models/gcn.py first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_risk_score_valid_request():
    response = client.post("/risk-score", json={"customer_id": 0, "category_id": 0})
    assert response.status_code == 200

    body = response.json()
    assert "risk_score" in body
    assert "risk_level" in body
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}


def test_risk_score_varies_across_inputs():
    r1 = client.post("/risk-score", json={"customer_id": 0, "category_id": 0})
    r2 = client.post("/risk-score", json={"customer_id": 5000, "category_id": 30})

    score1 = r1.json()["risk_score"]
    score2 = r2.json()["risk_score"]

    # Different inputs should not produce the exact same score —
    # this would indicate the model isn't actually using the input.
    assert score1 != score2


def test_risk_score_rejects_out_of_range_customer_id():
    response = client.post("/risk-score", json={"customer_id": 999999, "category_id": 0})
    assert response.status_code == 400


def test_risk_score_rejects_out_of_range_category_id():
    response = client.post("/risk-score", json={"customer_id": 0, "category_id": 999})
    assert response.status_code == 400


def test_risk_score_rejects_missing_fields():
    response = client.post("/risk-score", json={"customer_id": 0})
    assert response.status_code == 422  # FastAPI/Pydantic validation error