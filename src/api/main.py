"""
FastAPI service for cascade-gnn.

POST /risk-score — submit a customer/category node pair, get back a
predicted late-delivery risk score using the trained GCN.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.gcn import GCNModel, normalize_graph, get_node_embeddings

app = FastAPI(title="cascade-gnn API")

# --- Load graph and trained model once, at startup ---
DATA_PATH = "data/processed/graph.pt"
WEIGHTS_PATH = "data/processed/gcn_weights.pt"

data = torch.load(DATA_PATH, weights_only=False)
data = normalize_graph(data)

edge_feat_dim = data["customer", "orders_from", "category"].edge_attr.shape[1]
model = GCNModel(hidden_dim=32, edge_feat_dim=edge_feat_dim)

# Run once to initialize lazy layers before loading weights
with torch.no_grad():
    model(data)
model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
model.eval()

with torch.no_grad():
    x_dict = model.encoder(data.x_dict, data.edge_index_dict)


class RiskScoreRequest(BaseModel):
    customer_id: int
    category_id: int


class RiskScoreResponse(BaseModel):
    customer_id: int
    category_id: int
    risk_score: float
    risk_level: str


@app.get("/")
def root():
    return {"status": "cascade-gnn API is running"}


@app.post("/risk-score", response_model=RiskScoreResponse)
def risk_score(request: RiskScoreRequest):
    num_customers = data["customer"].x.shape[0]
    num_categories = data["category"].x.shape[0]

    if not (0 <= request.customer_id < num_customers):
        raise HTTPException(status_code=400, detail=f"customer_id must be between 0 and {num_customers - 1}")
    if not (0 <= request.category_id < num_categories):
        raise HTTPException(status_code=400, detail=f"category_id must be between 0 and {num_categories - 1}")

    cust_emb = x_dict["customer"][request.customer_id].unsqueeze(0)
    cat_emb = x_dict["category"][request.category_id].unsqueeze(0)

    # Use average edge features as a stand-in, since we don't have a
    # specific real order to reference for an arbitrary customer/category pair
    avg_edge_attr = data["customer", "orders_from", "category"].edge_attr.mean(dim=0, keepdim=True)

    with torch.no_grad():
        logit = model.classifier(cust_emb, cat_emb, avg_edge_attr)
        prob = torch.sigmoid(logit).item()

    if prob < 0.4:
        level = "low"
    elif prob < 0.6:
        level = "medium"
    else:
        level = "high"

    return RiskScoreResponse(
        customer_id=request.customer_id,
        category_id=request.category_id,
        risk_score=round(prob, 4),
        risk_level=level,
    )