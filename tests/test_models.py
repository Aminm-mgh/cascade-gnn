"""
Tests for GNN models (src/models/gcn.py, graphsage.py, gat.py).
Structural tests only — verifying models run and produce correctly
shaped output. Performance/accuracy is evaluated separately by the
training scripts themselves.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.models.utils import normalize_graph


@pytest.fixture(scope="module")
def graph_data():
    path = "data/processed/graph.pt"
    if not os.path.exists(path):
        pytest.skip("graph.pt not found — run src/graph/builder.py first")
    data = torch.load(path, weights_only=False)
    return normalize_graph(data)


def test_gcn_forward_pass(graph_data):
    from src.models.gcn import GCNModel

    edge_feat_dim = graph_data["customer", "orders_from", "category"].edge_attr.shape[1]
    model = GCNModel(hidden_dim=32, edge_feat_dim=edge_feat_dim)

    with torch.no_grad():
        logits = model(graph_data)

    num_edges = graph_data["customer", "orders_from", "category"].edge_index.shape[1]
    assert logits.shape == (num_edges,)
    assert not torch.isnan(logits).any()


def test_graphsage_forward_pass(graph_data):
    from src.models.graphsage import SAGEModel

    edge_feat_dim = graph_data["customer", "orders_from", "category"].edge_attr.shape[1]
    model = SAGEModel(hidden_dim=32, edge_feat_dim=edge_feat_dim)

    with torch.no_grad():
        logits = model(graph_data)

    num_edges = graph_data["customer", "orders_from", "category"].edge_index.shape[1]
    assert logits.shape == (num_edges,)
    assert not torch.isnan(logits).any()


def test_gat_forward_pass(graph_data):
    from src.models.gat import GATModel

    edge_feat_dim = graph_data["customer", "orders_from", "category"].edge_attr.shape[1]
    model = GATModel(hidden_dim=32, edge_feat_dim=edge_feat_dim, heads=4)

    with torch.no_grad():
        logits = model(graph_data)

    num_edges = graph_data["customer", "orders_from", "category"].edge_index.shape[1]
    assert logits.shape == (num_edges,)
    assert not torch.isnan(logits).any()