"""
Tests for graph construction (src/graph/builder.py).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


@pytest.fixture(scope="module")
def graph_data():
    """Load the pre-built graph once for all tests in this file."""
    path = "data/processed/graph.pt"
    if not os.path.exists(path):
        pytest.skip("graph.pt not found — run src/graph/builder.py first")
    return torch.load(path, weights_only=False)


def test_node_counts(graph_data):
    assert graph_data["category"].x.shape[0] == 50
    assert graph_data["customer"].x.shape[0] > 0


def test_edge_shapes_match(graph_data):
    edge_store = graph_data["customer", "orders_from", "category"]
    num_edges = edge_store.edge_index.shape[1]
    assert edge_store.edge_attr.shape[0] == num_edges
    assert edge_store.y.shape[0] == num_edges


def test_no_nans_in_features(graph_data):
    assert not torch.isnan(graph_data["category"].x).any()
    assert not torch.isnan(graph_data["customer"].x).any()
    edge_attr = graph_data["customer", "orders_from", "category"].edge_attr
    assert not torch.isnan(edge_attr).any()


def test_split_masks_cover_all_edges_exactly_once(graph_data):
    edge_store = graph_data["customer", "orders_from", "category"]
    train_mask = edge_store.train_mask
    val_mask = edge_store.val_mask
    test_mask = edge_store.test_mask

    total_flags = train_mask.long() + val_mask.long() + test_mask.long()
    assert (total_flags == 1).all()


def test_split_proportions_roughly_80_10_10(graph_data):
    edge_store = graph_data["customer", "orders_from", "category"]
    n = edge_store.train_mask.shape[0]

    train_frac = edge_store.train_mask.sum().item() / n
    val_frac = edge_store.val_mask.sum().item() / n
    test_frac = edge_store.test_mask.sum().item() / n

    assert abs(train_frac - 0.8) < 0.01
    assert abs(val_frac - 0.1) < 0.01
    assert abs(test_frac - 0.1) < 0.01


def test_reverse_edges_exist(graph_data):
    assert ("category", "rev_orders_from", "customer") in graph_data.edge_types


def test_label_is_binary(graph_data):
    y = graph_data["customer", "orders_from", "category"].y
    unique_vals = set(y.unique().tolist())
    assert unique_vals.issubset({0, 1})