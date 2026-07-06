"""
Shared utility functions used across cascade-gnn's GNN models,
simulation, and API — consolidated here to avoid scattered duplicate
definitions across gcn.py, graphsage.py, gat.py, cascade.py, etc.
"""

import torch


def normalize_graph(data):
    """
    Z-score normalize node features (across the whole graph) and edge
    features (using TRAINING edges only, to avoid leakage into val/test).
    """
    for node_type in data.node_types:
        x = data[node_type].x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-6
        data[node_type].x = (x - mean) / std

    train_mask = data["customer", "orders_from", "category"].train_mask
    edge_attr = data["customer", "orders_from", "category"].edge_attr
    train_mean = edge_attr[train_mask].mean(dim=0, keepdim=True)
    train_std = edge_attr[train_mask].std(dim=0, keepdim=True) + 1e-6

    data["customer", "orders_from", "category"].edge_attr = (edge_attr - train_mean) / train_std
    data["category", "rev_orders_from", "customer"].edge_attr = (
        data["category", "rev_orders_from", "customer"].edge_attr - train_mean
    ) / train_std

    return data


def get_node_embeddings(model, data):
    """Run a model's encoder once, return embeddings for all node types."""
    with torch.no_grad():
        x_dict = model.encoder(data.x_dict, data.edge_index_dict)
    return x_dict