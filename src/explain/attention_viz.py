"""
Extract and inspect GAT attention weights for cascade-gnn.

Investigates whether GAT's attention weights are meaningfully varied
(picking out specific important neighbors) or close to uniform — which
would help explain why GAT didn't outperform mean-aggregation models
(GCN/GraphSAGE) on this graph.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from src.models.gat import GATModel
from src.models.gcn import normalize_graph


def train_gat(data, epochs=150):
    torch.manual_seed(42)
    edge_feat_dim = data["customer", "orders_from", "category"].edge_attr.shape[1]
    model = GATModel(hidden_dim=32, edge_feat_dim=edge_feat_dim, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    edge_store = data["customer", "orders_from", "category"]
    train_mask = edge_store.train_mask
    y = edge_store.y.float()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(data)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def extract_trained_attention(model, data):
    """
    Pull attention weights directly from the trained model's first-layer
    GATConv for the customer->category edge type.
    """
    edge_store = data["customer", "orders_from", "category"]
    edge_index = edge_store.edge_index

    x_src = data["customer"].x
    x_dst = data["category"].x

    conv = model.encoder.conv1.convs[("customer", "orders_from", "category")]

    with torch.no_grad():
        out, (edge_index_out, alpha) = conv(
            (x_src, x_dst), edge_index, return_attention_weights=True
        )

    return alpha


if __name__ == "__main__":
    data = torch.load("data/processed/graph.pt", weights_only=False)
    data = normalize_graph(data)

    print("Training GAT...")
    model = train_gat(data)

    print("Extracting trained attention weights...")
    alpha = extract_trained_attention(model, data)

    print(f"Attention weight tensor shape: {alpha.shape}")
    print()
    print("--- Trained attention weight statistics ---")
    print(f"Mean: {alpha.mean().item():.6f}")
    print(f"Std:  {alpha.std().item():.6f}")
    print(f"Min:  {alpha.min().item():.6f}")
    print(f"Max:  {alpha.max().item():.6f}")

    # Compare against what perfectly UNIFORM attention would look like,
    # given each category's actual number of neighbors
    from collections import Counter
    dst_counts = Counter(edge_index_out_list if False else data["customer", "orders_from", "category"].edge_index[1].tolist())
    avg_degree = sum(dst_counts.values()) / len(dst_counts)
    uniform_weight = 1.0 / avg_degree
    print()
    print(f"Average category degree: {avg_degree:.1f}")
    print(f"'Uniform attention' baseline would be: {uniform_weight:.6f}")
    print(f"Ratio of actual std to uniform baseline: {(alpha.std().item() / uniform_weight):.2f}x")