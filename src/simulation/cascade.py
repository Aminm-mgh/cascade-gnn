"""
Risk propagation / cascade simulation for cascade-gnn.

Simulates a disruption starting at one node and propagating outward,
using the trained GCN's predicted risk scores as the "infection
probability" along each edge — reusing the actual trained model rather
than a separate simulation-specific classifier.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from src.models.gcn import GCNModel, normalize_graph


def load_trained_model(data):
    """
    Recreate and retrain the GCN quickly for simulation use.
    (In a more mature version, we'd save/load trained weights directly —
    flagged as a TODO for later polish.)
    """
    torch.manual_seed(42)
    edge_feat_dim = data["customer", "orders_from", "category"].edge_attr.shape[1]
    model = GCNModel(hidden_dim=32, edge_feat_dim=edge_feat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    edge_store = data["customer", "orders_from", "category"]
    train_mask = edge_store.train_mask
    y = edge_store.y.float()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(data)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def get_node_embeddings(model, data):
    """Run the encoder once, return embeddings for all nodes."""
    with torch.no_grad():
        x_dict = model.encoder(data.x_dict, data.edge_index_dict)
    return x_dict


def simulate_cascade(model, data, start_node_type, start_node_id, threshold=0.5, max_hops=5):
    """
    Simulate disruption spreading from a single starting node.

    Returns a dict: {(node_type, node_id): hop_reached}
    """
    x_dict = get_node_embeddings(model, data)

    edge_store = data["customer", "orders_from", "category"]
    cust_to_cat_edges = edge_store.edge_index  # [2, E]: customer_idx, category_idx
    edge_attr = edge_store.edge_attr

    # Build adjacency: for quick lookup of which edges touch a given node
    from collections import defaultdict
    customer_edges = defaultdict(list)  # customer_idx -> list of (edge_idx, category_idx)
    category_edges = defaultdict(list)  # category_idx -> list of (edge_idx, customer_idx)

    src, dst = cust_to_cat_edges[0].tolist(), cust_to_cat_edges[1].tolist()
    for e_idx, (c, cat) in enumerate(zip(src, dst)):
        customer_edges[c].append((e_idx, cat))
        category_edges[cat].append((e_idx, c))

    infected = {(start_node_type, start_node_id): 0}
    frontier = [(start_node_type, start_node_id)]

    for hop in range(1, max_hops + 1):
        next_frontier = []

        for node_type, node_id in frontier:
            if node_type == "category":
                neighbors = category_edges[node_id]  # (edge_idx, customer_idx)
                for e_idx, cust_id in neighbors:
                    key = ("customer", cust_id)
                    if key in infected:
                        continue
                    cust_emb = x_dict["customer"][cust_id].unsqueeze(0)
                    cat_emb = x_dict["category"][node_id].unsqueeze(0)
                    attr = edge_attr[e_idx].unsqueeze(0)
                    logit = model.classifier(cust_emb, cat_emb, attr)
                    prob = torch.sigmoid(logit).item()

                    if prob > threshold:
                        infected[key] = hop
                        next_frontier.append(key)

            elif node_type == "customer":
                neighbors = customer_edges[node_id]  # (edge_idx, category_idx)
                for e_idx, cat_id in neighbors:
                    key = ("category", cat_id)
                    if key in infected:
                        continue
                    cust_emb = x_dict["customer"][node_id].unsqueeze(0)
                    cat_emb = x_dict["category"][cat_id].unsqueeze(0)
                    attr = edge_attr[e_idx].unsqueeze(0)
                    logit = model.classifier(cust_emb, cat_emb, attr)
                    prob = torch.sigmoid(logit).item()

                    if prob > threshold:
                        infected[key] = hop
                        next_frontier.append(key)

        if not next_frontier:
            break
        frontier = next_frontier

    return infected


def summarize_cascade(infected):
    depth = max(infected.values())
    breadth = len(infected) - 1  # exclude the starting node itself
    by_hop = {}
    for (node_type, node_id), hop in infected.items():
        by_hop.setdefault(hop, 0)
        by_hop[hop] += 1

    print(f"Cascade depth (max hops reached): {depth}")
    print(f"Cascade breadth (total nodes affected, excl. start): {breadth}")
    print("Nodes affected per hop:")
    for hop in sorted(by_hop.keys()):
        print(f"  Hop {hop}: {by_hop[hop]} nodes")


if __name__ == "__main__":
    data = torch.load("data/processed/graph.pt", weights_only=False)
    data = normalize_graph(data)

    print("Training GCN for simulation use...")
    model = load_trained_model(data)

    # Start the simulation from category node 0 (whichever category that is)
    print()
    print("--- Simulating cascade starting from category node 0 ---")
    infected = simulate_cascade(model, data, "category", 0, threshold=0.5, max_hops=5)
    summarize_cascade(infected)