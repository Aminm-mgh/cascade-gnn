"""
Baseline models for cascade-gnn — non-graph-aware comparisons.

These prove whether the GNN's graph structure actually adds predictive
power over models that ignore the graph entirely.
"""

import torch
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def load_graph(path: str = "data/processed/graph.pt"):
    return torch.load(path, weights_only=False)


def build_edge_features(data):
    """
    Build a flat feature matrix for each edge (order) using ONLY the
    connected customer's and category's node features + edge_attr.
    No graph structure (no propagation, no neighbor info) is used —
    this is the point of a baseline.
    """
    edge_index = data["customer", "orders_from", "category"].edge_index
    edge_attr = data["customer", "orders_from", "category"].edge_attr
    y = data["customer", "orders_from", "category"].y

    cust_x = data["customer"].x
    cat_x = data["category"].x

    src, dst = edge_index[0], edge_index[1]
    customer_feats = cust_x[src]
    category_feats = cat_x[dst]

    X = torch.cat([customer_feats, category_feats, edge_attr], dim=1)
    return X.numpy(), y.numpy()


def run_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)[:, 1]

    print("--- Logistic Regression Baseline ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, probs):.4f}")


def compute_centrality_features(data):
    """
    Build a NetworkX graph from the customer-category edges to compute
    degree and betweenness centrality for each node, then map those
    values back onto each edge (order) as extra features.
    """
    edge_index = data["customer", "orders_from", "category"].edge_index
    num_customers = data["customer"].x.shape[0]
    num_categories = data["category"].x.shape[0]

    G = nx.Graph()
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy() + num_customers
    G.add_edges_from(zip(src, dst))

    print(f"Computing degree centrality for {G.number_of_nodes()} nodes...")
    degree_cent = nx.degree_centrality(G)

    print("Computing approximate betweenness centrality (sampled)...")
    betweenness_cent = nx.betweenness_centrality(G, k=500, seed=42)

    cust_degree = np.array([degree_cent.get(i, 0.0) for i in range(num_customers)])
    cust_betweenness = np.array([betweenness_cent.get(i, 0.0) for i in range(num_customers)])

    cat_degree = np.array([degree_cent.get(i + num_customers, 0.0) for i in range(num_categories)])
    cat_betweenness = np.array([betweenness_cent.get(i + num_customers, 0.0) for i in range(num_categories)])

    return cust_degree, cust_betweenness, cat_degree, cat_betweenness


def build_edge_features_with_centrality(data):
    X, y = build_edge_features(data)

    cust_degree, cust_betweenness, cat_degree, cat_betweenness = compute_centrality_features(data)

    edge_index = data["customer", "orders_from", "category"].edge_index
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    centrality_feats = np.column_stack([
        cust_degree[src],
        cust_betweenness[src],
        cat_degree[dst],
        cat_betweenness[dst],
    ])

    X_with_centrality = np.column_stack([X, centrality_feats])
    return X_with_centrality, y


def run_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print("--- XGBoost Baseline (with centrality) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, probs):.4f}")

    importances = clf.feature_importances_
    print()
    print("Top 5 most important features (by index):")
    top5 = np.argsort(importances)[::-1][:5]
    for idx in top5:
        print(f"  Feature {idx}: importance {importances[idx]:.4f}")



def run_centrality_heuristic(data):
    """
    Simple heuristic baseline: flag orders as high-risk if the connected
    customer's betweenness centrality is above the median. No learning
    involved — just a fixed threshold rule.
    """
    cust_degree, cust_betweenness, cat_degree, cat_betweenness = compute_centrality_features(data)

    edge_index = data["customer", "orders_from", "category"].edge_index
    src = edge_index[0].numpy()
    y = data["customer", "orders_from", "category"].y.numpy()

    edge_betweenness = cust_betweenness[src]

    threshold = np.median(edge_betweenness)
    preds = (edge_betweenness > threshold).astype(int)

    print("--- Simple Betweenness-Centrality Heuristic ---")
    print(f"Threshold (median betweenness): {threshold:.6f}")
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(f"F1 Score: {f1_score(y, preds):.4f}")
    # No probability scores exist for a hard threshold rule, so no ROC AUC here




if __name__ == "__main__":
    data = load_graph()

    X, y = build_edge_features(data)
    print(f"Feature matrix shape: {X.shape}")
    run_logistic_regression(X, y)

    print()
    X_centrality, y_centrality = build_edge_features_with_centrality(data)
    print(f"Feature matrix with centrality shape: {X_centrality.shape}")
    run_xgboost(X_centrality, y_centrality)
    print()
    run_centrality_heuristic(data)