"""
Baseline models for cascade-gnn — non-graph-aware comparisons.

These prove whether the GNN's graph structure actually adds predictive
power over models that ignore the graph entirely.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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

    # Scale features — critical for logistic regression to converge properly
    # when features are on very different scales (e.g. Sales in thousands
    # vs discount rate in 0-1 vs categorical codes in 0-3)
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

if __name__ == "__main__":
    data = load_graph()
    X, y = build_edge_features(data)
    print(f"Feature matrix shape: {X.shape}")
    run_logistic_regression(X, y)