"""
GCN baseline model for cascade-gnn.

Since plain GCNConv doesn't support bipartite/heterogeneous graphs well,
this uses GraphConv (simple neighbor aggregation, no attention) wrapped
in HeteroConv as the closest honest analog to "GCN" for our graph type.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv, Linear
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class GCNEncoder(torch.nn.Module):
    """
    Two-layer heterogeneous GNN encoder. Produces learned embeddings
    for both 'customer' and 'category' nodes by aggregating neighbor
    information across both edge directions.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()

        self.conv1 = HeteroConv({
            ("customer", "orders_from", "category"): GraphConv(-1, hidden_dim,aggr="mean"),
            ("category", "rev_orders_from", "customer"): GraphConv(-1, hidden_dim,aggr="mean"),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("customer", "orders_from", "category"): GraphConv(hidden_dim, hidden_dim,aggr="mean"),
            ("category", "rev_orders_from", "customer"): GraphConv(hidden_dim, hidden_dim,aggr="mean"),
        }, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        return x_dict


class EdgeClassifier(torch.nn.Module):
    """
    Takes a customer embedding + category embedding + raw edge features,
    and predicts late-delivery-risk (binary).
    """
    def __init__(self, hidden_dim=32, edge_feat_dim=6):
        super().__init__()
        input_dim = hidden_dim * 2 + edge_feat_dim
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, 1)

    def forward(self, cust_emb, cat_emb, edge_attr):
        x = torch.cat([cust_emb, cat_emb, edge_attr], dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.squeeze(-1)


class GCNModel(torch.nn.Module):
    def __init__(self, hidden_dim=32, edge_feat_dim=6):
        super().__init__()
        self.encoder = GCNEncoder(hidden_dim)
        self.classifier = EdgeClassifier(hidden_dim, edge_feat_dim)

    def forward(self, data):
        x_dict = self.encoder(data.x_dict, data.edge_index_dict)

        edge_store = data["customer", "orders_from", "category"]
        src, dst = edge_store.edge_index

        cust_emb = x_dict["customer"][src]
        cat_emb = x_dict["category"][dst]

        logits = self.classifier(cust_emb, cat_emb, edge_store.edge_attr)
        return logits


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    logits = model(data)
    edge_store = data["customer", "orders_from", "category"]
    mask = edge_store.train_mask
    y = edge_store.y.float()

    loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask_name="val_mask"):
    model.eval()
    logits = model(data)
    edge_store = data["customer", "orders_from", "category"]
    mask = getattr(edge_store, mask_name)
    y = edge_store.y.float()

    probs = torch.sigmoid(logits[mask])
    preds = (probs > 0.5).long()

    y_true = y[mask].numpy()
    y_pred = preds.numpy()
    y_prob = probs.numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    data = torch.load("data/processed/graph.pt", weights_only=False)

    # --- Normalize features (critical for stable GNN training) ---
    # Node features: z-score normalize each column
    for node_type in data.node_types:
        x = data[node_type].x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-6
        data[node_type].x = (x - mean) / std

    # Edge features: normalize using TRAINING edges only, to avoid leakage
    train_mask = data["customer", "orders_from", "category"].train_mask
    edge_attr = data["customer", "orders_from", "category"].edge_attr
    train_mean = edge_attr[train_mask].mean(dim=0, keepdim=True)
    train_std = edge_attr[train_mask].std(dim=0, keepdim=True) + 1e-6

    normalized_edge_attr = (edge_attr - train_mean) / train_std
    data["customer", "orders_from", "category"].edge_attr = normalized_edge_attr
    # Also normalize the reverse edge type's attr with the SAME stats,
    # since it's the same underlying data just flipped
    data["category", "rev_orders_from", "customer"].edge_attr = (
        data["category", "rev_orders_from", "customer"].edge_attr - train_mean
    ) / train_std

    edge_feat_dim = data["customer", "orders_from", "category"].edge_attr.shape[1]
    model = GCNModel(hidden_dim=32, edge_feat_dim=edge_feat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print("Starting training on CPU...")
    for epoch in range(1, 51):
        loss = train_epoch(model, data, optimizer)

        if epoch % 10 == 0:
            val_metrics = evaluate(model, data, "val_mask")
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val AUC: {val_metrics['roc_auc']:.4f}"
            )

    print()
    test_metrics = evaluate(model, data, "test_mask")
    print("--- Final Test Set Performance (GCN) ---")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"ROC AUC:  {test_metrics['roc_auc']:.4f}")