"""
GAT (Graph Attention Network) model for cascade-gnn.

Unlike GCN/GraphSAGE, which treat all neighbors equally (mean/sum),
GAT learns attention weights that let the model decide which neighbors
matter more for a given prediction — also enables interpretability
(visualizing which relationships the model considers most important).
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.models.utils import normalize_graph, get_node_embeddings


class GATEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=32, heads=4):
        super().__init__()
        per_head_dim = hidden_dim // heads

        self.conv1 = HeteroConv({
            ("customer", "orders_from", "category"): GATConv(
                -1, per_head_dim, heads=heads, add_self_loops=False
            ),
            ("category", "rev_orders_from", "customer"): GATConv(
                -1, per_head_dim, heads=heads, add_self_loops=False
            ),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("customer", "orders_from", "category"): GATConv(
                hidden_dim, per_head_dim, heads=heads, add_self_loops=False
            ),
            ("category", "rev_orders_from", "customer"): GATConv(
                hidden_dim, per_head_dim, heads=heads, add_self_loops=False
            ),
        }, aggr="mean")

        # Explicit self-transform, since GATConv with add_self_loops=False
        # (required for bipartite graphs) drops the node's own features
        # entirely — GCN/GraphSAGE include this automatically, GAT does not.
        self.self_lin1 = torch.nn.ModuleDict({
            "customer": Linear(-1, hidden_dim),
            "category": Linear(-1, hidden_dim),
        })
        self.self_lin2 = torch.nn.ModuleDict({
            "customer": Linear(hidden_dim, hidden_dim),
            "category": Linear(hidden_dim, hidden_dim),
        })

    def forward(self, x_dict, edge_index_dict):
        x_input = x_dict

        conv_out = self.conv1(x_dict, edge_index_dict)
        x_dict = {
            k: F.elu(v + self.self_lin1[k](x_input[k]))
            for k, v in conv_out.items()
        }

        x_input2 = x_dict
        conv_out = self.conv2(x_dict, edge_index_dict)
        x_dict = {
            k: F.elu(v + self.self_lin2[k](x_input2[k]))
            for k, v in conv_out.items()
        }

        return x_dict


class EdgeClassifier(torch.nn.Module):
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


class GATModel(torch.nn.Module):
    def __init__(self, hidden_dim=32, edge_feat_dim=6, heads=4):
        super().__init__()
        self.encoder = GATEncoder(hidden_dim, heads)
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


'''def normalize_graph(data):
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

    return data'''


if __name__ == "__main__":
    torch.manual_seed(42)

    data = torch.load("data/processed/graph.pt", weights_only=False)
    data = normalize_graph(data)

    edge_feat_dim = data["customer", "orders_from", "category"].edge_attr.shape[1]
    model = GATModel(hidden_dim=32, edge_feat_dim=edge_feat_dim, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    print("Starting training on CPU...")
    for epoch in range(1, 151):
        loss = train_epoch(model, data, optimizer)

        if epoch % 20 == 0:
            val_metrics = evaluate(model, data, "val_mask")
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val AUC: {val_metrics['roc_auc']:.4f}"
            )

    print()
    test_metrics = evaluate(model, data, "test_mask")
    print("--- Final Test Set Performance (GAT) ---")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"ROC AUC:  {test_metrics['roc_auc']:.4f}")