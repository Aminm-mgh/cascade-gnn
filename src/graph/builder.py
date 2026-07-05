import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import numpy as np


def build_time_split_masks(df: pd.DataFrame, train_frac=0.8, val_frac=0.1):
    """
    Sort orders by date and assign train/val/test masks based on time —
    earliest 80% train, next 10% val, most recent 10% test.
    This mimics real deployment: predicting future orders from past patterns.
    """
    dates = pd.to_datetime(df["order date (DateOrders)"])
    order = dates.argsort().values

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[order[:train_end]] = True
    val_mask[order[train_end:val_end]] = True
    test_mask[order[val_end:]] = True

    return (
        torch.tensor(train_mask),
        torch.tensor(val_mask),
        torch.tensor(test_mask),
    )


def load_raw_data(path: str = "data/raw/DataCoSupplyChainDataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    return df


def build_graph(df: pd.DataFrame) -> HeteroData:
    """
    Construct a bipartite Category <-> Customer graph.
    """
    # --- Build node ID mappings ---
    categories = df["Category Name"].unique()
    customers = df["Customer Id"].unique()

    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    cust_to_idx = {cust: i for i, cust in enumerate(customers)}

    # --- Node features ---
    cat_stats = df.groupby("Category Name").agg(
        avg_price=("Product Price", "mean"),
        avg_profit_ratio=("Order Item Profit Ratio", "mean"),
        order_count=("Order Item Id", "count"),
    ).reindex(categories)
    category_x = torch.tensor(cat_stats.values, dtype=torch.float)

    cust_stats = df.groupby("Customer Id").agg(
        total_sales=("Sales", "sum"),
        order_count=("Order Item Id", "count"),
        avg_late_risk=("Late_delivery_risk", "mean"),
    ).reindex(customers)
    customer_x = torch.tensor(cust_stats.values, dtype=torch.float)

    # --- Edges (one per order line) ---
    src = df["Customer Id"].map(cust_to_idx).values
    dst = df["Category Name"].map(cat_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # --- Edge features (order-level, no label leakage) ---
    shipping_mode_codes = df["Shipping Mode"].astype("category").cat.codes.values
    order_region_codes = df["Order Region"].astype("category").cat.codes.values

    edge_numeric = df[[
        "Sales",
        "Order Item Quantity",
        "Order Item Discount Rate",
        "Days for shipment (scheduled)",
    ]].values

    edge_attr = torch.tensor(
        np.column_stack([
            edge_numeric,
            shipping_mode_codes,
            order_region_codes,
        ]),
        dtype=torch.float,
    )

    edge_label = torch.tensor(df["Late_delivery_risk"].values, dtype=torch.long)

    train_mask, val_mask, test_mask = build_time_split_masks(df)

    # --- Assemble HeteroData object ---
    data = HeteroData()
    data["category"].x = category_x
    data["customer"].x = customer_x
    data["customer", "orders_from", "category"].edge_index = edge_index
    data["customer", "orders_from", "category"].edge_attr = edge_attr
    data["customer", "orders_from", "category"].y = edge_label
    data["customer", "orders_from", "category"].train_mask = train_mask
    data["customer", "orders_from", "category"].val_mask = val_mask
    data["customer", "orders_from", "category"].test_mask = test_mask

    data = ToUndirected()(data)

    return data


def save_graph(data: HeteroData, path: str = "data/processed/graph.pt") -> None:
    torch.save(data, path)
    print(f"Graph saved to {path}")


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows")
    graph = build_graph(df)
    print(graph)
    save_graph(graph)