"""
Streamlit dashboard for cascade-gnn.

Two parts:
1. A sampled, color-coded subgraph visualization (PyVis) showing
   customer-category risk relationships.
2. A lookup tool to check the predicted risk score for any real
   customer/category pair using the trained GCN.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import torch
from pyvis.network import Network
import streamlit.components.v1 as components

from src.models.gcn import GCNModel
from src.models.utils import normalize_graph, get_node_embeddings

st.set_page_config(page_title="cascade-gnn Dashboard", layout="wide")


@st.cache_resource
def load_model_and_data():
    data = torch.load("data/processed/graph.pt", weights_only=False)
    data = normalize_graph(data)

    edge_feat_dim = data["customer", "orders_from", "category"].edge_attr.shape[1]
    model = GCNModel(hidden_dim=32, edge_feat_dim=edge_feat_dim)

    with torch.no_grad():
        model(data)
    model.load_state_dict(torch.load("data/processed/gcn_weights.pt", weights_only=True))
    model.eval()

    x_dict = get_node_embeddings(model, data)
    return data, model, x_dict


data, model, x_dict = load_model_and_data()

num_customers = data["customer"].x.shape[0]
num_categories = data["category"].x.shape[0]

st.title("cascade-gnn: Supply Chain Disruption Risk Dashboard")
st.caption("Showing top 10 customers by order count, connected to their ordered categories, colored by predicted risk")

# --- Section 1: Risk Score Lookup ---
st.header("Risk Score Lookup")
col1, col2 = st.columns(2)
with col1:
    customer_id = st.number_input("Customer ID", min_value=0, max_value=num_customers - 1, value=0)
with col2:
    category_id = st.number_input("Category ID", min_value=0, max_value=num_categories - 1, value=0)

if st.button("Predict Risk"):
    cust_emb = x_dict["customer"][customer_id].unsqueeze(0)
    cat_emb = x_dict["category"][category_id].unsqueeze(0)
    avg_edge_attr = data["customer", "orders_from", "category"].edge_attr.mean(dim=0, keepdim=True)

    with torch.no_grad():
        logit = model.classifier(cust_emb, cat_emb, avg_edge_attr)
        prob = torch.sigmoid(logit).item()

    st.metric("Predicted Late-Delivery Risk", f"{prob:.1%}")
    if prob >= 0.6:
        st.error("High risk")
    elif prob >= 0.4:
        st.warning("Medium risk")
    else:
        st.success("Low risk")

# --- Section 2: Subgraph Visualization ---
st.header("Sample Network Visualization")
st.caption("Showing top 10 customers by order count, connected to their ordered categories, colored by predicted risk")
edge_store = data["customer", "orders_from", "category"]
src, dst = edge_store.edge_index[0].tolist(), edge_store.edge_index[1].tolist()

# Find top 30 customers by degree (order count)
from collections import Counter
customer_degree = Counter(src)
top_customers = [c for c, _ in customer_degree.most_common(10)]
top_customers_set = set(top_customers)

net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
net.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "springLength": 100,
      "springConstant": 0.08
    },
    "maxVelocity": 50,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": {"iterations": 150}
  }
}
""")

# Only add categories that are actually connected to our sampled customers,
# to avoid disconnected nodes drifting off unpredictably
connected_categories = set(
    cat_id for cust_id, cat_id in zip(src, dst) if cust_id in top_customers_set
)
for cat_id in connected_categories:
    net.add_node(f"cat_{cat_id}", label=f"Category {cat_id}", color="#4A90D9", shape="box")
with torch.no_grad():
    avg_edge_attr = edge_store.edge_attr.mean(dim=0, keepdim=True)
    for cust_id in top_customers:
        net.add_node(f"cust_{cust_id}", label=f"Customer {cust_id}", color="#CCCCCC")

    for cust_id, cat_id in zip(src, dst):
        if cust_id in top_customers_set:
            cust_emb = x_dict["customer"][cust_id].unsqueeze(0)
            cat_emb = x_dict["category"][cat_id].unsqueeze(0)
            logit = model.classifier(cust_emb, cat_emb, avg_edge_attr)
            prob = torch.sigmoid(logit).item()

            color = "#D9534F" if prob >= 0.6 else "#F0AD4E" if prob >= 0.4 else "#5CB85C"
            net.add_edge(f"cust_{cust_id}", f"cat_{cat_id}", color=color, title=f"Risk: {prob:.1%}")

net.save_graph("data/processed/network_temp.html")
with open("data/processed/network_temp.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=620)