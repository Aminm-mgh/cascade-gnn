# cascade-gnn

**Predicting supply chain disruption risk with Graph Neural Networks — and demonstrating that graph structure genuinely beats traditional ML.**

A live risk-scoring API, an interactive dashboard, and three trained GNN architectures (GCN, GraphSAGE, GAT) benchmarked against classical baselines on real supply chain data — built to show that modelling supply chains as *networks*, not spreadsheets, materially improves disruption prediction.

---

## What this does

Given a customer and a product category, `cascade-gnn` predicts the probability that an order will be delivered late — using a model that has learned from the *structure* of customer-category relationships, not just individual order features.

- **REST API** — `POST /risk-score`, get a live prediction in milliseconds
- **Interactive dashboard** — look up any customer/category risk, and visualize a real risk network
- **Three trained GNN architectures** — GCN, GraphSAGE, GAT, fully benchmarked
- **Proven lift over traditional ML** — graph-based models beat logistic regression and XGBoost by a wide margin (see results below)

## Quick demo

```bash
curl -X POST  \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 100, "category_id": 5}'

# {"customer_id":100,"category_id":5,"risk_score":0.539,"risk_level":"medium"}
```

Or launch the dashboard for a visual, clickable version:

```bash
streamlit run src/dashboard/app.py
```

---

## Results: does the graph actually help?

This was the central question. The answer is yes — clearly.

| Model | ROC AUC | Notes |
|---|---|---|
| Betweenness centrality heuristic | 0.50 | Chance level — structure alone, no learning |
| Logistic Regression | 0.86 | No graph awareness at all |
| XGBoost + centrality features | 0.88 | Graph structure as hand-crafted features |
| **GCN** | **0.943** | Learned graph representations |
| **GraphSAGE** | **0.945** | Best performer |
| **GAT** | 0.925 | Attention-based; see findings below |

**Headline finding:** letting a GNN learn its own representations from graph structure outperforms even a strong gradient-boosted model armed with manually engineered network features (centrality). This is a genuine, evaluated answer to a real research question, not an assumption.

### Notable nuances (the interesting part)

- **Betweenness centrality has zero standalone predictive power** on this graph — it's chance level. Late-delivery risk here isn't driven by network position; it's driven by operational factors (shipping mode is the single dominant feature across every model tested).
- **GAT underperforms GCN/GraphSAGE**, and we know why: its learned attention weights show only mild differentiation between neighbors (3.18x the "uniform attention" baseline) — this graph's structure doesn't have strong "a few key neighbors matter most" asymmetry for attention to exploit. A legitimate negative result, not a failure.
- **The cascade simulation reveals a real limitation:** propagating disruption through the graph using the trained GNN reaches only 198 nodes at hop 1, then stops. This isn't a bug — the model was trained to predict per-order lateness risk, not neighbor-to-neighbor contagion, so it doesn't behave like an epidemic model. Documented honestly as a scope boundary for future work, not hidden.

---

## Architecture

The DataCo Smart Supply Chain dataset (180,519 orders) is reframed as a **bipartite graph**:

- **Category nodes** (50) — product categories, standing in for the "supplier" side
- **Customer nodes** (20,652) — individual customers
- **Edges** — one per order, carrying shipping mode, scheduled delivery time, discount rate, and sales/quantity as features
- **Target label** — `Late_delivery_risk` (54.8% positive rate — a real, non-trivial classification task)

Models are trained on a **time-based 80/10/10 split** (not random) — earliest orders for training, most recent for testing — to mimic realistic deployment rather than inflate performance with data leakage between similar time periods.

```
Customer ──orders_from──▶ Category
        ◀─rev_orders_from──
```

Each GNN encoder produces learned embeddings for both node types via two rounds of message passing; an MLP classifier then combines a customer's embedding + a category's embedding + the specific order's features to predict risk.

---

## Project structure

```
cascade-gnn/
├── data/
│   ├── raw/              # DataCo CSV (not committed — see Setup)
│   └── processed/        # Built graph object + trained model weights
├── src/
│   ├── graph/            # Graph construction from raw data
│   ├── models/           # GCN, GraphSAGE, GAT + shared utilities
│   ├── simulation/       # Cascade/disruption propagation simulation
│   ├── explain/          # Attention weight analysis
│   ├── api/              # FastAPI risk-scoring service
│   └── dashboard/        # Streamlit interactive dashboard
├── environment.yml       # Conda environment (recommended)
└── requirements.txt      # Pip fallback
```

## Setup

**1. Environment** (conda recommended — this project's numerical stack needs a consistent BLAS/OpenMP backend):

```bash
conda env create -f environment.yml
conda activate cascade-gnn
```

**2. Data** — download the [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) from Kaggle, and place `DataCoSupplyChainDataset.csv` in `data/raw/`.

**3. Build the graph and train a model:**

```bash
python3 src/graph/builder.py     # builds data/processed/graph.pt
python3 src/models/gcn.py        # trains GCN, saves weights, prints results
```

**4. Run the API:**

```bash
python3 -m uvicorn src.api.main:app --reload
# Visit http://127.0.0.1:8000/docs for interactive API testing
```

**5. Run the dashboard:**

```bash
streamlit run src/dashboard/app.py
```

---

## What's next

- SIR/SIS epidemic-model baseline for the cascade simulation (currently GNN-only)
- GNNExplainer for edge/feature-level interpretability (currently attention-weights only)
- Temporal extension (TGN) to capture how risk evolves over time, not just structure
- Model checkpointing for GraphSAGE/GAT (currently only GCN weights are saved)

## License

MIT
##
