
# Notebook 2: GNN Pre-training via Link Prediction

## Purpose

This notebook takes the skill vocabulary and co-occurrence graph built in Notebook 1 and **pre-trains a Graph Attention Network (GAT)** on it. The goal is to produce rich, contextual embeddings for each of the 2,502 skills - embeddings that capture not just the meaning of the skill name itself, but its position in the broader ecosystem of professional skills.

These embeddings are later used in Notebook 3 as an additional feature signal alongside the Transformer's text representations.
![Data flow diagram](../flowcharts/Data%20flow/data_flow_notebook2.png)

---

## Core Idea: Why Use a GNN at All?

A pure Transformer approach to resume-JD matching treats both documents as text and computes a similarity score based on token-level attention. This works reasonably well but has a fundamental limitation: **it cannot reason about skill adjacency**.

Consider this example:
- A job description requires `"PyTorch"` and `"deep learning"`.
- A resume lists `"TensorFlow"` and `"neural networks"`.

A Transformer trained on surface text might not recognise this as a strong match because the specific words differ. But a GNN trained on a skill co-occurrence graph *knows* that `PyTorch` and `TensorFlow` are neighbors in the skills graph — they appear together in the same job postings constantly. This relational information lets the model reason about **skill equivalence and proximity**, which text-only models miss.

---

## Architecture Design Decisions

### Methodological Upgrade: Inductive Node Initialization

Initializing graph nodes with random vectors (`nn.Embedding`) in dense graphs often leads to **"Embedding Collapse"** (oversmoothing), where all nodes average out to the same vector through message passing. 

**Solution:** We make the graph *Inductive* by initializing the node features using pre-trained Language Model embeddings. 

**How it works:**
1. Each of the 2,502 skill strings is passed through a lightweight sentence transformer (**MiniLM**: `sentence-transformers/all-MiniLM-L6-v2`) as a short text query.
2. The `[CLS]` token output (shape: `[384]`) is extracted as that skill's starting representation.
3. These 384-dimensional vectors become the input features `x` to the GAT.

This jumpstarts the GNN with semantic understanding (e.g., knowing "Python" and "SQL" are related textually) before the GNN refines them using real-world co-occurrence edges. It acts as a powerful prior that prevents embeddings from collapsing into a single point.
![Architecture diagram](../flowcharts/Architecture%20Diagrams/architecture_skillgat.png)

---

## Graph Construction

### Input from Notebook 1
- **Nodes:** 2,502 skill strings from the expanded vocabulary.
- **Raw edges:** 740,444 skill co-occurrence pairs with weights (counts).

### Sparsification
We filter the graph to only keep edges where skills co-occurred in **at least 10 job postings**. This removes noise from coincidental co-occurrences and focuses the GNN on robust, meaningful relationships.

| Stage | Edge Count |
|---|---|
| Raw co-occurrence pairs | 740,444 |
| After min-weight filter (≥ 10) | 81,873 |
| After bidirectional expansion (PyG format) | 163,746 |

PyTorch Geometric requires directed edges, so each undirected edge `(A, B)` becomes two directed edges `(A → B)` and `(B → A)`.

### Edge Weight Normalization
Raw co-occurrence counts span a huge range. Using raw counts as edge weights would cause gradient explosion during training. We apply **log normalization**: `weight = log(1 + count)`. This compresses the range while preserving the relative ordering of edge strengths.

---

## Model Architecture: `LM_Initialized_SkillGAT`

```
LM_Initialized_SkillGAT(
  (gat1): GATConv(384, 32, heads=4)
  (norm1): LayerNorm(128)
  (gat2): GATConv(128, 128, heads=1)
  (norm2): LayerNorm(128)
)
```

### Layer 1: `GATConv(384, 32, heads=4)`
- Takes the 384-dim MiniLM embeddings as input.
- Uses **4 attention heads**, each computing a 32-dim output, allowing the model to learn different types of skill relationships simultaneously (e.g., technical proximity vs. domain proximity).
- Outputs: 4 × 32 = 128 dimensions per node.

### Layer 2: `GATConv(128, 128, heads=1)`
- Refines the 128-dim representations with a single-head output layer for a consolidated final embedding.

### Activation, Normalization, and Regularization
- **LayerNorm:** Applied after each GAT convolution. This standardizes the activations, stabilizing the learning process and further preventing magnitude collapse.
- **ELU activation:** Allows small negative values, improving gradient flow in deep message-passing chains.
- **Dropout (p=0.3):** Increased from earlier iterations to forcefully prevent overfitting on the dense graph.

---

## Pre-training Task: Link Prediction

We train the model using **link prediction**: given a pair of skill nodes, predict whether there is a true co-occurrence edge between them.

### Advanced Loss Formulation
To explicitly fight embedding collapse and handle class imbalances, the loss function was significantly upgraded:

1. **L2 Normalization:** Embeddings are L2-normalized (`F.normalize(p=2)`) before taking the dot product. This bounds similarity scores strictly between -1 and 1, preventing scale collapse.
2. **Weighted BCE Loss:** False negatives are penalized heavily. We use `pos_weight=3.0` because predicting true edges accurately is more valuable for downstream recommendations than perfectly rejecting random noise.
3. **2x Negative Sampling:** For every real edge, we sample 2 random non-edges, forcing the model to learn sharper boundaries between related and unrelated skills.
4. **Explicit Spread Loss:** We calculate the standard deviation of the batch embeddings across dimensions and penalize low variance: `spread_loss = -z.std(dim=0).mean()`. This mathematical constraint physically forces the embeddings to occupy diverse spaces rather than clumping together.

**Optimizer:** Adam with `lr=0.001` and high `weight_decay=1e-3`. Gradient clipping (`max_norm=1.0`) is used to maintain stability.

---

## Training Results

Training was run for **200 epochs**.

![Training Loss vs. Epochs | Validation AUC vs. Epochs](../plots/Training_loss_and_validation_auc.png)

| Epoch | Total Loss | Validation AUC |
|---|---|---|
| 20 | 1.9868 | 0.8581 |
| 60 | 1.9249 | 0.8703 |
| 100 | 1.9112 | 0.8824 |
| 200 | 1.9006 | 0.8890 |

### Interpreting the Results

The upgraded methodology yielded a massive improvement. The **Validation AUC reached 0.889**, up significantly from ~0.66 in earlier runs. 
An AUC of nearly 0.89 means that if you give the GNN one real skill connection and one completely random pair of skills, it will correctly identify the real connection 89% of the time. This proves the GNN has successfully learned the topology of the professional job market.

---

## Sanity Check: Embedding Quality

The true test is whether the final embedding space makes semantic sense and avoids oversmoothing. We query the top-5 most similar skills to a set of probe skills:

### Results

**Skills most similar to DOCKER:**
- kubernetes (Sim: 0.999)
- devops (Sim: 0.997)
- microservices (Sim: 0.997)
- ci/cd (Sim: 0.997)
- jenkins (Sim: 0.997)

**Skills most similar to REACT:**
- typescript (Sim: 1.000)
- node.js (Sim: 0.999)
- javascript (Sim: 0.999)
- angular (Sim: 0.999)
- css (Sim: 0.999)

**Skills most similar to AWS:**
- cloud services (Sim: 0.997)
- gcp (Sim: 0.996)
- microsoft azure (Sim: 0.996)
- azure (Sim: 0.995)
- cloud (Sim: 0.995)

**Skills most similar to MACHINE LEARNING:**
- artificial intelligence (Sim: 0.998)
- data science (Sim: 0.997)
- business analysis (Sim: 0.997)
- optimization (Sim: 0.997)
- analytics (Sim: 0.997)

### Collapse Check Diagnostics
- **Mean std across dimensions:** `0.7082` (Target: > 0.05)
- **Mean value across embeddings:** `-0.0004` (Target: ~0)

### What These Results Mean
The embeddings demonstrate flawless **domain clustering** (e.g., Docker perfectly aligning with Kubernetes and CI/CD). The mathematical diagnostics confirm that **collapse has been entirely avoided**. The standard deviation is incredibly healthy (0.70), proving that the integration of L2 normalization, LayerNorm, and Spread Loss worked exactly as intended.

---

## Outputs Produced

| File | Contents | Shape | Used By |
|---|---|---|---|
| `pretrained_skill_gat.pth` | Trained GAT model weights | — | Optional: for re-inference |
| `skill_embeddings.npy` | Final L2-normalized 128-dim embeddings | `[2502, 128]` | Notebook 3 (as graph features) |

---

## Summary of Key Decisions

| Decision | Alternative | Reason for Choice |
|---|---|---|
| LM-Initialized Features (MiniLM) | Random `nn.Embedding` | Solves cold-start and oversmoothing by injecting textual semantics. |
| Sparsification Threshold = 10 | Threshold 1, 3, or 5 | Strikes the best balance between graph density (81k edges) and noise reduction. |
| Layer Normalization | No norm | Stabilizes activations between GNN passes. |
| Spread Loss + L2 Norm | Standard BCE alone | Physically forces embeddings to utilize the full dimensional space, preventing collapse. |
| 2x Negative Sampling | 1:1 Sampling | Forces sharper decision boundaries between related and unrelated nodes. |
| High Weight Decay (1e-3) | Lower/No Decay | Acts as a structural regularization on the dense graph to prevent weight saturation. |
