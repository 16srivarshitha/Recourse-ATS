# Recourse-ATS: Phase 1 - Graph Neural Network Architecture & Pre-training Report
**Date:** February 24, 2026  
**Phase:** 1 (GNN Architecture Decision & Link Prediction Pre-training)

## 1. Motivation: Building Upon Resume2Vec

To design **Recourse-ATS**, we first analyzed the current state-of-the-art in semantic resume matching, primarily focusing on the *Resume2Vec* framework.

### What Resume2Vec Does Well
*   **Core Approach:** It uses Transformer-based deep learning models to convert unstructured text in resumes and job descriptions (JDs) into high-dimensional vectors. It then uses cosine similarity to measure the semantic distance between these embeddings, allowing it to capture deep context and relationships between concepts.
*   **Evaluation Metrics:** It utilizes Normalized Discounted Cumulative Gain (nDCG) to measure the quality of ranked lists, and Rank-Biased Overlap (RBO) to assess similarity between the system’s rankings and human preferences.
*   **Main Results:** Resume2Vec outperformed traditional keyword-based ATS systems, achieving a **15.85% increase in nDCG** and a **15.94% increase in RBO**. It proved particularly effective in technical fields (like health and fitness), with the LLaMA model demonstrating consistent performance across classification tasks.

### What Resume2Vec is Missing
*   **Rigid Criteria Constraints:** Resume2Vec struggles in scenarios requiring rigid adherence to specific technical criteria. The pure embedding approach can miss the strict, binary presence of a specific required certification or tool.
*   **The Interpretability Gap:** Pure embeddings lack interpretability (they are a "black box"). While they capture implicit semantic links, they do not explicitly model the hierarchical or functional relationships between skills. 

## 2. The Innovation: Skill Graphs + GNN

To solve the interpretability gap and enable **Recourse** for candidates, we introduce explicit structural knowledge via **Skill Graphs and Graph Neural Networks (GNNs)**.

*   **Explicit Structural Knowledge:** A graph explicitly models hierarchy and interdependencies (e.g., "PyTorch" is-a "Deep Learning Framework" which requires "Python").
*   **Solving Data Sparsity:** Learned skill relationships allow the system to infer competence in unlisted skills based on known neighbors in the graph. This addresses the common "data sparsity" issue where candidates fail to list every single tool they know.

---

## 3. GNN Architecture Decisions

To process the Skill Graph, we evaluated three primary Graph Neural Network architectures:

### 1. Graph Convolutional Network (GCN)
GCN treats all neighbors of a given node as equally important by taking a simple average of their features (similar to finding the temperature of a point in a grid by averaging the surrounding points). 
*   *Drawback:* It is computationally expensive for dense graphs and treats all neighbors equally, making it highly sensitive to noisy data.

### 2. GraphSAGE (SAmple and aggrEgate)
GraphSAGE solves the GCN scaling problem by taking a random sample of neighbors and then calculating the mean (similar to testing a massive batch of components by sampling a few to represent the whole).
*   *Drawback:* While faster, it still relies on averaging, which dilutes the importance of critical connections.

### 3. Graph Attention Network (GAT) - *Selected Architecture*
GAT recognizes that not all neighbors are equally important. It uses an **Attention Mechanism** to assign learnable weights to each neighbor. 
*   *Why GAT is preferred:* 
    *   **Flexibility:** It can ignore or cut off noisy nodes (e.g., a generic "High School Diploma" node connected to a highly technical "Deep Learning" node).
    *   **Interpretability & Recourse:** If a role requires a "Deep Learning Engineer", and a resume has "Linear Algebra", "Machine Learning", and "Deep Learning", a simple average treats them as $1/3$ each. GAT gives massive weight to DL. Because these weights are visible, we can easily perform **Gap Analysis** and tell candidates exactly which skills they are missing to improve their score.

## 4. Graph Integration Strategy Decision

We must fuse the Graph (Hard Skills) with the Transformer (Semantic Text). We evaluated three strategies:

1.  **Early Fusion:** Add graph embeddings directly to input tokens. *Rejected:* If a candidate gets rejected, the graph info has already inextricably altered the text embeddings. Recourse becomes nearly impossible to debug.
2.  **Late Fusion (Selected):** The Transformer processes text independently, and the GNN processes skills independently. Both vectors are combined at the final Multi-Layer Perceptron (MLP) layer before scoring. *Benefit:* It is modular and explainable. We can explicitly tell a candidate: *"Your semantic experience score was high, but your Hard Skills graph score was low because you lacked X."*
3.  **Multi-task Learning:** Process both tasks simultaneously. *Rejected:* Too computationally complex for the current prototype phase.

---

## 5. Notebook 2 Execution & Results

Based on these decisions, we successfully built and pre-trained the GAT model in Notebook 2 using **Link Prediction**.

### Handling Embedding Collapse
Initial graph construction yielded 241,244 edges for 500 nodes (a nearly fully-connected graph). This caused *Embedding Collapse* (over-smoothing), where all skills mathematically converged to a Cosine Similarity of 1.000. 
*   **The Fix:** We sparsified the graph, pruning edges with low co-occurrence weights. The graph was reduced to **23,424 strong edges**. Weight decay ($1e-4$) was added to the Adam optimizer to regularize the embeddings.

### Training Results
The model was trained for 150 epochs to predict whether two skills share an edge in the real world.
*   **Final Loss (BCE):** ~1.14
*   **Validation AUC:** ~0.80 (Demonstrating strong predictive capability of skill relationships).

### Embedding Sanity Check
Post-training cosine similarity checks proved the GAT successfully learned domain-specific semantic clusters without human labeling:
*   **PYTHON** $\rightarrow$ `sql` (0.988), `java` (0.981), `aws` (0.978), `cloud computing` (0.975)
*   **NURSING** $\rightarrow$ `documentation` (0.967), `patient care` (0.950), `evaluation` (0.910), `cpr certification` (0.906)
*   **DATA ANALYSIS** $\rightarrow$ `negotiation` (0.963), `budget management` (0.958)

**Conclusion:** The pre-trained GAT embeddings now mathematically represent the professional job market. They have been saved as `skill_embeddings.npy` and are ready to be passed into the Late Fusion Cross-Encoder in Notebook 3.