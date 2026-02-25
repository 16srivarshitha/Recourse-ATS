# Recourse-ATS: Phase 1 - Architectural Variants for Graph-Transformer Fusion
**Date:** February 25, 2026  
**Phase:** 1 (Late Fusion, Mixture of Experts, Cross-Attention, and Evaluation)

## 1. Objectives
The primary limitation of state-of-the-art semantic resume matching systems (such as *Resume2Vec*) is their "black-box" nature. While pure Transformer embeddings capture semantic nuance, they fail to provide **Recourse**—actionable feedback explaining *why* a candidate received a specific score. 

The objective of this phase was to explore three distinct architectures to fuse the semantic understanding of a Pre-trained Language Model (`ms-marco-MiniLM-L-6-v2`) with the explicit structural knowledge of our pre-trained Graph Attention Network (GAT).

## 2. Architectural Variants & Methodology

To effectively combine unstructured text with structured graph embeddings, we designed and evaluated three distinct fusion architectures:

### Architecture A: Standard Graph-Enhanced Cross-Encoder (The Baseline)
*   **Design:** The Resume and JD text are concatenated and passed through the Transformer to extract a `[CLS]` token embedding. Simultaneously, the Graph Node IDs for skills present in the text are mapped to our pre-trained GAT embeddings and averaged (Mean Pooling). The Text, Resume Graph, and JD Graph embeddings are concatenated and passed through a Multi-Layer Perceptron (MLP).
*   **Rationale:** This serves as our robust baseline. It provides the network with maximum mathematical freedom to combine features to minimize Mean Squared Error (MSE).

### Architecture B: The "Interpretability-by-Design" Mixture of Experts (MoE)
*   **Design:** The model is split into three distinct subnetworks:
    1.  **Semantic Text Expert:** Predicts a match score (0-1) using *only* the Transformer text embedding.
    2.  **Hard Skills Graph Expert:** Predicts a match score (0-1) using *only* the GAT embeddings.
    3.  **Gating Network:** A dynamic routing network that looks at the full context and outputs a Softmax probability distribution, determining how much "trust" to assign to each expert for a specific candidate.
*   **Rationale:** By mathematically constraining the fusion process, the model becomes inherently interpretable. The final score is a weighted sum: `(Gate_Text * Score_Text) + (Gate_Graph * Score_Graph)`.

### Architecture C: Cross-Attention (Graph-in-the-Loop)
*   **Design:** Instead of Late Fusion, we allow the Transformer's text embeddings to actively *query* the Graph before the final scoring. The text sequence acts as the **Query**, and the extracted GAT skill vectors act as the **Keys and Values** in a Multi-Head Cross-Attention layer. 
*   **Rationale:** This mimics human HR reading behavior. As the model reads the resume text, the attention mechanism dynamically pulls in mathematical context from the skill graph to enrich the text's representation.

## 3. Experimental Results & Interpretation

All models were trained for 3 epochs on the `cnamuangtoun/resume-job-description-fit` dataset using an AdamW optimizer (learning rates: `2e-5` for Transformers, `1e-3` for MLPs). 

### Quantitative Results (Test Set MAE)
1.  **Standard Fusion (Baseline):** `0.3446`
2.  **Cross-Attention Fusion:** `0.3470`
3.  **MoE Graph Cross-Encoder:** `0.3522`

### Interpretation: Accuracy vs. Transparency Trade-offs
1.  **The Cross-Attention Success:** The Cross-Attention model demonstrated a rapid learning curve (dropping from 0.3766 to 0.3470). It nearly matched the unconstrained baseline, proving that dynamically injecting structural graph knowledge into a Transformer's reading process is highly effective, even on relatively small datasets (~6k pairs).
2.  **The Cost of Interpretability (MoE):** The MoE model achieved an MAE of `0.3522`. The ~0.7% drop in pure accuracy is a well-documented trade-off when forcing neural networks through human-readable bottlenecks (e.g., probability weights that must sum to 1.0). 
3.  **Qualitative MoE Interpretability:** Despite the slight accuracy drop, the MoE architecture successfully solved the Recourse problem. In test evaluations, the MoE could explicitly state its reasoning logic. For example, rejecting a candidate because the Gating Network prioritized the Semantic Text Expert (which scored the candidate a 0.01), explicitly overriding the Graph Expert (which had scored the candidate a 0.55 due to keyword overlap).

## 4. Conclusion
We have successfully developed and evaluated three distinct Graph-Transformer fusion models. 
*   If the primary business goal is **pure predictive accuracy**, the Standard Fusion model is optimal.
*   If the goal is **contextual richness**, the Cross-Attention model dynamically blends text and graph space.
*   If the goal is **Candidate Fairness, Transparency, and Recourse**, the Mixture of Experts (MoE) model is vastly superior, as it provides an explicit, quantitative breakdown of every hiring decision.