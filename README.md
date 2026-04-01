# Recourse-ATS

**Semantic Interpretability and Algorithmic Recourse in Automated Resume Screening**

> A transparent, ranking-focused resume screening system that tells candidates *why* they were rejected — and *what to do about it*.

---

## Motivation

Automated resume screening is stuck between two bad options: keyword-matching systems that miss semantic context, and deep-learning models that are accurate but completely opaque. On top of that, standard Transformers cap at 512 tokens, silently truncating the technical requirements buried later in long job descriptions.

Recourse-ATS addresses both problems at once. It combines a Graph Attention Network (GNN) for skill-level reasoning with a Mixture of Experts (MoE) fusion model for interpretable scoring — and layers Integrated Gradients and SHAP on top so every decision can be audited and explained to a candidate.

---

## Architecture Overview

The pipeline runs in four phases:

```
Resume Text ──┐                              ┌── Integrated Gradients (Captum)
              ├─► Phase 1: GNN Skill Graph ──┤
JD Text    ──┘   (GAT, link prediction)      └── Phase 2: MoE Fusion & Scoring
                                                          │
                                               Phase 3: XAI (SHAP + IG)
                                                          │
                                               Phase 4: Recourse + Bias Audit [planned]
```

### Phase 1 — Skill Representation (GNN)
- Extracts skills from raw text using **FlashText** (O(N)) against a 2,500-skill vocabulary
- Builds a **co-occurrence graph** from 1.3M LinkedIn job records (135,104 pruned edges, weight ≥ 5)
- Initialises skill nodes with **MiniLM [CLS] embeddings** (384-d) to prevent over-smoothing
- Trains a **2-layer Graph Attention Network** (4 heads, 128-d output) via **link prediction** (BCE loss, Adam, 100 epochs)

### Phase 2 — Multi-Modal Fusion & Scoring (MoE)
- **Text stream**: MiniLM Transformer → 384-d [CLS] embedding → Text Expert MLP (384→64→1)
- **Graph stream**: averaged GAT skill embeddings (128-d) → Graph Expert MLP (128→64→1)
- **Gating network**: Softmax over both experts → explicit weights (`g_text`, `g_graph`) directly tell you whether a match was driven by written experience or hard skill evidence
- Final score: `ŷ = g_text · E_text(x) + g_graph · E_graph(x)`

### Phase 3 — Explainable AI
- **Integrated Gradients** (via `captum`): integrates prediction change along the path from a blank baseline to the actual text; top-15 tokens visualised as waterfall plots (blue = positive impact, red = negative)
- **SHAP** with a custom `DynamicMultiModalWrapper`: re-extracts graph IDs from each SHAP-masked text sample so multi-modal attribution is correct; produces colour-coded text highlight plots
- **Faithfulness testing** for both methods: masks top-k tokens and measures score drop ΔF to confirm explanations reflect genuine model behaviour

### Phase 4 — Algorithmic Recourse & Bias Audit *(planned)*
- **Counterfactual recourse**: given a "No Fit" decision + XAI explanation, generate concrete skill-gap suggestions (e.g., "adding certification X moves you from No Fit to Potential Fit")
- **Demographic bias audit**: test for scoring disparities across inferred gender, institution name, and geography using standard fairness metrics (disparate impact)

---

## Results

Models evaluated with **ranking metrics** (not regression error) because an ATS needs to surface the best candidate, not predict an exact score.

| Model | MAE ↓ | Spearman ↑ | nDCG@10 ↑ | RBO ↑ |
|---|---|---|---|---|
| TF-IDF (Baseline) | 0.3802 | 0.0908 | 0.6328 | 0.3927 |
| Pure Semantic | 0.3767 | 0.0899 | **0.6453** | 0.3946 |
| Late Fusion | 0.3793 | 0.0528 | 0.5607 | 0.4034 |
| Cross-Attention | **0.3619** | **0.1961** | 0.6004 | 0.4070 |
| **Mixture of Experts** | 0.3737 | 0.1070 | 0.6032 | **0.4162** |

**Why MoE?** It achieves the best RBO — the metric that penalises errors at the very top of the ranked list most heavily. Surfacing the single best candidate reliably matters more than marginal nDCG gains, and the gating weights make every decision auditable.

---

## Datasets

| Dataset | Source | Records | Use |
|---|---|---|---|
| resume-job-description-fit | HuggingFace (`cnamuangtoun`) | 8,000 (train/test) | Primary labelled training set |
| Resume-Screening-Dataset | HuggingFace (`AzharAli05`) | 10,174 | Human-written decision reasons for recourse validation |
| LinkedIn Skills | Kaggle | 1.3M | Skill co-occurrence graph construction |

Label distribution: ~50% No Fit, ~25% Potential Fit, ~25% Good Fit.

---

## Repo Structure

```
Recourse-ATS/
├── data-preparation-inlp-project.ipynb   # EDA, word count stats, truncation hypothesis
├── gnn-pre-training.ipynb                # GAT training, link prediction, sanity checks
├── experimenting-with-fusion-architectures.ipynb  # All four fusion models + ranking eval
├── ig-implementation.ipynb               # Integrated Gradients + faithfulness testing
├── shap-implementation.ipynb             # SHAP + DynamicMultiModalWrapper + decay curves
├── skill_embeddings.npy                  # Trained 128-d GAT embeddings (2,500 skills)
└── README.md
```

---

## Setup

```bash
git clone https://github.com/16srivarshitha/Recourse-ATS.git
cd Recourse-ATS
pip install torch torch-geometric transformers captum shap flashtext scikit-learn pandas matplotlib seaborn
```

Notebooks are self-contained and run in order. A GPU is recommended for the GAT training and fusion model experiments.

---

## Key Design Decisions

**Why MoE over Cross-Attention?**  
Cross-Attention gets better Spearman and MAE, but its internal attention mechanism blends text and graph signals in a way that cannot be cleanly separated. The MoE keeps both streams isolated through the full computation — the gating weights are a first-class output, not a byproduct.

**Why Integrated Gradients + SHAP together?**  
IG is path-based (attribution along a gradient path from baseline to input); SHAP is coalition-based (Shapley values via cooperative game theory). Agreement between the two methods is stronger evidence of faithfulness than either alone.

**Why the `DynamicMultiModalWrapper`?**  
Standard SHAP masks words in text but leaves the graph IDs untouched — so masking "Python" in text doesn't remove Python from the graph stream, causing systematic attribution underestimation. The wrapper intercepts each masked sample and re-extracts graph IDs from the masked text before passing to the model.

---

## Known Limitations

- **Truncation partially solved**: the GNN handles skill-level truncation, but the Transformer backbone still reads only the first 512 tokens of concatenated resume + JD text
- **Skill vocabulary**: 2,500-skill FlashText vocabulary may miss domain-specific or non-English jargon
- **ΔF = 0.0 for Pure Semantic and MoE** (Table 6 in report): suspected cause is [MASK] tokens being re-encoded close to the original embedding; fix planned (full token deletion, k=15)
- **SHAP computational cost**: `DynamicMultiModalWrapper` is slow at scale; KernelSHAP with smaller background datasets needed for production use
- **Recourse unvalidated**: Phase 4 recommendations are planned but not yet shown to genuinely improve candidate outcomes

---

## References

- Bevara et al. (2025). *Resume2Vec*. Electronics, 14:794.
- Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks*. ICML.
- Veličković et al. (2018). *Graph Attention Networks*. ICLR.
- Mothilal et al. (2020). *Explaining ML Classifiers through Diverse Counterfactual Explanations*. FAccT.
- Wachter et al. (2017). *Counterfactual Explanations without Opening the Black Box*. Harvard JOLT.
- Barocas & Selbst (2016). *Big Data's Disparate Impact*. California Law Review.

---

## Authors
Srivarshitha Medarametla . Vaishnavi Manda

IIIT Hyderabad — Introduction to NLP, 2026
