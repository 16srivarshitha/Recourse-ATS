# Experimenting with Fusion Architectures for Resume-JD Matching

## What is this?

This notebook is an architectural comparison study for resume-to-job description (JD) matching. The goal is to figure out which model architecture best answers the question: *"How well does this candidate fit this job?"*

The problem with most ATS (Applicant Tracking Systems) out there is that they do keyword matching — if the resume doesn't use the exact words in the job description, the candidate gets filtered out. Someone who writes "built ML pipelines" instead of "machine learning engineering" might get rejected even if they're the best fit. We want to move past that.

The broader project has two signals available for matching:
- **Semantic signal** — what the text means, captured via a Bi-Encoder transformer.
- **Skill graph signal** — which concrete skills are mentioned and how they relate to each other, captured via pretrained GNN embeddings (2500 skills × 128 dimensions).

This notebook asks: what is the best way to combine those two signals?

---

## Dataset

- **Training set:** 6,166 resume–JD pairs
- **Test set:** 1,754 resume–JD pairs
- Each pair has a human-assigned match score (the target), extracted resume skills, extracted JD skills, and the full resume + JD text
- Skill embeddings are 128-dimensional GNN embeddings pretrained on a skill co-occurrence graph (loaded from `skill_embeddings.npy`)

---

## The Approach: An Architectural Evolution Study

Rather than jumping straight to the most complex model, we run four architectures in increasing order of sophistication. 

### Transformer Backbone
All four deep learning models use `sentence-transformers/all-MiniLM-L6-v2` as the text encoder. We use a **Bi-Encoder** (Twin Tower) setup. 
*Architectural Note:* For the fusion models (Models 2, 3, and 4), the text embedding focuses purely on evaluating the *Candidate's Resume*, while the *Skill Graph* pathway is entirely responsible for mapping the candidate to the Job Description context.

All models are trained with AdamW (lr=1e-4), MSE loss, batch size 16, for 3 epochs.

---

## The Four Architectures

### 1. TF-IDF Baseline
Before any deep learning, we establish what traditional ATS systems actually do. TF-IDF with cosine similarity — a bag of words representation with no semantic understanding whatsoever. This is our "null hypothesis".

### 2. Pure Semantic (Model 1)
Text only, no skill graph at all. Encodes the Resume and JD separately, calculates their Cosine Similarity, and passes the similarity score through an MLP. It captures semantic meaning well but has no structured understanding of specific hard skills. It's also a black box.

![](../flowcharts/Architecture%20Diagrams/architecture_diagram_pure_semantic.png)

### 3. Late Fusion (Model 2)
The first architecture that uses both signals. The text embedding of the resume and the skill graph embeddings are computed separately and concatenated before the final MLP. 

**Key architectural decision — JD-Attended Skill Pooler:** The resume skill representation is not a naive mean. Instead, we use a cross-attention module where the **JD skills act as the Query and the resume skills are Key/Value**. Resume skills that are semantically close to what the JD is asking for get high attention weights, and generic skills get suppressed.

![](../flowcharts/Architecture%20Diagrams/architecture_diagram_late_fusion.png)

### 4. Cross-Attention (Model 3)
The transformer's token sequence actively attends over the graph skill embeddings. Specifically, the Resume text token sequence is the Query, and the projected skill embeddings (Resume + JD skills) are the Keys and Values. The text representation is fundamentally shaped by the skill information before scoring.

![](../flowcharts/Architecture%20Diagrams/architecture_diagram_cross_attn.png)

### 5. Mixture of Experts / MoE (Model 4)
The key difference: the two signals are kept **separate all the way through** and only combined at the very end via a learned gating network. The gate evaluates the pair and outputs two weights — one for the text expert, one for the graph expert.

This is the architecture we care most about. Keeping the experts separate means we can inspect them. We can see the text score and graph score independently, understand which one drove the decision, and in a downstream system, provide specific **Recourse** to the candidate.

![](../flowcharts/Architecture%20Diagrams/architecture_diagram_moe.png)

---

## Why These Metrics?

HR doesn't care if a score is 0.73 vs 0.81 in absolute terms. What matters is whether the best candidates land at the top of the pile.
- **nDCG@10** — did the top 10 retrieved candidates match the human's top 10?
- **RBO (Rank-Biased Overlap)** — top-weighted list similarity.
- **Spearman ρ** — overall monotonic ranking correlation across all pairs.
- **MAE** — included for completeness.

---

## Results

| Model | MAE ↓ | Spearman ↑ | nDCG@10 ↑ | RBO ↑ |
|---|---|---|---|---|
| TF-IDF | 0.3812 | 0.0874 | 0.6329 | 0.3903 |
| Pure Semantic | 0.3771 | -0.0437 | 0.6204 | 0.4114 |
| Late Fusion | 0.3570 | 0.2274 | 0.6583 | 0.4294 |
| Cross-Attention | 0.3638 | 0.1944 | 0.6094 | 0.3883 |
| **MoE** | **0.3567** | **0.2282** | **0.6759** | **0.4268** |

### MoE Expert Specialisation

- **Expert correlation: -0.3296** — The text expert and graph expert are highly distinct. A negative correlation is fantastic here: it means the experts haven't "collapsed" into doing the exact same thing. When the text score is high, the graph score might be low (and vice versa), allowing the gating network to dynamically balance the two distinct perspectives.

---

## Conclusions

1. **Pure Semantic alone struggles to rank:** The Bi-Encoder semantic model actually performed worse than TF-IDF on Spearman correlation. Semantic text matching alone isn't enough to properly rank ATS data.
2. **Structured skill graphs provide massive gains:** Adding the GNN embeddings (Late Fusion & MoE) resulted in the highest RBO, nDCG, and Spearman scores.
3. **MoE is the optimal choice:** It yields top-tier ranking metrics (nDCG 0.6759) while completely avoiding expert collapse. Because the text and graph scores are generated independently, we can use Model 4 to generate **Explainable AI / Recourse** for rejected candidates.
4. **Cross-Attention was too noisy:** Attending over padded variable-length graph sequences likely introduced noise, resulting in a drop in performance. Clean separation (Late Fusion / MoE) proved much more effective given our 3-epoch training budget.

---

## Files Saved

All four deep learning models are saved as state dicts to `/kaggle/working/`:

- `best_pure_semantic.pth`
- `best_late_fusion.pth`
- `best_cross_attention.pth`
- `best_moe_model.pth`

## Dependencies
transformers==5.2.0
torch
scikit-learn
scipy
seaborn
matplotlib
pandas
numpy
tqdm
