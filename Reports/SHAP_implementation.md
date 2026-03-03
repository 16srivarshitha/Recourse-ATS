# Recourse-ATS: Phase 2 Report (SHAP Implementation)
**Notebook:** 6 - Feature Attribution and Explainability  
**Date:** March 3, 2026

## 1. Objectives
The goal of Notebook 6 was to implement the **SHapley Additive exPlanations (SHAP)** framework to generate feature attribution explanations for our Resume-Job Description matching models. This phase is critical for satisfying the "Explainability" requirement of the project proposal, allowing us to move beyond black-box scoring and identify exactly which words and skills influenced the model's decision.

## 2. Methodology and Implementation

### A. The Challenge: Multi-Modal Explanation
Standard SHAP implementations for NLP assume a single text input. However, our architectures (Baseline, MoE, and Cross-Attention) are **multi-modal**: they accept Resume Text, Resume Graph IDs, and Job Description Graph IDs simultaneously. 
*   *Problem:* If SHAP masks a word like "Python" in the text, but the Graph ID for Python is still passed to the model, the model will "cheat" and the attribution score for "Python" will be incorrectly zero.
*   *Solution:* We engineered a **Dynamic SHAP Wrapper**. This custom function intercepts SHAP's masked text variations and dynamically re-extracts the Skill IDs for every single permutation before passing them to the model. This ensures that when SHAP hides a skill in the text, it is also hidden from the graph, guaranteeing accurate attribution.

### B. Implementation Details
1.  **Architecture Definitions:** We redefined all three model classes (GraphEnhancedCrossEncoder, MoE_GraphCrossEncoder, CrossAttention_GraphEncoder) within the notebook to ensure compatibility with the SHAP environment.
2.  **Weight Loading:** We loaded the pre-trained weights (`best_fusion_model.pth`, `best_moe_model.pth`, `best_crossattn_model.pth`) and the Graph Embeddings (`skill_embeddings.npy`) from Phase 1.
3.  **Tokenization Handling:** To prevent tokenizer crashes on long documents, we implemented a truncation preprocessing step, limiting resume length to the first 350 words and job descriptions to 100 words.

### C. The Explanation Loop
Due to the high computational cost of SHAP (which requires hundreds of forward passes per explanation), running the analysis on the entire test set (1,759 resumes) was infeasible. Instead, we selected a representative sample of **15 candidates**, including:
*   **True Positives:** High-scoring resumes correctly identified by the model.
*   **False Negatives:** Qualified candidates that the model rejected (critical for Recourse analysis).
*   **False Positives:** Unqualified candidates that the model accepted.

We iterated through this sample using a `shap.Explainer` with a custom `shap.maskers.Text` masker. The loop successfully generated SHAP values for all 15 candidates across all three model architectures.

## 3. Outputs and Artifacts
The notebook successfully produced the following deliverables:
1.  **Interactive Visualizations:** HTML-based force plots were generated within the notebook, visually highlighting positive drivers (words increasing the score) in red and negative drivers (words decreasing the score) in blue.
2.  **Serialized Data:** The complete set of SHAP values for all 45 explanations (15 candidates × 3 models) was serialized and saved to `all_shap_results.pkl`. This file preserves the exact attribution score for every token, enabling the deep comparative analysis (Text vs. Graph reliance) and Recourse Generation in the final phase.

## 4. Conclusion
We have successfully enabled "White-Box" transparency for our custom neural architectures. The Dynamic Wrapper solved the multi-modal masking problem, and the serialized outputs are now ready for the final quantitative analysis of model fairness and skill reliance.