# Notebook 1: Data Preparation for Resume-JD Fit Prediction

## Project Overview

This notebook is the **first stage** of a multi-notebook NLP pipeline that predicts how well a resume matches a job description. The goal is to build a hybrid model combining Transformer-based text understanding with Graph Neural Networks (GNNs) that understand skill relationships — but before any model can be trained, the data must be carefully understood, cleaned, and engineered.

![](../flowcharts/Data%20flow/data_flow_notebook1.png)

---

## Datasets Used

### Primary Dataset: `cnamuangtoun/resume-job-description-fit`
- **Train split:** 6,241 samples
- **Test split:** 1,759 samples
- **Features:** `resume_text`, `job_description_text`, `label`
- **Labels:** `No Fit`, `Potential Fit`, `Good Fit`

### Skills Reference Dataset: `asaniczka/1-3m-linkedin-jobs-and-skills-2024`
- **File used:** `job_skills.csv`
- **Features:** `job_link`, `job_skills`
- **Purpose:** Build the skill vocabulary and co-occurrence graph edges

### Backup Dataset: `AzharAli05/Resume-Screening-Dataset`
- **Size:** 10,174 samples (train only)
- **Features:** `Role`, `Resume`, `Decision`, `Reason_for_decision`, `Job_Description`
- **Labels:** Binary (`select` / `reject`)

The **primary dataset** is used for model training and evaluation because it has a richer, three-class label scheme that captures nuance. The backup dataset serves as a potential augmentation source if needed.

---

## Tech-Domain Filtering

Before any analysis, both the primary dataset and the LinkedIn skills dataset are filtered to retain only **tech-domain** rows. This scoping decision keeps the skill vocabulary and graph edges relevant to the domain the model is deployed in.

**Primary dataset filter:** JDs are checked against a list of 35 technology keywords (`software`, `engineer`, `developer`, `data`, `machine learning`, `AI`, `cloud`, `devops`, `python`, `java`, etc.) using a case-insensitive pattern match on `job_description_text`.

| Split | Before Filter | After Filter |
|---|---|---|
| Train | 6,241 | 6,166 |
| Test | 1,759 | 1,754 |

**Label distribution in filtered train set:**

| Label | Count |
|---|---|
| No Fit | 3,100 |
| Potential Fit | 1,541 |
| Good Fit | 1,525 |

**Label distribution in filtered test set:**

| Label | Count |
|---|---|
| No Fit | 852 |
| Good Fit | 458 |
| Potential Fit | 444 |

**LinkedIn filter:** Job titles (extracted from the `job_link` URL) are matched against a 38-term tech-role pattern covering software engineers, data scientists, ML engineers, DevOps, cybersecurity, and more. This reduces the LinkedIn dataset to **40,745 tech-role rows**.

---

## Label Distribution Analysis

### Why This Matters
Before any modelling decision, understanding class balance is critical. An imbalanced dataset causes a classifier to be biased toward the majority class, giving misleadingly high accuracy while failing on minority classes.

### What We Found
The dataset is **moderately imbalanced** — "No Fit" is roughly twice as common as either positive class. 

![label_distribution_train_set.png](../plots/label_distribution.png)

---

## Word Count Analysis

### Why This Matters — The Transformer Token Limit Problem
Transformer models like BERT and RoBERTa have a hard limit of **512 tokens** (approximately 350–400 words). Any text beyond this limit is simply **truncated** — the model never sees it.

![](../plots/JD_words_count.png)

**Key Observation:** While the median JD is ~330 words (within the Transformer limit), a significant portion of JDs exceed 400 words. And as we show next, the *critical* content of those JDs often appears *after* the cutoff.

---

## The Hidden Requirements Problem (Key Data Insight)

### Hypothesis
Job descriptions typically open with company branding, culture statements, and role overviews. The actual **technical requirements and qualifications** often appear much later in the document.

### Analysis & Result
A function locates the keywords `"requirements"`, `"qualifications"`, and `"what you need"` within each JD and measures the word position at which they first appear.

![requirements_start_word_histogram.png](../plots/Requirements_word_index.png)

**462 job descriptions in the training set had their requirements section appearing *after* the typical Transformer cutoff (~384 words).**

---

## Solution: Smart JD Parsing (Dual Pipeline Design)

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| Use raw JD text as-is | Simple | Truncation cuts off requirements |
| Truncate from the end | Keeps start | Same problem — marketing text first |
| Sliding window / chunking | Captures more | Very slow, complex aggregation |
| **Smart parsing: start from Requirements section** | Captures what matters | Requires heuristic keyword detection |
| Summarization (LLM-based) | Would be ideal | Too slow and expensive at scale |

### Chosen Approach: `smart_parse_jd()`

A simple but effective function that:
1. Scans for section keywords in priority order: `"requirements"`, `"qualifications"`, `"what you need"`, `"what you bring"`, `"skills"`
2. Finds the **first** occurrence and breaks immediately
3. Extracts text *starting from that section*, up to 350 words

This ensures the Transformer reads the **most discriminative content** in the JD rather than the preamble. If no keywords are found, it falls back to the beginning of the document.

The output is stored in a new column: `smart_jd_text`.

## Dual Pipeline for Graph vs. Transformer

### Why Two Pipelines?

The architecture uses two different models that have different text requirements:

1. **Transformer (RoBERTa/BERT):** Works on raw, natural text. It relies on **stopwords, punctuation, and grammar** for context. A sentence like "does NOT require Python" changes meaning entirely if "NOT" is removed.

2. **Graph Neural Network (GNN):** Needs to identify **skill entities** in text. For this, we need clean text and fast keyword matching. Stopwords add noise here, not signal.

| Column | Used By | Processing |
|---|---|---|
| `resume_text` | Transformer | Raw, natural text |
| `smart_jd_text` | Transformer | Smart-parsed, raw text |
| `graph_res_text` | GNN | Lowercased, stripped of special chars (keeps `+` for C++) |
| `graph_jd_text` | GNN | Same cleaning, applied to smart-parsed JD |

The `clean_for_graph()` function applies `re.sub(r'[^a-z\s\+]', ' ', text)` — the `+` is deliberately preserved so that `c++` survives cleaning intact.

---

## Skill Extraction (LinkedIn Skills Vocabulary)

### Why Not Just Use Regex or a Fixed List?

Early iterations used a hand-curated list of ~500 skills and Regex matching. This had two problems:
1. **Speed:** Running 500 Regex patterns per document is O(n×k) — very slow for 6,000+ documents.
2. **Coverage:** 500 skills misses many domain-specific or emerging terms.

### The FlashText Solution

**FlashText** is an O(N) keyword extraction library (based on the Aho-Corasick algorithm). It processes text in a single pass regardless of how many keywords you're searching for.

### Building the Vocabulary

The full pipeline on the LinkedIn tech-filtered dataset (40,745 rows):

1. Explode the comma-separated `job_skills` column into individual skill strings.
2. Lowercase and strip whitespace.
3. Drop skills with fewer than 2 or more than 40 characters.
4. Filter out noise phrases (`"this context does not mention"`, `"none found"`).
5. Apply a **blocklist** of soft skills and non-technical terms — `communication`, `teamwork`, `leadership`, `problem solving`, `bachelor's degree`, `computer science`, `software engineering`, etc. These terms are pervasive in the raw data (e.g. `communication` appears 7,104 times before filtering) but carry no signal for GNN skill-graph matching.
6. Rank by frequency. We found an incredible **155,818 unique skills/phrases**, but we restricted the final vocabulary to the **top 2,500** most frequent terms to keep the graph focused and dense.

The top technical skills after blocklist filtering are: `python` (12,336), `sql` (8,274), `java` (7,984), `aws` (7,582), `kubernetes` (4,860), `javascript` (4,830), `docker` (4,525), `agile` (4,445), `machine learning` (3,975).

After vocabulary construction, a **synonym map** is applied via FlashText's aliasing feature, normalising abbreviations to canonical forms (`ml` → `machine learning`, `k8s` → `kubernetes`, `reactjs` → `react`, `llm` → `large language models`, etc.). The final vocabulary contains **2,502 skills**.

### Skill Extraction Results

Most resumes contain 5–25 extracted skills and most JDs contain 3–15 skills — dense enough to build meaningful skill overlap features, sparse enough to avoid noise.

![](../plots/Density_hard_skills.png)

---

## Building Graph Co-occurrence Edges

To pretrain the GNN, we need to know which skills tend to appear together in job postings. The LinkedIn dataset is used to compute skill co-occurrence counts over the first **50,000 rows**.

### Methodology
For each job posting, all valid skills present in the 2,502-skill vocabulary are extracted. Every combination of two skills appearing in the same posting forms a pair (edge), and the edge weight is the count of postings containing both skills.

### Results

| Metric | Value |
|---|---|
| Total unique skill pairs computed | **740,444** |

All 740,444 edges are saved directly — no sparsification step is applied in this notebook.

### Top 10 Strongest Skill Connections

| Skill A | Skill B | Weight |
|---|---|---|
| python | sql | 5,340 |
| java | python | 5,015 |
| aws | python | 4,514 |
| docker | kubernetes | 3,719 |
| aws | java | 3,646 |
| java | sql | 3,579 |
| aws | sql | 3,132 |
| kubernetes | python | 3,086 |
| aws | kubernetes | 2,995 |
| docker | python | 2,963 |

These relationships reflect real-world technical skill clusters — cloud and container tooling (`aws`, `docker`, `kubernetes`) co-occurring heavily with core languages (`python`, `java`, `sql`). Note that the blocklist successfully removed soft-skill terms like `communication` from these edges; the strongest connections are entirely technical.
...

---

## Outputs Produced

All outputs are saved to `/kaggle/working/` for downstream notebooks. Note that the datasets now include the Groq skill extraction columns as part of the overall pipeline:

| File | Contents | Used By |
|---|---|---|
| `train_clean.csv` | Processed train data with all new columns (including FlashText + Groq skill columns) | Notebook 3 (model training) |
| `test_clean.csv` | Processed test data | Notebook 3 |
| `skill_vocab.json` | 2,502 skill strings (FlashText vocab + synonyms) | Notebook 2 (GNN pretraining) |
| `graph_edges.json` | **740,444** `{source, target, weight}` co-occurrence pairs | Notebook 2 |

### Columns in `train_clean.csv` / `test_clean.csv`

| Column | Description |
|---|---|
| `resume_text` | Raw resume text (for Transformer) |
| `job_description_text` | Raw JD text (original) |
| `label` | Text label: `No Fit` / `Potential Fit` / `Good Fit` |
| `score` | Numeric label: 0.0 / 0.5 / 1.0 |
| `resume_word_count` | Word count of resume |
| `jd_word_count` | Word count of original JD |
| `req_start_word_idx` | Word index where requirements section begins (0 if not found) |
| `smart_jd_text` | Smart-parsed JD starting from requirements section (for Transformer) |
| `graph_res_text` | Cleaned resume text (for GNN) |
| `graph_jd_text` | Cleaned smart JD text (for GNN) |
| `extracted_skills_res` | Skills extracted from resume via FlashText |
| `extracted_skills_jd` | Skills extracted from smart JD via FlashText |
| `skill_count_res` | Number of unique skills in resume |
| `skill_count_jd` | Number of unique skills in smart JD |

---

## Summary of Key Decisions

| Decision | Alternatives | Reason for Choice |
|---|---|---|
| Tech-domain filter applied upfront | Train on all domains | Keeps skill vocab and graph edges domain-relevant; reduces noise from non-tech roles |
| 3-class labels → numeric scores | Binary classification | Captures nuance; enables both classification and regression heads |
| Smart JD parsing from Requirements section | Full raw text | 462 JDs hide requirements past Transformer cutoff; this recovers critical signal |
| FlashText with 2,502 skills + synonyms | Regex with 500 skills | O(N) speed; 5× larger vocabulary at no performance cost; synonym map handles abbreviations |
| Dual text pipelines (raw vs. cleaned) | Single pipeline | Transformer needs stopwords for context; GNN needs clean text for entity matching |
| Soft-skills blocklist in vocab construction | Keep all frequent skills | Removes high-frequency noise (e.g. `communication` at 7,104 raw occurrences) that would otherwise dominate the graph edges |
| All **740,444** co-occurrence edges saved | Sparsify at min weight | No sparsification applied in this notebook; filtering can be done downstream if needed |
