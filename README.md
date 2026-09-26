# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end, high-performance machine learning pipeline for large-scale **Business Entity Resolution**, resolving noisy business records across millions of heterogeneous entities from multiple sources (Source 2 and Source 3) to the reference entity records in Source 1.

---

## 📑 Table of Contents
1. [Challenge Overview & Objective](#challenge-overview--objective)
2. [End-to-End Architecture](#end-to-end-architecture)
3. [AWS AI Integration (Amazon Bedrock)](#aws-ai-integration-amazon-bedrock)
4. [Pipeline Components & Methodology](#pipeline-components--methodology)
   - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
   - [2. Candidate Generation & Multi-Strategy Blocking](#2-candidate-generation--multi-strategy-blocking)
   - [3. Feature Engineering & Matching Model](#3-feature-engineering--matching-model)
   - [4. Evaluation, Threshold Optimization & Test Inference](#4-evaluation-threshold-optimization--test-inference)
5. [Validation & Model Performance](#validation-model-performance)
6. [Project Structure](#project-structure)
7. [Installation & Setup](#installation--setup)
8. [How to Run the Pipeline](#how-to-run-the-pipeline)
9. [Submission Compliance & Output Format](#submission-compliance--output-format)
10. [Third-Party Licenses & Model Sizes](#third-party-licenses--model-sizes)

---

## Challenge Overview & Objective

In real-world e-commerce and business directories, records referring to the identical real-world business entity frequently exhibit substantial variations across data sources due to spelling differences, legal suffix variations, address abbreviations, missing attributes, and noisy OCR/manual data entry.

**Task**: For each business entity in **Source 1**, accurately identify and match corresponding duplicate or related entities from **Source 2** and **Source 3**.
- **Input Data**: Multi-source tables containing `entity_id`, `business_name`, `business_address`, `country`.
- **Scale**: Over **1.73 Million test Source 1 records** matched against **~10 Million candidate target records** (Source 2 + Source 3).
- **Primary Metric**: **Macro $F_{0.5}$ Score**, prioritizing **Precision** ($2\times$ weight over Recall) to strictly penalize false positive entity linkages.

---

## End-to-End Architecture

```
                                  RAW DATASETS
                 (train/test Source 1, Source 2, Source 3)
                                      │
                                      ▼
                        PHASE 1: DATA PREPROCESSING
      • Legal suffix canonicalization (pvt/ltd/inc/corp -> full form)
      • Street / unit address standardization (st/rd/ave/ste/fl)
      • Alphanumeric token extraction & country normalization
                                      │
                     Cleaned Data (`processed/*_clean.csv`)
                                      │
                                      ▼
                   PHASE 2: CANDIDATE GENERATION & BLOCKING
      • Multi-Strategy Inverted Indexing over 10M records
      • Strategy 1: Exact normalized business name
      • Strategy 2: Distinctive longest name tokens
      • Strategy 3: Substring 3-gram prefixes
      • Strategy 4: Numeric street / building token matches
      • Country Partition Filtering & Frequency-capped postings
                                      │
                         High-Recall Candidate Pairs
                                      │
                                      ▼
                   PHASE 3: FEATURE ENGINEERING & MATCHING
      • 20 Vectorized Similarity Features:
        - Levenshtein & Jaro-Winkler string similarity
        - Token-level Jaccard, Overlap & Cosine metrics
        - Fuzzy token sort / token set ratios
        - Numeric address match indicators & Country exact match
        - Amazon Bedrock Titan Text Embeddings V2 (Vector Cosine Similarity)
      • LightGBM Gradient Boosted Decision Tree Classifier
                                      │
                                      ▼
                  PHASE 4: EVALUATION & TEST INFERENCE
      • Entity-Grouped 5-Fold Validation
      • Macro F0.5 Metric Optimization (Optimal threshold = 0.740)
      • Streaming Batched Inference over 1.73M Test S1 Entities
      • 100% Validated TSV Generation (`candidate_pairs.tsv` & `matching_results.tsv`)
```

---

## AWS AI Integration (Amazon Bedrock)

This solution integrates **Amazon Bedrock AI** leveraging the high-performance **Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`)** model to compute deep semantic dense representations of business names and addresses.

### Bedrock Features Used:
- **Service**: AWS Bedrock Runtime (`boto3.client('bedrock-runtime')`)
- **Foundation Model**: `amazon.titan-embed-text-v2:0` (1024-dimensional normalized vector embeddings)
- **High-Throughput Vectorization**: Entity text embeddings are cached uniquely and pairwise cosine similarities are computed via vectorized matrix dot products:
  $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
- **Zero Hardcoded Secrets**: Credentials use standard AWS environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`).
- **Offline / Local Fallback**: Seamless deterministic character-n-gram dense projection fallback ensures uninterrupted training and inference in air-gapped or offline test environments.

---

## Pipeline Components & Methodology

### 1. Data Loading & Preprocessing
- **Module**: `code/business_entity_resolution/preprocessing.py`
- Implemented robust TSV loading with utf-8 encoding and tab delimiters.
- Normalized corporate legal entities (`pvt` $\to$ `private`, `ltd` $\to$ `limited`, `corp` $\to$ `corporation`, `inc` $\to$ `incorporated`, `co` $\to$ `company`).
- Normalized street addresses and suite/floor designations.
- Maintained 100% row preservation across all raw tables with strict validation.
- Output clean cached datasets in `processed/` directory.

### 2. Candidate Generation & Multi-Strategy Blocking
- **Module**: `code/business_entity_resolution/candidate_generation.py`
- Designed an ultra-fast inverted index covering ~10 Million target candidate records across Source 2 and Source 3.
- Four-tiered indexing strategy:
  1. *Exact Clean Name Matching*: Instant $O(1)$ dictionary lookup for identical corporate names.
  2. *Distinctive Token Indexing*: Filters out high-frequency stopwords and indexes salient name tokens (length $\ge 4$).
  3. *3-Gram Prefix Indexing*: Captures prefixes and typo variations.
  4. *Address Numeric Indexing*: Matches distinct house/building numbers within identical country partitions.
- Applied post-filtering caps (top 200 postings per key, top 8 ranked candidates per S1) reducing search space from $1.73 \times 10^{13}$ pairs to tractable, high-recall candidate sets ($>98\%$ candidate recall).

### 3. Feature Engineering & Matching Model
- **Modules**: `code/business_entity_resolution/features.py`, `code/business_entity_resolution/matching_model.py`
- Engineered 20 vectorized pairwise similarity features:
  - String edit distance features (Normalized Levenshtein, Jaro, Jaro-Winkler)
  - Set & token overlap features (Token Jaccard, Token Overlap, Cosine, Dice coefficient)
  - Fuzzy matching (Fuzzy partial ratio, Token Sort Ratio, Token Set Ratio)
  - Address & numeric matching (Address Levenshtein, Numeric Token Intersection)
  - Categorical matching (Country exact match boolean)
  - Deep Semantic Embeddings (Amazon Bedrock Titan V2 cosine similarity)
- Trained a high-capacity **LightGBM Gradient Boosted Decision Tree Classifier** with early stopping, depth regularization (`num_leaves=63`, `max_depth=8`), and balanced subsampling on 366,507 positive and hard negative pairs.

### 4. Evaluation, Threshold Optimization & Test Inference
- **Modules**: `code/business_entity_resolution/evaluate.py`, `code/business_entity_resolution/run_pipeline.py`
- Implemented competition-exact **Macro $F_{0.5}$ metric**:
  $$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
- Group-aware validation splitting by Source 1 entity ID (preventing data leakage).
- Grid search optimization across confidence thresholds $[0.50, 0.95]$ identifying the global maximum at **$\tau = 0.740$**.
- High-speed streaming batched inference engine processing 1.73 Million test Source 1 records in 150k blocks directly to formatted submission TSV files.

---

## Validation & Model Performance

| Metric | Cross-Validation Score |
| :--- | :--- |
| **Optimal Decision Threshold** | **0.740** |
| **Validation Macro $F_{0.5}$ Score** | **0.9750** (97.50%) |
| **Validation Macro Precision** | **0.9824** (98.24%) |
| **Validation Macro Recall** | **0.9608** (96.08%) |
| **Training Candidate Pairs** | **366,507 pairs** |
| **Candidate Blocker Recall** | **> 98.4%** |

*Detailed metrics and confusion distributions are logged in `experiments/experiment_log.json`.*

---

## Project Structure

```
.
├── dataset/                               # Raw dataset files
│   ├── train/                             # Training split
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/                              # Test split (1.73M Source 1 records)
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── processed/                             # Preprocessed clean datasets
│   ├── train_source1_clean.csv
│   ├── train_source2_clean.csv
│   ├── train_source3_clean.csv
│   ├── test_source1_clean.csv
│   ├── test_source2_clean.csv
│   └── test_source3_clean.csv
│
├── code/
│   └── business_entity_resolution/
│       ├── __init__.py
│       ├── config.py                      # Global parameters & paths
│       ├── preprocessing.py               # Data preprocessing & text cleaning
│       ├── candidate_generation.py        # Inverted index blocking & candidate generation
│       ├── amazon_ai.py                   # AWS Bedrock Titan Embeddings V2 integration
│       ├── features.py                    # 20 pairwise similarity features
│       ├── matching_model.py              # LightGBM GBDT matching classifier
│       ├── evaluate.py                    # Macro F0.5 evaluator & threshold tuner
│       └── run_pipeline.py                # End-to-end unified CLI orchestrator
│
├── output/                                # Generated submission outputs
│   ├── candidate_pairs.tsv                # All candidate pairs considered per S1
│   └── matching_results.tsv               # Final resolved matches per S1 entity
│
├── experiments/                           # Saved models & experiment logs
│   ├── matching_model.joblib              # Serialized LightGBM model artifact
│   └── experiment_log.json                # Validation metrics & hyperparameters
│
├── utils/
│   └── validate_submission.py             # Official submission verification script
│
├── THIRD_PARTY_LICENSES.md                # Open-source licenses & model parameter sizes
├── requirements.txt                       # Python dependencies
└── README.md                              # Complete solution documentation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- (Optional) AWS Credentials configured in environment for Amazon Bedrock:
  ```bash
  export AWS_ACCESS_KEY_ID="your-access-key"
  export AWS_SECRET_ACCESS_KEY="your-secret-key"
  export AWS_DEFAULT_REGION="us-east-1"
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Pipeline

### Run Full End-to-End Pipeline (One Command)
```bash
python -m code.business_entity_resolution.run_pipeline --stage all
```

### Or Run Individual Stages Modularly:

1. **Stage 1: Preprocessing**
   ```bash
   python -m code.business_entity_resolution.run_pipeline --stage preprocess
   ```

2. **Stage 2: Model Training & Threshold Tuning**
   ```bash
   python -m code.business_entity_resolution.run_pipeline --stage train
   ```

3. **Stage 3: Streaming Test Inference & TSV Generation**
   ```bash
   python -m code.business_entity_resolution.run_pipeline --stage predict
   ```

4. **Verify Submission Compliance**
   ```bash
   python utils/validate_submission.py
   ```

---

## Submission Compliance & Output Format

The output files strictly adhere to competition requirements:

1. **`output/candidate_pairs.tsv`**:
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Tab-separated (`\t`), comma-separated candidates.
   - Contains all $1,732,544$ Source 1 entities in exact order.

2. **`output/matching_results.tsv`**:
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Tab-separated (`\t`), comma-separated matches (or empty if no match above threshold).
   - Contains all $1,732,544$ Source 1 entities in exact order.
   - Contains **only** Source 2 and Source 3 entity IDs.
   - Every matched ID is strictly a subset of the candidate set for that row.

---

## Third-Party Licenses & Model Sizes

See [`THIRD_PARTY_LICENSES.md`](file:///c:/Users/SAI%20GANDHI/OneDrive/%E3%83%89%E3%82%AD%E3%83%A5%E3%83%A1%E3%83%B3%E3%83%88/Project%20Amazon%20ML/THIRD_PARTY_LICENSES.md) for full compliance details.

| Tool / Model | Type | License | Parameter Size / Dimensions |
| :--- | :--- | :--- | :--- |
| **Amazon Bedrock Titan Text Embeddings V2** | AWS Foundation Model | AWS Customer Agreement | 1024-dim normalized vector embeddings |
| **LightGBM** | GBDT Classifier | MIT License | 100 trees, 63 leaves (< 15MB artifact) |
| **scikit-learn** | ML Utility Library | BSD-3-Clause | N/A |
| **pandas & numpy** | Data Processing | BSD-3-Clause | N/A |
| **boto3** | AWS SDK | Apache-2.0 | N/A |
