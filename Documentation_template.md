# Business Entity Resolution — Methodology & Solution Documentation

**Competition**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Primary Metric**: Macro $F_{0.5}$ Score  
**Evaluation Focus**: High Precision Entity Resolution, Scalable Candidate Blocking, Zero Data Leakage  

---

## 1. Methodology Used

### 1.1 Problem Formulation
Business Entity Resolution (ER) in heterogeneous multi-source commercial environments requires matching entity records from independent data sources ($S_2$ and $S_3$) to deduplicated reference entities ($S_1$) without shared unique identifiers. Records are corrupted by OCR errors, abbreviation discrepancies, missing address fields, and phonetic/transliteration noise.

The optimization objective is the **Macro $F_{0.5}$ score** computed across all $S_1$ entities:
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

The $F_{0.5}$ metric places $2\times$ more importance on **Precision** than Recall because false merges (erroneously combining distinct commercial businesses) carry severe consequences in real-world commercial platforms compared to missed links.

### 1.2 End-to-End Pipeline Architecture
The solution is structured as a 4-phase decoupled pipeline:
1. **Deterministic Preprocessing & Canonicalization**: Standardizes legal suffixes, cleans street and unit addresses, extracts alphanumeric tokens, and preserves country labels.
2. **Multi-Strategy Inverted Index Blocking**: Reduces the $1.73 \times 10^{13}$ pairwise comparison space down to a tight candidate set ($\le 8$ candidates per $S_1$ entity) while achieving $>98.4\%$ recall ceiling.
3. **Multi-Faceted Feature Engineering**: Extracts 20 similarity features spanning string edit distances, token set metrics, character n-grams, numeric address matching, and **Amazon Bedrock Titan Text Embeddings V2** cosine similarity.
4. **Gradient Boosted Matching Model & Threshold Tuning**: LightGBM binary classifier trained on positive and hard negative pairs with entity-grouped splits, optimized for the global Macro $F_{0.5}$ threshold peak ($\tau = 0.740$).
5. **Streaming Batch Inference Engine**: Evaluates $1.73\text{M}$ test $S_1$ records in $150\text{k}$ chunked streams, producing formatted `candidate_pairs.tsv` and `matching_results.tsv`.

---

## 2. Candidate Generation & Blocking Strategy

### 2.1 The Scalability Challenge & Reduction Ratio
Directly comparing $1,732,544$ Source 1 records against $9,969,589$ target records ($S_2 + S_3$) involves $1.73 \times 10^{13}$ pairwise comparisons. To achieve real-time streaming capability and minimize candidate set size (rewarded in competition ranking), we developed a multi-tier inverted index blocker.

### 2.2 Four-Tiered Blocking Indexing Structure
Target entities ($S_2$ and $S_3$) are indexed in memory across 4 complementary lookup tables:

1. **Exact Clean Name Hash Index**:
   - Maps normalized business names directly to candidate entity IDs for instantaneous $O(1)$ lookup of clean matches.
2. **Distinctive Token Inverted Index**:
   - Indexes salient name tokens (length $\ge 4$) after filtering out high-frequency generic corporate stopwords (`company`, `enterprises`, `services`, `group`, `holdings`, `india`, `international`).
3. **Character 3-Gram Prefix Index**:
   - Indexes initial 3-character prefixes of normalized business names to recover typos and spelling variations.
4. **Address Numeric Inverted Index**:
   - Indexes distinct house numbers, street numbers, and PIN/postal codes within country partitions to catch businesses sharing physical premises.

### 2.3 Blocking Optimization & Candidate Capping
- **Posting List Frequency Capping**: Tokens appearing in more than $200$ target records are truncated to prevent runaway complexity on generic terms.
- **Per-Entity Candidate Selection**: Candidates are gathered in priority order (Exact Name $\to$ Salient Tokens $\to$ 3-Gram $\to$ Address Numerics) and capped at a maximum of **8 candidates per $S_1$ entity**.
- **Candidate Size & Recall Performance**:
  - **Reduction Ratio**: $> 99.999\%$ reduction in comparison space.
  - **Candidate Blocker Recall**: **$> 98.4\%$** on held-out ground truth.
  - **Average Candidates per $S_1$**: $\sim 3.8$ candidates, producing an exceptionally compact `candidate_pairs.tsv` file that maximizes the final ranking criteria.

---

## 3. Model Architecture & Feature Engineering

### 3.1 Feature Engineering (20 Vectorized Pairwise Signals)
For every candidate pair $(S_1, S_{cand})$, 20 engineered features capture syntactic, token-level, structural, and semantic similarities:

| Feature Category | Features Extracted | Description |
| :--- | :--- | :--- |
| **String Edit Distances** | `exact_name_match`, `name_prefix_match_3`, `name_prefix_match_5`, `name_char_overlap_ratio`, `name_length_diff`, `name_length_ratio` | Measures normalized Levenshtein-like and character overlap metrics. |
| **Token Set Metrics** | `name_token_jaccard`, `name_token_overlap`, `name_token_dice`, `name_char_3gram_jaccard` | Word-level set overlap: $J(A, B) = \frac{\|A \cap B\|}{\|A \cup B\|}$ and Overlap Coefficient $\frac{\|A \cap B\|}{\min(\|A\|, \|B\|)}$. |
| **Address & Spatial Signals** | `address_token_jaccard`, `address_token_overlap`, `address_char_3gram_jaccard`, `address_numeric_match`, `address_numeric_jaccard`, `address_length_diff` | Numeric street/postal code intersection and token similarity. |
| **Categorical & Metadata** | `country_exact_match`, `country_missing`, `is_source2`, `is_source3` | Country consistency check and target source origin attribution. |
| **AWS Bedrock Semantic Embedding** | `bedrock_embedding_cosine_sim` | Pairwise cosine similarity from Amazon Titan Text Embeddings V2: $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$. |

### 3.2 Machine Learning Model: LightGBM Gradient Boosted Trees
- **Model Choice**: **LightGBM** (`LGBMClassifier`), fully open-source (MIT License) with under 15MB model footprint (< 8 Billion parameters constraint).
- **Hyperparameters**:
  - `n_estimators`: 100
  - `learning_rate`: 0.08
  - `num_leaves`: 63
  - `max_depth`: 8
  - `subsample`: 0.85
  - `colsample_bytree`: 0.85
- **Data Leakage Prevention**: Split training data strictly at the **Source 1 Entity ID level** (80% Train, 20% Validation), ensuring no entity in validation appeared during training.
- **Hard Negative Mining**: Trained on $366,507$ pairs combining true ground truth positive links with blocking-derived hard negatives (distractor businesses sharing similar names or locations).

---

## 4. Evaluation, Metric Optimization & AWS AI Integration

### 4.1 Macro $F_{0.5}$ Metric & Threshold Optimization
Precision is prioritized over Recall ($2\times$). A threshold sweep over $[0.50, 0.95]$ on the validation set identified the global optimal operating threshold:
- **Optimal Decision Threshold ($\tau$)**: **`0.740`**
- **Validation Macro $F_{0.5}$ Score**: **`0.9750`** ($97.50\%$)
- **Validation Macro Precision**: **`0.9824`** ($98.24\%$)
- **Validation Macro Recall**: **`0.9608`** ($96.08\%$)

### 4.2 Singleton (Zero-Match) Entity Handling
In accordance with competition rules, Source 1 entities with no true matches score a full $1.0$ if an empty match list is predicted, and $0.0$ if any false positive match is made. With optimal threshold $\tau = 0.740$, our model predicted zero matches for $47.95\%$ of test entities ($830,782$ entities), avoiding false merges and boosting the macro score.

### 4.3 AWS AI Integration (Amazon Bedrock)
- **Model**: `amazon.titan-embed-text-v2:0`
- **Architecture**: Boto3 client dynamically creates 1024-dimensional normalized dense vectors.
- **Vectorized Caching**: Embeddings are computed only once per distinct entity string and stored in an in-memory cache, enabling ultra-fast dot-product matrix similarity calculation for millions of candidate pairs.
- **Security & Portability**: Fully compatible with AWS IAM credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) with deterministic offline fallback for air-gapped environments.

---

## 5. Submission Package Structure & Reproducibility

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv       # 1,732,544 rows, TSV, validated
│   └── candidate_pairs.tsv        # 1,732,544 rows, TSV, validated
├── code/
│   └── business_entity_resolution/
│       ├── config.py              # Centralized paths and hyperparameters
│       ├── preprocessing.py       # Data normalization & cleaning
│       ├── candidate_generation.py# Inverted index blocking engine
│       ├── amazon_ai.py           # Amazon Bedrock Titan Embeddings
│       ├── features.py            # 20 similarity features
│       ├── matching_model.py      # LightGBM pair classifier
│       ├── evaluate.py            # Macro F0.5 metric & threshold optimizer
│       ├── run_pipeline.py        # End-to-end CLI runner
│       └── __init__.py
├── utils/
│   └── validate_submission.py     # Official submission validator
├── experiments/
│   ├── matching_model.joblib      # Serialized model artifact
│   └── experiment_log.json        # Training metrics & validation logs
├── THIRD_PARTY_LICENSES.md        # Open-source licenses & model parameters
├── requirements.txt               # Dependencies
└── Documentation_template.md      # This methodology documentation
```

### Reproducibility Commands:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline end-to-end (preprocess -> train -> predict)
python -m code.business_entity_resolution.run_pipeline --stage all

# 3. Validate submission files
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
