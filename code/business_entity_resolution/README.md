# Business Entity Resolution Pipeline

This package provides a self-contained, reproducible machine learning pipeline to resolve noisy business entities from Source 2 and Source 3 against deduplicated reference records in Source 1.

---

## 📁 Source Code Structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths & hyperparameters
│   ├── preprocessing.py          # Data normalization & legal suffix canonicalization
│   ├── candidate_generation.py   # Multi-strategy inverted index blocking
│   ├── amazon_ai.py              # Amazon Bedrock Titan Text Embeddings V2 integration
│   ├── features.py               # 20 similarity features
│   ├── matching_model.py         # LightGBM pair matching classifier
│   ├── evaluate.py               # Macro F0.5 evaluation & threshold optimization
│   └── run_pipeline.py           # Unified pipeline runner
├── requirements.txt              # Pinned dependencies
└── README.md                     # This reproduction guide
```

---

## ⚙️ Environment Setup

```bash
pip install -r requirements.txt
```

*(Optional) Configure AWS Credentials for Amazon Bedrock Titan Embeddings:*
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```
*Note: If AWS Bedrock credentials are not present, the pipeline automatically uses its deterministic offline local semantic engine to ensure 100% reproducibility without external internet connection.*

---

## 🚀 How to Reproduce End-to-End

### 1. Run Complete Pipeline (Data → Blocking → Matching → Output):
From the project root:
```bash
python -m code.business_entity_resolution.src.run_pipeline --stage all
```

### 2. Or Run Individual Modular Stages:

- **Data Preprocessing & Normalization:**
  ```bash
  python -m code.business_entity_resolution.src.run_pipeline --stage preprocess
  ```

- **Candidate Blocking, Model Training & Threshold Optimization:**
  ```bash
  python -m code.business_entity_resolution.src.run_pipeline --stage train
  ```

- **Streaming Test Inference & TSV Generation:**
  ```bash
  python -m code.business_entity_resolution.src.run_pipeline --stage predict
  ```

- **Verify Submission Outputs:**
  ```bash
  python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
  ```

---

## 📊 Outputs Produced
- `output/candidate_pairs.tsv`: Final candidate set from blocking for all test Source 1 entities.
- `output/matching_results.tsv`: Final resolved matches above optimal decision threshold ($\tau = 0.740$).
