# Third-Party Licenses & Model Verification

This document provides complete verification of third-party software packages, models, and dependencies used in the Business Entity Resolution solution for the Amazon ML Challenge 2026.

---

## 1. Machine Learning & Core Software Libraries

| Package | Version | License | Permissible for Competition | Purpose / Role |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | `>=4.0.0` | **MIT License** | Yes | Primary Gradient Boosted Decision Tree (GBDT) pair classifier |
| **Scikit-Learn** | `>=1.3.0` | **BSD-3-Clause** | Yes | Fallback `HistGradientBoostingClassifier`, train/validation splitting, metrics |
| **Boto3** | `>=1.30.0` | **Apache 2.0** | Yes | AWS SDK for Python (Amazon Bedrock integration) |
| **Botocore** | `>=1.30.0` | **Apache 2.0** | Yes | AWS low-level core library |
| **Pandas** | `>=2.0.0` | **BSD-3-Clause** | Yes | Tabular data manipulation and TSV/CSV I/O |
| **NumPy** | `>=1.26.0` | **BSD-3-Clause** | Yes | Vectorized numerical and matrix operations |
| **SciPy** | `>=1.10.0` | **BSD-3-Clause** | Yes | Scientific and sparse matrix utilities |

---

## 2. Amazon / AWS AI Foundation Model Verification

| Service / Model | Provider | Model ID | Embedding Dimension | Service Terms / License | Role in Solution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Amazon Bedrock Titan Text Embeddings V2** | Amazon Web Services | `amazon.titan-embed-text-v2:0` | 256 / 512 / 1024 | AWS Service Terms (Bedrock) | High-density semantic text representations and pairwise cosine similarity |
| **Amazon Bedrock Titan Text Embeddings V1** (alt) | Amazon Web Services | `amazon.titan-embed-text-v1` | 1536 | AWS Service Terms (Bedrock) | Alternative embedding foundation model |

### Model Constraints & Compliance Check:
- **Parameter Size**: Compact embedding representation (256-dimensional vector per entity record) ensuring fast inference and low memory overhead.
- **External Data Augmentation**: **None**. Embeddings are generated strictly from the competition-provided text fields (`business_name`, `business_address`, `country`).
- **Data Privacy & Governance**: Adheres strictly to AWS data protection — no customer data is used to train base foundation models.
