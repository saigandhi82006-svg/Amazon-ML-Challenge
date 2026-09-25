# Business Entity Resolution — ML Challenge 2026

This repository contains the machine learning solution for the Business Entity Resolution Challenge 2026. The objective is to resolve and match noisy business records from secondary sources (Source 2 and Source 3) to the reference entity records in Source 1.

---

## Project Structure

```
.
├── dataset/                    # Raw input TSV files (or train_ / test folders)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── processed/                  # Generated preprocessed CSV datasets
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
│       └── preprocessing.py    # Member 1 Preprocessing Module
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Member 1 — Data Preprocessing

### Overview
The preprocessing pipeline is built by **Member 1** to provide clean, normalized, and validated data ready for downstream Candidate Generation / Blocking (Member 2) and Matching Model / Feature Engineering (Member 3).

### What Preprocessing Does:
1. **Data Loading & Inspection**: Loads all TSV files safely with tab delimiters, checks schemas, inspects row counts, null values, duplicate IDs, and displays summaries.
2. **Business Name Normalization (`normalize_name`)**:
   - Converts to lowercase.
   - Replaces `&` with `and`.
   - Normalizes common legal suffix abbreviations conservatively (`pvt`/`pvt.` -> `private`, `ltd`/`ltd.` -> `limited`, `corp`/`corp.` -> `corporation`, `inc`/`inc.` -> `incorporated`, `co`/`co.` -> `company`).
   - Removes unnecessary punctuation while preserving alphanumeric characters.
   - Collapses and strips whitespace.
3. **Business Address Normalization (`normalize_address`)**:
   - Converts to lowercase.
   - Replaces `&` with `and`.
   - Normalizes road and unit abbreviations conservatively (`rd` -> `road`, `st` -> `street`, `ave` -> `avenue`, `blvd` -> `boulevard`, `hwy` -> `highway`, `ln` -> `lane`, `dr` -> `drive`, `apt` -> `apartment`, `ste` -> `suite`, `fl` -> `floor`).
   - Preserves house numbers, PIN/postal codes, and all meaningful address tokens.
4. **Country Normalization (`normalize_country`)**:
   - Strips whitespace and converts to lowercase.
   - Preserves all countries (US, India, France, etc.) without filtering.
5. **Preservation & Feature Enrichment**:
   - Never overwrites or modifies original raw columns (`entity_id`, `business_name`, `business_address`, `country`).
   - Adds clean columns: `business_name_clean`, `business_address_clean`, `country_clean`.
   - Adds helper columns: `name_tokens`, `name_length`, `address_length`.
6. **Data Quality Validation**:
   - Verifies 100% row preservation across all datasets.
   - Verifies entity IDs remain unaltered.
   - Verifies no records were dropped.
7. **Saving Processed Data**:
   - Saves reproducible CSV files in the `processed/` directory.

### How to Run Preprocessing

To run the complete data loading, inspection, normalization, validation, and saving pipeline:

```bash
python code/business_entity_resolution/preprocessing.py
```

### Module Usage in Python

Downstream teammates (Members 2 and 3) can import the functions directly:

```python
from code.business_entity_resolution import (
    load_all_data,
    preprocess_dataframe,
    normalize_name,
    normalize_address,
    normalize_country,
    extract_name_tokens,
)

# Load raw datasets dictionary
raw_data = load_all_data()

# Or preprocess any custom DataFrame
df_clean = preprocess_dataframe(raw_data["train_source1"])
```
