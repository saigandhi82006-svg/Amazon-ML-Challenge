"""Business Entity Resolution — Data Loading & Preprocessing Pipeline.

This module provides data loading, inspection, normalization, quality validation,
and data persistence for the Business Entity Resolution pipeline.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure UTF-8 stdout/stderr for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

# Expected columns
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

# Pre-compiled regex patterns for business name normalization
NAME_LEGAL_SUFFIXES = {
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "corp": "corporation",
    "corp.": "corporation",
    "inc": "incorporated",
    "inc.": "incorporated",
    "co": "company",
    "co.": "company",
}

# Pre-compiled regex patterns for address normalization
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "hwy": "highway",
    "hwy.": "highway",
    "ln": "lane",
    "ln.": "lane",
    "dr": "drive",
    "dr.": "drive",
    "apt": "apartment",
    "apt.": "apartment",
    "ste": "suite",
    "ste.": "suite",
    "fl": "floor",
    "fl.": "floor",
}

PUNCTUATION_REGEX = re.compile(r"[\.,;:!?\'\"/\\#\$\%\^\*\(\)\[\]\{\}\<\>\=\+\-\_\~\|`@•·°©®™\u2000-\u206F\u2E00-\u2E7F\ufeff]")
WHITESPACE_REGEX = re.compile(r"\s+")


def get_project_root() -> Path:
    """Resolve the project root directory in a cross-platform manner."""
    curr = Path(__file__).resolve().parent
    for _ in range(5):
        if (
            (curr / "dataset").exists()
            or (curr / "processed").exists()
            or (curr / "output").exists()
            or (curr / "train").exists()
            or (curr / "train_").exists()
        ):
            return curr
        if curr.parent == curr:
            break
        curr = curr.parent
    return Path.cwd().resolve()


def find_data_paths(project_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """Locate the train and test directories flexibly within the project root."""
    if project_root is None:
        project_root = get_project_root()

    train_candidates = [
        project_root / "dataset" / "train",
        project_root / "train",
        project_root / "train_",
    ]
    train_dir = None
    for cand in train_candidates:
        if cand.exists() and cand.is_dir():
            train_dir = cand
            break

    test_candidates = [
        project_root / "dataset" / "test",
        project_root / "test",
    ]
    test_dir = None
    for cand in test_candidates:
        if cand.exists() and cand.is_dir():
            test_dir = cand
            break

    if train_dir is None:
        raise FileNotFoundError(
            f"Could not locate training directory. Checked: {[str(c) for c in train_candidates]}"
        )
    if test_dir is None:
        raise FileNotFoundError(
            f"Could not locate test directory. Checked: {[str(c) for c in test_candidates]}"
        )

    return train_dir, test_dir


def normalize_name(text: Any) -> str:
    """Normalize a business name string conservatively."""
    if text is None or pd.isna(text):
        return ""
    
    t = str(text).lower()
    t = t.replace("&", " and ")
    
    tokens = t.split()
    replaced_tokens = []
    for tok in tokens:
        clean_tok = tok.strip(".,;:'\"-")
        if clean_tok in NAME_LEGAL_SUFFIXES:
            replaced_tokens.append(NAME_LEGAL_SUFFIXES[clean_tok])
        else:
            replaced_tokens.append(tok)
    t = " ".join(replaced_tokens)
    
    t = PUNCTUATION_REGEX.sub(" ", t)
    t = WHITESPACE_REGEX.sub(" ", t).strip()
    return t


def normalize_address(text: Any) -> str:
    """Normalize a business address string conservatively."""
    if text is None or pd.isna(text):
        return ""
    
    t = str(text).lower()
    t = t.replace("&", " and ")
    
    tokens = t.split()
    replaced_tokens = []
    for tok in tokens:
        clean_tok = tok.strip(".,;:'\"-/#")
        if clean_tok in ADDRESS_ABBREVIATIONS:
            replaced_tokens.append(ADDRESS_ABBREVIATIONS[clean_tok])
        else:
            replaced_tokens.append(tok)
    t = " ".join(replaced_tokens)
    
    t = PUNCTUATION_REGEX.sub(" ", t)
    t = WHITESPACE_REGEX.sub(" ", t).strip()
    return t


def normalize_country(text: Any) -> str:
    """Normalize country string."""
    if text is None or pd.isna(text):
        return ""
    return str(text).strip().lower()


def extract_name_tokens(clean_name: str) -> List[str]:
    """Extract individual words/tokens from cleaned business name."""
    if not clean_name:
        return []
    return clean_name.split()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a single source dataframe."""
    df_clean = df.copy()
    
    names_raw = df_clean["business_name"].tolist()
    addresses_raw = df_clean["business_address"].tolist()
    countries_raw = df_clean["country"].tolist()
    
    clean_names = [normalize_name(n) for n in names_raw]
    clean_addresses = [normalize_address(a) for a in addresses_raw]
    clean_countries = [normalize_country(c) for c in countries_raw]
    
    name_tokens = [extract_name_tokens(n) for n in clean_names]
    name_lengths = [len(n) for n in clean_names]
    address_lengths = [len(a) for a in clean_addresses]
    
    df_clean["business_name_clean"] = clean_names
    df_clean["business_address_clean"] = clean_addresses
    df_clean["country_clean"] = clean_countries
    df_clean["name_tokens"] = name_tokens
    df_clean["name_length"] = name_lengths
    df_clean["address_length"] = address_lengths
    
    return df_clean


def load_dataset_file(filepath: Path, expected_columns: List[str]) -> pd.DataFrame:
    """Load a TSV file with pandas and validate required columns."""
    if not filepath.exists():
        raise FileNotFoundError(f"Required data file not found: {filepath}")
    
    df = pd.read_csv(filepath, sep="\t")
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"File '{filepath.name}' is missing expected columns: {missing_cols}. "
            f"Found columns: {df.columns.tolist()}"
        )
    
    return df


def load_all_data(project_root: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load all train, test, and ground truth datasets using relative paths."""
    train_dir, test_dir = find_data_paths(project_root)
    
    print(f"Loading datasets from:")
    print(f"  Train directory: {train_dir}")
    print(f"  Test directory:  {test_dir}\n")
    
    datasets = {
        "train_source1": load_dataset_file(train_dir / "train_source1.tsv", SOURCE_COLUMNS),
        "train_source2": load_dataset_file(train_dir / "train_source2.tsv", SOURCE_COLUMNS),
        "train_source3": load_dataset_file(train_dir / "train_source3.tsv", SOURCE_COLUMNS),
        "train_ground_truth": load_dataset_file(train_dir / "train_ground_truth.tsv", GROUND_TRUTH_COLUMNS),
        "test_source1": load_dataset_file(test_dir / "test_source1.tsv", SOURCE_COLUMNS),
        "test_source2": load_dataset_file(test_dir / "test_source2.tsv", SOURCE_COLUMNS),
        "test_source3": load_dataset_file(test_dir / "test_source3.tsv", SOURCE_COLUMNS),
    }
    
    return datasets


def inspect_data(df: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """Print a thorough inspection summary for a single DataFrame."""
    print(f"{'='*60}")
    print(f"DATASET INSPECTION: {dataset_name}")
    print(f"{'='*60}")
    print(f"Shape: {len(df):,} rows x {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    missing = df.isnull().sum()
    print("\nMissing Values per Column:")
    for col in df.columns:
        cnt = missing[col]
        pct = (cnt / len(df)) * 100 if len(df) > 0 else 0.0
        print(f"  - {col}: {cnt:,} ({pct:.2f}%)")
    
    if "entity_id" in df.columns:
        dup_count = df["entity_id"].duplicated().sum()
        print(f"\nDuplicate Entity IDs: {dup_count:,}")
    elif "source1_entity_id" in df.columns:
        dup_count = df["source1_entity_id"].duplicated().sum()
        print(f"\nDuplicate Source1 Entity IDs: {dup_count:,}")
    
    if "country" in df.columns:
        top_countries = df["country"].value_counts(dropna=False).head(5).to_dict()
        print(f"\nCountry Distribution (top entries): {top_countries}")
        
    print("\nSample Rows:")
    print(df.head(3))
    print("\n")


def inspect_all_data(data_dict: Dict[str, pd.DataFrame]) -> None:
    """Inspect all loaded raw datasets."""
    print("\n" + "#" * 60)
    print("BEGINNING COMPREHENSIVE DATA INSPECTION")
    print("#" * 60 + "\n")
    for name, df in data_dict.items():
        inspect_data(df, dataset_name=name)


def validate_preprocessing(
    raw_dict: Dict[str, pd.DataFrame],
    clean_dict: Dict[str, pd.DataFrame],
) -> bool:
    """Validate that preprocessing satisfied all data integrity requirements."""
    print("\n" + "=" * 60)
    print("PREPROCESSING VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    
    source_keys = [
        "train_source1",
        "train_source2",
        "train_source3",
        "test_source1",
        "test_source2",
        "test_source3",
    ]
    
    for key in source_keys:
        raw = raw_dict[key]
        clean = clean_dict[key]
        
        rows_match = len(raw) == len(clean)
        if not rows_match:
            print(f"[{key}] FAIL: Row count mismatch! Raw={len(raw)}, Clean={len(clean)}")
            all_passed = False
        
        ids_match = (raw["entity_id"].values == clean["entity_id"].values).all()
        if not ids_match:
            print(f"[{key}] FAIL: entity_id values altered!")
            all_passed = False
            
        for col in ["business_name", "business_address", "country"]:
            orig_raw = raw[col].fillna("").values
            orig_clean = clean[col].fillna("").values
            if not (orig_raw == orig_clean).all():
                print(f"[{key}] FAIL: Original column '{col}' was modified!")
                all_passed = False
                
        expected_new = [
            "business_name_clean",
            "business_address_clean",
            "country_clean",
            "name_tokens",
            "name_length",
            "address_length",
        ]
        for c in expected_new:
            if c not in clean.columns:
                print(f"[{key}] FAIL: Clean column '{c}' is missing!")
                all_passed = False
                
        raw_countries = set(raw["country"].dropna().unique())
        clean_countries = set(clean["country"].dropna().unique())
        if raw_countries != clean_countries:
            print(f"[{key}] FAIL: Country values were altered or filtered!")
            all_passed = False

    print(f"Train S1 rows: {len(clean_dict['train_source1']):,}")
    print(f"Train S2 rows: {len(clean_dict['train_source2']):,}")
    print(f"Train S3 rows: {len(clean_dict['train_source3']):,}")
    print(f"Test S1 rows:  {len(clean_dict['test_source1']):,}")
    print(f"Test S2 rows:  {len(clean_dict['test_source2']):,}")
    print(f"Test S3 rows:  {len(clean_dict['test_source3']):,}")
    print("-" * 30)
    print(f"Rows preserved:           {'PASS' if all_passed else 'FAIL'}")
    print(f"Entity IDs preserved:     {'PASS' if all_passed else 'FAIL'}")
    print(f"Original columns kept:    {'PASS' if all_passed else 'FAIL'}")
    print(f"Clean columns created:    {'PASS' if all_passed else 'FAIL'}")
    print(f"Country filtering:        NONE (Preserved all countries)")
    print("=" * 60 + "\n")
    
    return all_passed


def save_processed_data(
    clean_dict: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
) -> None:
    """Save cleaned datasets to CSV files in the processed directory."""
    if output_dir is None:
        project_root = get_project_root()
        output_dir = project_root / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed datasets to: {output_dir}\n")
    
    file_mapping = {
        "train_source1": "train_source1_clean.csv",
        "train_source2": "train_source2_clean.csv",
        "train_source3": "train_source3_clean.csv",
        "test_source1": "test_source1_clean.csv",
        "test_source2": "test_source2_clean.csv",
        "test_source3": "test_source3_clean.csv",
    }
    
    for key, filename in file_mapping.items():
        if key in clean_dict:
            target_file = output_dir / filename
            print(f"  Saving {filename} ({len(clean_dict[key]):,} rows)...", end="", flush=True)
            clean_dict[key].to_csv(target_file, index=False)
            print(" [DONE]")
            
    print(f"\nAll processed files saved successfully in: {output_dir}")


def run_pipeline() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Execute the full data loading, inspection, preprocessing, validation, and saving pipeline."""
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION — DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")
    
    project_root = get_project_root()
    print(f"Project Root: {project_root}\n")
    
    print("Step 1: Loading raw datasets...")
    raw_datasets = load_all_data(project_root)
    print("Raw datasets loaded successfully.\n")
    
    print("Step 2: Inspecting datasets...")
    inspect_all_data(raw_datasets)
    
    print("Step 3: Preprocessing source datasets...")
    source_keys = [
        "train_source1",
        "train_source2",
        "train_source3",
        "test_source1",
        "test_source2",
        "test_source3",
    ]
    
    clean_datasets: Dict[str, pd.DataFrame] = {}
    for key in source_keys:
        print(f"  Preprocessing {key} ({len(raw_datasets[key]):,} rows)...", end="", flush=True)
        clean_datasets[key] = preprocess_dataframe(raw_datasets[key])
        print(" [DONE]")
    print()
    
    print("Step 4: Validating preprocessing results...")
    validation_passed = validate_preprocessing(raw_datasets, clean_datasets)
    if not validation_passed:
        raise RuntimeError("Preprocessing validation failed! Please check logs.")
        
    print("Step 5: Saving processed data...")
    save_processed_data(clean_datasets)
    
    print("\nData Preprocessing Pipeline Complete!")
    return raw_datasets, clean_datasets


if __name__ == "__main__":
    run_pipeline()
