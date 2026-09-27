"""
Configuration module for the Business Entity Resolution Pipeline.
Defines paths, blocking parameters, model hyperparameters, and AWS settings.
"""

from pathlib import Path
import os


def find_project_root() -> Path:
    """Finds project root by locating known directories or markers."""
    curr = Path(__file__).resolve().parent
    for _ in range(5):
        if (
            (curr / "dataset").exists()
            or (curr / "processed").exists()
            or (curr / "output").exists()
            or (curr / "Documentation_template.md").exists()
        ):
            return curr
        if curr.parent == curr:
            break
        curr = curr.parent
    return Path(__file__).resolve().parent.parent.parent.parent


# Base directory paths
PROJECT_ROOT = find_project_root()
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "amazon_ai_cache"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# File names
PROCESSED_FILES = {
    "train_source1": PROCESSED_DIR / "train_source1_clean.csv",
    "train_source2": PROCESSED_DIR / "train_source2_clean.csv",
    "train_source3": PROCESSED_DIR / "train_source3_clean.csv",
    "test_source1": PROCESSED_DIR / "test_source1_clean.csv",
    "test_source2": PROCESSED_DIR / "test_source2_clean.csv",
    "test_source3": PROCESSED_DIR / "test_source3_clean.csv",
}

RAW_TRAIN_GROUND_TRUTH_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv",
    PROJECT_ROOT / "train" / "train_ground_truth.tsv",
    PROJECT_ROOT / "train_" / "train_ground_truth.tsv",
]


def get_train_ground_truth_path() -> Path:
    for path in RAW_TRAIN_GROUND_TRUTH_CANDIDATES:
        if path.exists():
            return path
    return PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"


# Output submission paths
CANDIDATE_OUTPUT_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_OUTPUT_PATH = OUTPUT_DIR / "matching_results.tsv"

# Candidate generation / Blocking parameters
BLOCKING_CONFIG = {
    "max_token_frequency": 50000,
    "min_token_len": 3,
    "ngram_size": 3,
    "max_ngram_frequency": 80000,
    "max_candidates_per_s1": 40,
    "min_numeric_len": 2,
}

# AWS Bedrock / Amazon AI Settings
AWS_CONFIG = {
    "model_id": os.getenv("AWS_BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0"),
    "region_name": os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1")),
    "embedding_dimension": 256,
    "batch_size": 25,
    "max_retries": 3,
    "cache_file": CACHE_DIR / "bedrock_embeddings.pkl",
    "enabled": os.getenv("ENABLE_AMAZON_BEDROCK", "1") == "1",
}

# Machine Learning Model Parameters
MODEL_CONFIG = {
    "model_type": "lightgbm",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 7,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "model_save_path": EXPERIMENTS_DIR / "matching_model.joblib",
}

# Evaluation & Threshold Optimization
EVAL_CONFIG = {
    "train_val_split_ratio": 0.8,
    "beta": 0.5,
    "threshold_grid_start": 0.10,
    "threshold_grid_end": 0.95,
    "threshold_grid_step": 0.02,
    "default_threshold": 0.65,
}
