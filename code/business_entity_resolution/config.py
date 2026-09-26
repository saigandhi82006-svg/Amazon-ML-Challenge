"""
Configuration module for the Business Entity Resolution Pipeline.
Defines paths, blocking parameters, model hyperparameters, and AWS settings.
"""

from pathlib import Path
import os

# Base directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
    PROJECT_ROOT / "train" / "train_ground_truth.tsv",
    PROJECT_ROOT / "train_" / "train_ground_truth.tsv",
    PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv",
]

def get_train_ground_truth_path() -> Path:
    for path in RAW_TRAIN_GROUND_TRUTH_CANDIDATES:
        if path.exists():
            return path
    # Default fallback
    return PROJECT_ROOT / "train" / "train_ground_truth.tsv"

# Output submission paths
CANDIDATE_OUTPUT_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_OUTPUT_PATH = OUTPUT_DIR / "matching_results.tsv"

# Candidate generation / Blocking parameters
BLOCKING_CONFIG = {
    "max_token_frequency": 50000,      # Ignore tokens occurring more than 50k times to avoid explosions
    "min_token_len": 3,                # Minimum character length for tokens
    "ngram_size": 3,                   # Character n-gram size
    "max_ngram_frequency": 80000,      # Max n-gram frequency filter
    "max_candidates_per_s1": 40,       # Cap candidates per S1 to ensure fast & scalable evaluation
    "min_numeric_len": 2,              # Minimum length for address numerical tokens (house/pin)
}

# AWS Bedrock / Amazon AI Settings
AWS_CONFIG = {
    "model_id": os.getenv("AWS_BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0"),
    "region_name": os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1")),
    "embedding_dimension": 256,       # Titan v2 supports 256, 512, 1024 (256 is fast & compact)
    "batch_size": 25,                 # Max batch size for Titan Embeddings
    "max_retries": 3,
    "cache_file": CACHE_DIR / "bedrock_embeddings.pkl",
    "enabled": os.getenv("ENABLE_AMAZON_BEDROCK", "1") == "1",
}

# Machine Learning Model Parameters
MODEL_CONFIG = {
    "model_type": "lightgbm",          # 'lightgbm' or 'hist_gradient_boosting'
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
    "train_val_split_ratio": 0.8,     # 80% train, 20% validation split by S1 entity
    "beta": 0.5,                      # Macro F0.5 metric weighting (precision prioritized)
    "threshold_grid_start": 0.10,
    "threshold_grid_end": 0.95,
    "threshold_grid_step": 0.02,
    "default_threshold": 0.65,        # Conservative initial default threshold
}
