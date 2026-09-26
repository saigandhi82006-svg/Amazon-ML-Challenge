"""
Business Entity Resolution Package

Modules:
- preprocessing: Data Cleaning, Tokenization & Legal Suffix Normalization
- candidate_generation: Multi-Strategy Inverted Index Blocking & Candidate Filtering
- features: Pairwise String, Token & Semantic Similarity Feature Engineering
- amazon_ai: Amazon Bedrock Titan Text Embeddings & Dense Semantic Cosine Vectors
- matching_model: Gradient Boosted Decision Tree (LightGBM) Pair Matcher
- evaluate: Macro F0.5 Metric Computation & Threshold Optimization
- config: Centralized Project Settings & Hyperparameters
"""

from .config import (
    BLOCKING_CONFIG,
    MODEL_CONFIG,
    EVAL_CONFIG,
    AWS_CONFIG,
    PROCESSED_FILES,
    CANDIDATE_OUTPUT_PATH,
    MATCHING_OUTPUT_PATH,
)
from .candidate_generation import MultiStrategyBlocker, evaluate_candidate_recall
from .features import compute_pair_features, FEATURE_COLUMNS
from .amazon_ai import AmazonBedrockEmbeddingClient, generate_semantic_features
from .matching_model import EntityMatchingClassifier
from .evaluate import evaluate_predictions, optimize_threshold, calculate_entity_f_beta

__all__ = [
    "MultiStrategyBlocker",
    "evaluate_candidate_recall",
    "compute_pair_features",
    "FEATURE_COLUMNS",
    "AmazonBedrockEmbeddingClient",
    "generate_semantic_features",
    "EntityMatchingClassifier",
    "evaluate_predictions",
    "optimize_threshold",
    "calculate_entity_f_beta",
    "BLOCKING_CONFIG",
    "MODEL_CONFIG",
    "EVAL_CONFIG",
    "AWS_CONFIG",
    "PROCESSED_FILES",
    "CANDIDATE_OUTPUT_PATH",
    "MATCHING_OUTPUT_PATH",
]
