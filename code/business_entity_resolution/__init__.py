"""
Business Entity Resolution Package
Modules:
- preprocessing: Member 1 Data Cleaning & Normalization
- candidate_generation: Member 2 Multi-Strategy Blocking & Recall Evaluation
- features: Member 3 Similarity Feature Engineering
- amazon_ai: AWS Bedrock Titan Embeddings & Semantic Representation
- matching_model: Member 3 GBDT Pair Matcher
- evaluate: Member 4 Macro F0.5 Metric & Threshold Optimization
- config: Central Configuration & Parameters
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
