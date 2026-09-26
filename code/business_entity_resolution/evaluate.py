"""
Evaluation and Threshold Optimization Module
Calculates official competition metric (Macro F0.5 per Source 1 entity)
and finds the optimal decision threshold.

F0.5 Metric Definition:
F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Precision is prioritized over Recall (beta = 0.5).
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd

from .config import EVAL_CONFIG

logger = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def calculate_entity_f_beta(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F_beta for a single Source 1 entity.
    Handles singletons (zero-match entities) strictly.
    """
    # Case 1: Ground truth is empty (singleton entity)
    if not true_matches:
        if not pred_matches:
            # Correct singleton prediction
            return 1.0, 1.0, 1.0
        else:
            # False positives on a singleton entity
            return 0.0, 0.0, 0.0

    # Case 2: Predicted matches is empty for a non-singleton entity
    if not pred_matches:
        return 0.0, 0.0, 0.0

    # Case 3: Both true and predicted matches exist
    tp = len(true_matches & pred_matches)
    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    if precision == 0.0 and recall == 0.0:
        return 0.0, 0.0, 0.0

    beta_sq = beta ** 2
    f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
    return precision, recall, f_beta


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluates predictions across all S1 entities and computes Macro-averaged metrics.
    """
    precisions = []
    recalls = []
    f_betas = []
    total_preds = 0
    zero_pred_s1 = 0
    singleton_s1_count = 0
    correct_singleton_count = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_preds += len(pred_set)
        if not pred_set:
            zero_pred_s1 += 1

        if not true_set:
            singleton_s1_count += 1
            if not pred_set:
                correct_singleton_count += 1

        p, r, f = calculate_entity_f_beta(true_set, pred_set, beta=beta)
        precisions.append(p)
        recalls.append(r)
        f_betas.append(f)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f_beta = float(np.mean(f_betas)) if f_betas else 0.0

    return {
        f"macro_f{beta}": macro_f_beta,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "total_predicted_matches": total_preds,
        "zero_prediction_s1_count": zero_pred_s1,
        "singleton_s1_count": singleton_s1_count,
        "correct_singleton_count": correct_singleton_count,
        "total_evaluated_s1": len(ground_truth),
    }


def optimize_threshold(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    grid_start: float = EVAL_CONFIG["threshold_grid_start"],
    grid_end: float = EVAL_CONFIG["threshold_grid_end"],
    grid_step: float = EVAL_CONFIG["threshold_grid_step"],
    beta: float = EVAL_CONFIG["beta"],
) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    """
    Finds the optimal decision threshold that maximizes validation Macro F0.5.
    """
    logger.info(f"Optimizing decision threshold over grid [{grid_start:.2f}, {grid_end:.2f}]...")
    best_threshold = 0.5
    best_f = -1.0
    best_metrics = {}
    history = []

    thresholds = np.arange(grid_start, grid_end + 1e-5, grid_step)

    for thresh in thresholds:
        thresh_val = round(float(thresh), 3)
        # Apply threshold to candidate scores
        preds = {}
        for s1_id in ground_truth.keys():
            cand_list = candidate_scores.get(s1_id, [])
            preds[s1_id] = {cid for cid, score in cand_list if score >= thresh_val}

        metrics = evaluate_predictions(ground_truth, preds, beta=beta)
        metrics["threshold"] = thresh_val
        history.append(metrics)

        if metrics[f"macro_f{beta}"] > best_f:
            best_f = metrics[f"macro_f{beta}"]
            best_threshold = thresh_val
            best_metrics = metrics

    logger.info("=" * 60)
    logger.info(f"OPTIMAL THRESHOLD FOUND: {best_threshold:.3f}")
    logger.info(f"Validation Macro F0.5:   {best_metrics['macro_f0.5']:.4f}")
    logger.info(f"Validation Macro Prec:   {best_metrics['macro_precision']:.4f}")
    logger.info(f"Validation Macro Recall: {best_metrics['macro_recall']:.4f}")
    logger.info(f"Total Matches Predicted: {best_metrics['total_predicted_matches']:,}")
    logger.info("=" * 60)

    return best_threshold, best_metrics, history
