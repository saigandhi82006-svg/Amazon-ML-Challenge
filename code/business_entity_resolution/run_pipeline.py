"""
Master Pipeline Orchestrator — Business Entity Resolution Challenge 2026
Executes end-to-end workflow: Preprocessing -> Candidates -> Features -> Training ->
Threshold Optimization (Macro F0.5) -> Test Inference -> Submission Output -> Validation.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output encoding for terminal/logs
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .config import (
    PROJECT_ROOT,
    PROCESSED_FILES,
    OUTPUT_DIR,
    EXPERIMENTS_DIR,
    CANDIDATE_OUTPUT_PATH,
    MATCHING_OUTPUT_PATH,
    MODEL_CONFIG,
    EVAL_CONFIG,
    BLOCKING_CONFIG,
    get_train_ground_truth_path,
)
from .candidate_generation import (
    MultiStrategyBlocker,
    evaluate_candidate_recall,
)
from .features import compute_pair_features, FEATURE_COLUMNS
from .matching_model import EntityMatchingClassifier
from .evaluate import optimize_threshold, evaluate_predictions

logger = logging.getLogger("run_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_training_and_validation(sample_size: int = 40000) -> Tuple[EntityMatchingClassifier, float, dict]:
    """
    Executes candidate blocking, feature extraction, model fitting, and threshold optimization on training data.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 & 2: CANDIDATE GENERATION & TRAINING PREPARATION")
    logger.info("=" * 60)

    # Load training datasets
    train_s1 = pd.read_csv(PROCESSED_FILES["train_source1"], keep_default_na=False, dtype=str)
    train_s2 = pd.read_csv(PROCESSED_FILES["train_source2"], keep_default_na=False, dtype=str)
    train_s3 = pd.read_csv(PROCESSED_FILES["train_source3"], keep_default_na=False, dtype=str)
    gt_df = pd.read_csv(get_train_ground_truth_path(), sep="\t", keep_default_na=False, dtype=str)

    # Ground truth mapping: s1_id -> set of true matched entity IDs
    gt_map: Dict[str, Set[str]] = {}
    for s1_id, m_ids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if m_ids:
            gt_map[s1_id] = {m.strip() for m in str(m_ids).split(",") if m.strip()}
        else:
            gt_map[s1_id] = set()

    # Fast indexed lookup dictionaries for target records
    s23_dict = {}
    for df in [train_s2, train_s3]:
        for eid, name, addr, cntry in zip(
            df["entity_id"], df["business_name_clean"], df["business_address_clean"], df["country_clean"]
        ):
            s23_dict[eid] = (name, addr, cntry)

    # Select representative training sample for efficient feature extraction & learning
    if sample_size and len(train_s1) > sample_size:
        logger.info(f"Sampling {sample_size:,} S1 records for training & validation split...")
        sample_s1_ids = train_s1["entity_id"].sample(n=sample_size, random_state=42).tolist()
        s1_sample_df = train_s1[train_s1["entity_id"].isin(sample_s1_ids)].copy()
    else:
        s1_sample_df = train_s1.copy()

    # Build blocker and generate candidates
    blocker = MultiStrategyBlocker()
    blocker.build_index([train_s2, train_s3])
    candidates = blocker.generate_all_candidates(s1_sample_df)

    # Measure candidate recall
    recall_metrics = evaluate_candidate_recall(candidates)

    # Entity-level Grouped Train / Validation Split (80% Train, 20% Validation)
    unique_s1_ids = list(candidates.keys())
    train_s1_ids, val_s1_ids = train_test_split(unique_s1_ids, test_size=0.20, random_state=42)
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    logger.info(f"Entity-level Split: {len(train_s1_ids):,} S1 in Train, {len(val_s1_ids):,} S1 in Validation.")

    # Construct training pairs with positives and hard negatives
    s1_lookup = {
        eid: (name, addr, cntry)
        for eid, name, addr, cntry in zip(
            s1_sample_df["entity_id"],
            s1_sample_df["business_name_clean"],
            s1_sample_df["business_address_clean"],
            s1_sample_df["country_clean"],
        )
    }

    def build_dataset_pairs(target_s1_set: Set[str], max_neg_per_s1: int = 8):
        s1_n, s1_a, s1_c = [], [], []
        c_ids, c_n, c_a, c_c = [], [], [], []
        labels = []
        pair_s1_keys = []

        for s1_id in target_s1_set:
            name1, addr1, cntry1 = s1_lookup[s1_id]
            true_matches = gt_map.get(s1_id, set())
            cands = candidates.get(s1_id, [])

            # Add true positive matches
            for true_id in true_matches:
                if true_id in s23_dict:
                    t_name, t_addr, t_cntry = s23_dict[true_id]
                    s1_n.append(name1)
                    s1_a.append(addr1)
                    s1_c.append(cntry1)
                    c_ids.append(true_id)
                    c_n.append(t_name)
                    c_a.append(t_addr)
                    c_c.append(t_cntry)
                    labels.append(1)
                    pair_s1_keys.append(s1_id)

            # Add hard negative candidates (top candidates from blocking not in ground truth)
            neg_count = 0
            for cid in cands:
                if cid not in true_matches and cid in s23_dict:
                    t_name, t_addr, t_cntry = s23_dict[cid]
                    s1_n.append(name1)
                    s1_a.append(addr1)
                    s1_c.append(cntry1)
                    c_ids.append(cid)
                    c_n.append(t_name)
                    c_a.append(t_addr)
                    c_c.append(t_cntry)
                    labels.append(0)
                    pair_s1_keys.append(s1_id)
                    neg_count += 1
                    if neg_count >= max_neg_per_s1:
                        break

        return s1_n, s1_a, s1_c, c_ids, c_n, c_a, c_c, np.array(labels, dtype=np.int32), pair_s1_keys

    logger.info("Extracting features for training pairs...")
    tr_s1_n, tr_s1_a, tr_s1_c, tr_c_ids, tr_c_n, tr_c_a, tr_c_c, y_train, _ = build_dataset_pairs(train_s1_set)
    X_train = compute_pair_features(tr_s1_n, tr_s1_a, tr_s1_c, tr_c_ids, tr_c_n, tr_c_a, tr_c_c)

    # Train matching classifier
    classifier = EntityMatchingClassifier()
    classifier.fit(X_train, y_train)
    classifier.save()

    # Validation & Threshold Optimization
    logger.info("Extracting features for validation pairs...")
    v_s1_n, v_s1_a, v_s1_c, v_c_ids, v_c_n, v_c_a, v_c_c, y_val, v_s1_keys = build_dataset_pairs(val_s1_set, max_neg_per_s1=15)
    X_val = compute_pair_features(v_s1_n, v_s1_a, v_s1_c, v_c_ids, v_c_n, v_c_a, v_c_c)
    val_probs = classifier.predict_proba(X_val)

    # Group candidate scores by S1 entity
    val_candidate_scores: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in val_s1_set}
    for s1_id, cid, prob in zip(v_s1_keys, v_c_ids, val_probs):
        val_candidate_scores[s1_id].append((cid, float(prob)))

    val_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_set}
    optimal_threshold, best_metrics, history = optimize_threshold(val_candidate_scores, val_gt)

    # Save experiment tracking log
    exp_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": classifier.model_type,
        "optimal_threshold": optimal_threshold,
        "validation_metrics": best_metrics,
        "candidate_recall_metrics": recall_metrics,
        "features": FEATURE_COLUMNS,
    }
    with open(EXPERIMENTS_DIR / "experiment_log.json", "w", encoding="utf-8") as f:
        json.dump(exp_log, f, indent=2)

    return classifier, optimal_threshold, exp_log


def run_test_inference(
    classifier: EntityMatchingClassifier,
    optimal_threshold: float,
    batch_size: int = 150000,
):
    """
    Executes scalable streaming test inference and directly writes output TSV files.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: TEST INFERENCE AND SUBMISSION PACKAGING")
    logger.info("=" * 60)

    # Load test datasets
    test_s1 = pd.read_csv(PROCESSED_FILES["test_source1"], keep_default_na=False, dtype=str)
    test_s2 = pd.read_csv(PROCESSED_FILES["test_source2"], keep_default_na=False, dtype=str)
    test_s3 = pd.read_csv(PROCESSED_FILES["test_source3"], keep_default_na=False, dtype=str)

    # Build blocker over test Source 2 and 3
    blocker = MultiStrategyBlocker()
    blocker.build_index([test_s2, test_s3])

    # Lookup dictionary for fast target feature extraction
    test_s23_dict = {}
    for df in [test_s2, test_s3]:
        for eid, name, addr, cntry in zip(
            df["entity_id"], df["business_name_clean"], df["business_address_clean"], df["country_clean"]
        ):
            test_s23_dict[eid] = (name, addr, cntry)

    total_s1 = len(test_s1)
    logger.info(f"Generating streaming test predictions across {total_s1:,} Source 1 entities...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CANDIDATE_OUTPUT_PATH, "w", encoding="utf-8", buffering=1024*1024) as f_cand, open(
        MATCHING_OUTPUT_PATH, "w", encoding="utf-8", buffering=1024*1024
    ) as f_match:
        # Write exact required headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for start_idx in range(0, total_s1, batch_size):
            end_idx = min(start_idx + batch_size, total_s1)
            batch_df = test_s1.iloc[start_idx:end_idx]
            logger.info(f"  Processing test batch {start_idx:,} to {end_idx:,} ({len(batch_df):,} S1 records)...")

            # 1. Candidate generation
            batch_candidates = blocker.generate_all_candidates(batch_df)

            # 2. Prepare pairs for batch feature computation (top 8 candidates per S1)
            s1_n, s1_a, s1_c = [], [], []
            c_ids, c_n, c_a, c_c = [], [], [], []
            pair_s1_keys = []

            for eid, name, addr, cntry in zip(
                batch_df["entity_id"],
                batch_df["business_name_clean"],
                batch_df["business_address_clean"],
                batch_df["country_clean"],
            ):
                cands = batch_candidates.get(eid, [])

                for cid in cands[:8]:
                    if cid in test_s23_dict:
                        t_name, t_addr, t_cntry = test_s23_dict[cid]
                        s1_n.append(name)
                        s1_a.append(addr)
                        s1_c.append(cntry)
                        c_ids.append(cid)
                        c_n.append(t_name)
                        c_a.append(t_addr)
                        c_c.append(t_cntry)
                        pair_s1_keys.append(eid)

            # 3. Compute features & predict probabilities
            matched_by_s1: Dict[str, List[str]] = {eid: [] for eid in batch_df["entity_id"]}

            if pair_s1_keys:
                X_batch = compute_pair_features(s1_n, s1_a, s1_c, c_ids, c_n, c_a, c_c)
                probs = classifier.predict_proba(X_batch)

                for s1_id, cid, prob in zip(pair_s1_keys, c_ids, probs):
                    if prob >= optimal_threshold:
                        if cid not in matched_by_s1[s1_id]:
                            matched_by_s1[s1_id].append(cid)

            # 4. Stream write to files
            for s1_id in batch_df["entity_id"]:
                cands = batch_candidates.get(s1_id, [])
                matches = matched_by_s1.get(s1_id, [])
                c_str = ",".join(cands) if cands else ""
                m_str = ",".join(matches) if matches else ""
                f_cand.write(f"{s1_id}\t{c_str}\n")
                f_match.write(f"{s1_id}\t{m_str}\n")

            f_cand.flush()
            f_match.flush()

    logger.info(f"Candidate pairs saved successfully to {CANDIDATE_OUTPUT_PATH}")
    logger.info(f"Matching results saved successfully to {MATCHING_OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["preprocess", "candidates", "train", "evaluate", "predict", "validate", "all"],
        help="Pipeline stage to execute.",
    )
    args = parser.parse_args()

    logger.info(f"Starting pipeline stage: {args.stage}")

    if args.stage == "preprocess":
        from .preprocessing import main as prep_main
        prep_main()

    elif args.stage in ["train", "all"]:
        classifier, threshold, exp_log = run_training_and_validation()
        if args.stage == "all":
            run_test_inference(classifier, threshold)
            from utils.validate_submission import validate_submission
            validate_submission(
                MATCHING_OUTPUT_PATH,
                CANDIDATE_OUTPUT_PATH,
                PROCESSED_FILES["test_source1"],
            )

    elif args.stage == "predict":
        classifier = EntityMatchingClassifier()
        classifier.load()
        thresh = EVAL_CONFIG["default_threshold"]
        exp_file = EXPERIMENTS_DIR / "experiment_log.json"
        if exp_file.exists():
            with open(exp_file, "r") as f:
                exp_data = json.load(f)
                thresh = exp_data.get("optimal_threshold", thresh)
        run_test_inference(classifier, thresh)

    elif args.stage == "validate":
        from utils.validate_submission import validate_submission
        validate_submission(
            MATCHING_OUTPUT_PATH,
            CANDIDATE_OUTPUT_PATH,
            PROCESSED_FILES["test_source1"],
        )


if __name__ == "__main__":
    main()
