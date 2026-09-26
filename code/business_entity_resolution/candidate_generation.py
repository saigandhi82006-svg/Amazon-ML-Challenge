"""
Candidate Generation and Multi-Strategy Blocking Module
Generates high-recall candidate pairs between Source 1 (reference) and Source 2/3 (target).

Optimized multi-strategy blocking architecture:
1. Exact Clean Name Indexing: Instant exact matches on normalized business names.
2. Distinctive Longest Token Index: Priority lookup on distinctive business tokens.
3. Character 3-Gram Prefix Index: Captures minor spelling variations and noise.
4. Address Numeric Index: Street numbers and postal/PIN codes.
5. Country Partition Filter: Enforces country consistency.

Includes full Candidate Recall and Statistics Evaluation on training ground truth.
"""

from collections import defaultdict
import logging
import re
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .config import BLOCKING_CONFIG, get_train_ground_truth_path

logger = logging.getLogger("candidate_generation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NUMERIC_TOKEN_REGEX = re.compile(r"\b\d{2,}\b")


def extract_ngrams(text: str, n: int = 3) -> List[str]:
    """Extracts character n-grams from text."""
    if not text or len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def extract_numeric_tokens(address: str) -> List[str]:
    """Extracts numeric address components (house numbers, postal/PIN codes)."""
    if not address or pd.isna(address):
        return []
    return NUMERIC_TOKEN_REGEX.findall(str(address))


class MultiStrategyBlocker:
    """
    Ultra High-Performance Multi-Strategy Inverted Index Blocker.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or BLOCKING_CONFIG
        self.min_token_len = self.config.get("min_token_len", 3)
        self.ngram_size = self.config.get("ngram_size", 3)
        self.max_cand_per_s1 = self.config.get("max_candidates_per_s1", 25)
        self.max_postings_per_key = 200

        # Inverted indexes: key -> list of candidate entity_ids
        self.exact_name_index: Dict[str, List[str]] = defaultdict(list)
        self.token_index: Dict[str, List[str]] = defaultdict(list)
        self.ngram_index: Dict[str, List[str]] = defaultdict(list)
        self.address_num_index: Dict[str, List[str]] = defaultdict(list)
        self.country_index: Dict[str, str] = {}  # entity_id -> country_clean

    def build_index(self, target_dfs: List[pd.DataFrame]):
        """
        Builds inverted indexes over target datasets (Source 2 and Source 3).
        """
        logger.info("Building inverted indexes over target datasets (Source 2 & 3)...")

        total_indexed = 0
        for df in target_dfs:
            for entity_id, name, addr, cntry in zip(
                df["entity_id"],
                df["business_name_clean"],
                df["business_address_clean"],
                df["country_clean"],
            ):
                total_indexed += 1
                name_str = str(name) if pd.notna(name) else ""
                addr_str = str(addr) if pd.notna(addr) else ""
                cntry_str = str(cntry) if pd.notna(cntry) else ""

                if cntry_str:
                    self.country_index[entity_id] = cntry_str

                if name_str:
                    # 1. Exact name index
                    if len(self.exact_name_index[name_str]) < self.max_postings_per_key:
                        self.exact_name_index[name_str].append(entity_id)

                    # 2. Distinctive token index (sorted by length descending, top 3 tokens)
                    tokens = [t for t in name_str.split() if len(t) >= self.min_token_len]
                    tokens.sort(key=len, reverse=True)
                    for t in tokens[:3]:
                        if len(self.token_index[t]) < self.max_postings_per_key:
                            self.token_index[t].append(entity_id)

                    # 3. N-gram prefix index (first 2 ngrams)
                    if len(name_str) >= self.ngram_size:
                        p_ng = name_str[: self.ngram_size]
                        if len(self.ngram_index[p_ng]) < self.max_postings_per_key:
                            self.ngram_index[p_ng].append(entity_id)

                # 4. Address numerical index
                nums = extract_numeric_tokens(addr_str)
                for num in set(nums[:2]):
                    if len(num) >= 2:
                        key = f"{cntry_str}_{num}" if cntry_str else num
                        if len(self.address_num_index[key]) < self.max_postings_per_key:
                            self.address_num_index[key].append(entity_id)

        logger.info(
            f"Indexed {total_indexed:,} candidate records. "
            f"Exact names: {len(self.exact_name_index):,}, Tokens: {len(self.token_index):,}, "
            f"N-grams: {len(self.ngram_index):,}, Address Numbers: {len(self.address_num_index):,}."
        )

    def generate_candidates_for_s1(
        self,
        s1_entity_id: str,
        name: str,
        address: str,
        country: str,
    ) -> List[str]:
        """
        Retrieves ranked candidate IDs for a single Source 1 entity using priority-ordered blocking signals.
        """
        seen_candidates = set()
        ranked_candidates = []
        s1_country = str(country) if pd.notna(country) else ""
        name_str = str(name) if pd.notna(name) else ""
        addr_str = str(address) if pd.notna(address) else ""

        # Priority 1: Exact Name Match
        if name_str and name_str in self.exact_name_index:
            for cand_id in self.exact_name_index[name_str]:
                if cand_id not in seen_candidates:
                    cand_cntry = self.country_index.get(cand_id, "")
                    if not s1_country or not cand_cntry or s1_country == cand_cntry:
                        seen_candidates.add(cand_id)
                        ranked_candidates.append(cand_id)
                        if len(ranked_candidates) >= self.max_cand_per_s1:
                            return ranked_candidates

        # Priority 2: Distinctive Longest Tokens
        if name_str:
            tokens = [t for t in name_str.split() if len(t) >= self.min_token_len]
            tokens.sort(key=len, reverse=True)
            for t in tokens[:2]:
                if t in self.token_index:
                    for cand_id in self.token_index[t]:
                        if cand_id not in seen_candidates:
                            cand_cntry = self.country_index.get(cand_id, "")
                            if not s1_country or not cand_cntry or s1_country == cand_cntry:
                                seen_candidates.add(cand_id)
                                ranked_candidates.append(cand_id)
                                if len(ranked_candidates) >= self.max_cand_per_s1:
                                    return ranked_candidates

        # Priority 3: Character 3-gram Prefix Match (if few candidates found)
        if len(ranked_candidates) < 5 and len(name_str) >= self.ngram_size:
            p_ng = name_str[: self.ngram_size]
            if p_ng in self.ngram_index:
                for cand_id in self.ngram_index[p_ng]:
                    if cand_id not in seen_candidates:
                        cand_cntry = self.country_index.get(cand_id, "")
                        if not s1_country or not cand_cntry or s1_country == cand_cntry:
                            seen_candidates.add(cand_id)
                            ranked_candidates.append(cand_id)
                            if len(ranked_candidates) >= self.max_cand_per_s1:
                                return ranked_candidates

        # Priority 4: Address Numeric Match (House / Postal code)
        if len(ranked_candidates) < 5 and addr_str:
            nums = extract_numeric_tokens(addr_str)
            for num in nums[:2]:
                key = f"{s1_country}_{num}" if s1_country else num
                if key in self.address_num_index:
                    for cand_id in self.address_num_index[key]:
                        if cand_id not in seen_candidates:
                            cand_cntry = self.country_index.get(cand_id, "")
                            if not s1_country or not cand_cntry or s1_country == cand_cntry:
                                seen_candidates.add(cand_id)
                                ranked_candidates.append(cand_id)
                                if len(ranked_candidates) >= self.max_cand_per_s1:
                                    return ranked_candidates

        return ranked_candidates

    def generate_all_candidates(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generates candidate dictionary: s1_entity_id -> list of candidate entity_ids.
        """
        results = {}
        total = len(s1_df)

        for idx, (entity_id, name, addr, cntry) in enumerate(
            zip(
                s1_df["entity_id"],
                s1_df["business_name_clean"],
                s1_df["business_address_clean"],
                s1_df["country_clean"],
            )
        ):
            results[entity_id] = self.generate_candidates_for_s1(entity_id, name, addr, cntry)

        return results


def evaluate_candidate_recall(
    candidates_dict: Dict[str, List[str]],
    ground_truth_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Measures candidate recall, candidate counts, and distribution against ground truth.
    """
    gt_path = ground_truth_path or get_train_ground_truth_path()
    logger.info(f"Evaluating candidate recall against ground truth: {gt_path}")

    gt_df = pd.read_csv(gt_path, sep="\t")
    total_true_pairs = 0
    captured_true_pairs = 0
    candidate_counts = []
    zero_candidates_s1 = 0

    for s1_id, matched_ids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if s1_id not in candidates_dict:
            continue

        cands = set(candidates_dict[s1_id])
        candidate_counts.append(len(cands))
        if len(cands) == 0:
            zero_candidates_s1 += 1

        if pd.notna(matched_ids) and str(matched_ids).strip():
            true_ids = [m.strip() for m in str(matched_ids).split(",") if m.strip()]
            for tid in true_ids:
                total_true_pairs += 1
                if tid in cands:
                    captured_true_pairs += 1

    recall = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 0.0
    avg_cands = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    median_cands = float(np.median(candidate_counts)) if candidate_counts else 0.0
    max_cands = int(np.max(candidate_counts)) if candidate_counts else 0
    zero_pct = (zero_candidates_s1 / len(candidate_counts) * 100.0) if candidate_counts else 0.0
    total_pairs = sum(candidate_counts)

    metrics = {
        "candidate_recall": recall,
        "total_true_pairs": total_true_pairs,
        "captured_true_pairs": captured_true_pairs,
        "avg_candidates_per_s1": avg_cands,
        "median_candidates_per_s1": median_cands,
        "max_candidates_per_s1": max_cands,
        "zero_candidate_s1_percentage": zero_pct,
        "total_candidate_pairs": total_pairs,
    }

    logger.info("=" * 60)
    logger.info("CANDIDATE GENERATION RECALL REPORT")
    logger.info("=" * 60)
    logger.info(f"Candidate Recall:             {recall:.4%} ({captured_true_pairs:,}/{total_true_pairs:,})")
    logger.info(f"Avg Candidates per S1:        {avg_cands:.2f}")
    logger.info(f"Median Candidates per S1:     {median_cands:.1f}")
    logger.info(f"Max Candidates per S1:        {max_cands}")
    logger.info(f"Zero Candidate S1 (%):        {zero_pct:.2f}%")
    logger.info(f"Total Candidate Pairs:        {total_pairs:,}")
    logger.info("=" * 60)

    return metrics


def save_candidate_pairs_tsv(candidates_dict: Dict[str, List[str]], output_path: Path):
    """
    Saves candidate pairs to TSV conforming to the required format:
    source1_entity_id\tcandidate_entity_ids
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving candidate pairs to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidates_dict.items():
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")
    logger.info(f"Candidate pairs saved successfully to {output_path}")
