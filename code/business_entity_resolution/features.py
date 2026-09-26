"""
Feature Engineering Module — Member 3
Extracts multi-faceted similarity features for candidate entity pairs.

Feature Groups:
1. Business Name Similarities: Exact match, Token Jaccard, Token Dice, Token Overlap,
   Character 3-Gram Jaccard, Prefix 3/5 Match, Length Difference/Ratio, Character Overlap.
2. Address Similarities: Token Jaccard, Token Overlap, Address Character 3-Gram Jaccard,
   Numeric Token Exact Match, Numeric Token Jaccard (House/PIN numbers), Length Diff.
3. Country Consistency: Exact Country Match, Country Missing Indicators.
4. Source Attribution: Is Candidate Source 2 vs Source 3.
5. Amazon AI Semantic Feature: Vectorized Cosine similarity from Amazon Bedrock Titan Embeddings.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

from .amazon_ai import generate_semantic_features

logger = logging.getLogger("features")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NUMERIC_RE = re.compile(r"\b\d{2,}\b")

FEATURE_COLUMNS = [
    "exact_name_match",
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_dice",
    "name_char_3gram_jaccard",
    "name_length_diff",
    "name_length_ratio",
    "name_prefix_match_3",
    "name_prefix_match_5",
    "name_char_overlap_ratio",
    "address_token_jaccard",
    "address_token_overlap",
    "address_char_3gram_jaccard",
    "address_numeric_match",
    "address_numeric_jaccard",
    "address_length_diff",
    "country_exact_match",
    "country_missing_either",
    "is_source2",
    "amazon_semantic_similarity",
]


def fast_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection / union) if union > 0 else 0.0


def fast_dice(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Dice similarity coefficient between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    total = len(tokens1) + len(tokens2)
    return float(2.0 * intersection / total) if total > 0 else 0.0


def get_char_ngrams_set(text: str, n: int = 3) -> Set[str]:
    """Generates set of character n-grams."""
    if not text or len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def compute_pair_features(
    s1_names: List[str],
    s1_addresses: List[str],
    s1_countries: List[str],
    cand_ids: List[str],
    cand_names: List[str],
    cand_addresses: List[str],
    cand_countries: List[str],
    include_semantic: bool = True,
) -> pd.DataFrame:
    """
    Computes dense similarity feature matrix for candidate pairs efficiently.
    """
    num_pairs = len(s1_names)
    matrix = np.zeros((num_pairs, len(FEATURE_COLUMNS)), dtype=np.float32)

    for i in range(num_pairs):
        n1 = str(s1_names[i] or "").strip()
        n2 = str(cand_names[i] or "").strip()
        a1 = str(s1_addresses[i] or "").strip()
        a2 = str(cand_addresses[i] or "").strip()
        c1 = str(s1_countries[i] or "").strip()
        c2 = str(cand_countries[i] or "").strip()
        cid = str(cand_ids[i])

        # Name Token & Set representations
        t1 = set(n1.split()) if n1 else set()
        t2 = set(n2.split()) if n2 else set()
        ng1 = get_char_ngrams_set(n1, 3)
        ng2 = get_char_ngrams_set(n2, 3)

        # 1. Name features
        exact_match = 1.0 if (n1 and n2 and n1 == n2) else 0.0
        tok_jaccard = fast_jaccard(t1, t2)
        tok_overlap = float(len(t1 & t2))
        tok_dice = fast_dice(t1, t2)
        char_jaccard = fast_jaccard(ng1, ng2)
        len1, len2 = len(n1), len(n2)
        len_diff = float(abs(len1 - len2))
        len_ratio = float(min(len1, len2) / max(len1, len2, 1))
        prefix_3 = 1.0 if (len1 >= 3 and len2 >= 3 and n1[:3] == n2[:3]) else 0.0
        prefix_5 = 1.0 if (len1 >= 5 and len2 >= 5 and n1[:5] == n2[:5]) else 0.0

        # Character set overlap
        s1_chars = set(n1)
        s2_chars = set(n2)
        char_overlap = float(len(s1_chars & s2_chars) / max(len(s1_chars | s2_chars), 1))

        # Address Token & Set representations
        at1 = set(a1.split()) if a1 else set()
        at2 = set(a2.split()) if a2 else set()
        ang1 = get_char_ngrams_set(a1, 3)
        ang2 = get_char_ngrams_set(a2, 3)
        num1 = set(NUMERIC_RE.findall(a1)) if a1 else set()
        num2 = set(NUMERIC_RE.findall(a2)) if a2 else set()

        # 2. Address features
        addr_tok_jaccard = fast_jaccard(at1, at2)
        addr_tok_overlap = float(len(at1 & at2))
        addr_char_jaccard = fast_jaccard(ang1, ang2)
        addr_num_match = 1.0 if (num1 and num2 and (num1 & num2)) else 0.0
        addr_num_jaccard = fast_jaccard(num1, num2)
        addr_len_diff = float(abs(len(a1) - len(a2)))

        # 3. Country features
        country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
        country_missing = 1.0 if (not c1 or not c2) else 0.0

        # 4. Source feature
        is_s2 = 1.0 if cid.startswith("S2") else 0.0

        matrix[i, 0] = exact_match
        matrix[i, 1] = tok_jaccard
        matrix[i, 2] = tok_overlap
        matrix[i, 3] = tok_dice
        matrix[i, 4] = char_jaccard
        matrix[i, 5] = len_diff
        matrix[i, 6] = len_ratio
        matrix[i, 7] = prefix_3
        matrix[i, 8] = prefix_5
        matrix[i, 9] = char_overlap
        matrix[i, 10] = addr_tok_jaccard
        matrix[i, 11] = addr_tok_overlap
        matrix[i, 12] = addr_char_jaccard
        matrix[i, 13] = addr_num_match
        matrix[i, 14] = addr_num_jaccard
        matrix[i, 15] = addr_len_diff
        matrix[i, 16] = country_match
        matrix[i, 17] = country_missing
        matrix[i, 18] = is_s2

    df_feats = pd.DataFrame(matrix, columns=FEATURE_COLUMNS)

    # 5. Semantic similarity from Amazon Bedrock
    if include_semantic:
        try:
            semantic_sims = generate_semantic_features(
                s1_names, s1_addresses, s1_countries, cand_names, cand_addresses, cand_countries
            )
            df_feats["amazon_semantic_similarity"] = semantic_sims
        except Exception as e:
            logger.warning(f"Error computing semantic features: {e}. Defaulting to 0.")
            df_feats["amazon_semantic_similarity"] = 0.0
    else:
        df_feats["amazon_semantic_similarity"] = 0.0

    return df_feats
