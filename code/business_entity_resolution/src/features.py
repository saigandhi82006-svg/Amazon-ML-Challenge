"""
High-Precision Feature Engineering Module
Extracts 24 multi-faceted similarity signals for candidate entity pairs.

Signal Categories:
1. Name Lexical & Syntactic Similarities (Exact match, First word match, Prefix matches)
2. Name Token-Set Metrics (Jaccard, Dice, Overlap Coefficient, Token containment)
3. Name Character N-Gram Metrics (3-Gram Jaccard, 4-Gram Jaccard, Character overlap)
4. Address Lexical & Numeric Metrics (Token Jaccard, Overlap, Street Number/Postal Code match)
5. Spatial & Structural Metrics (Address length diff, Numeric token Jaccard)
6. Metadata & Geography (Country exact match, Country missing indicator, Target source indicator)
7. Amazon Bedrock Titan Text Embeddings V2 (1024-dim Vector Cosine Similarity)
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
    "first_word_match",
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_dice",
    "name_char_3gram_jaccard",
    "name_char_4gram_jaccard",
    "name_length_diff",
    "name_length_ratio",
    "name_prefix_match_3",
    "name_prefix_match_5",
    "name_char_overlap_ratio",
    "name_token_containment",
    "address_token_jaccard",
    "address_token_overlap",
    "address_char_3gram_jaccard",
    "address_numeric_match",
    "address_numeric_jaccard",
    "address_length_diff",
    "country_exact_match",
    "country_missing_either",
    "is_source2",
    "is_source3",
    "amazon_semantic_similarity",
]


def fast_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection / union) if union > 0 else 0.0


def fast_overlap(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Overlap Coefficient (containment) between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    min_len = min(len(tokens1), len(tokens2))
    return float(intersection / min_len) if min_len > 0 else 0.0


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
    Computes 24 dense pairwise similarity features for candidate pairs.
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
        w1 = n1.split() if n1 else []
        w2 = n2.split() if n2 else []
        t1 = set(w1)
        t2 = set(w2)
        ng3_1 = get_char_ngrams_set(n1, 3)
        ng3_2 = get_char_ngrams_set(n2, 3)
        ng4_1 = get_char_ngrams_set(n1, 4)
        ng4_2 = get_char_ngrams_set(n2, 4)

        # 1. Name features
        exact_match = 1.0 if (n1 and n2 and n1 == n2) else 0.0
        first_w_match = 1.0 if (w1 and w2 and w1[0] == w2[0]) else 0.0
        tok_jaccard = fast_jaccard(t1, t2)
        tok_overlap = fast_overlap(t1, t2)
        tok_dice = fast_dice(t1, t2)
        char_3g_jaccard = fast_jaccard(ng3_1, ng3_2)
        char_4g_jaccard = fast_jaccard(ng4_1, ng4_2)
        len1, len2 = len(n1), len(n2)
        len_diff = float(abs(len1 - len2))
        len_ratio = float(min(len1, len2) / max(len1, len2, 1))
        prefix_3 = 1.0 if (len1 >= 3 and len2 >= 3 and n1[:3] == n2[:3]) else 0.0
        prefix_5 = 1.0 if (len1 >= 5 and len2 >= 5 and n1[:5] == n2[:5]) else 0.0

        # Character set overlap
        s1_chars = set(n1)
        s2_chars = set(n2)
        char_overlap = float(len(s1_chars & s2_chars) / max(len(s1_chars | s2_chars), 1))
        tok_containment = 1.0 if (t1 and (t1.issubset(t2) or t2.issubset(t1))) else 0.0

        # Address Token & Set representations
        at1 = set(a1.split()) if a1 else set()
        at2 = set(a2.split()) if a2 else set()
        ang1 = get_char_ngrams_set(a1, 3)
        ang2 = get_char_ngrams_set(a2, 3)
        num1 = set(NUMERIC_RE.findall(a1)) if a1 else set()
        num2 = set(NUMERIC_RE.findall(a2)) if a2 else set()

        # 2. Address features
        addr_tok_jaccard = fast_jaccard(at1, at2)
        addr_tok_overlap = fast_overlap(at1, at2)
        addr_char_jaccard = fast_jaccard(ang1, ang2)
        addr_num_match = 1.0 if (num1 and num2 and (num1 & num2)) else 0.0
        addr_num_jaccard = fast_jaccard(num1, num2)
        addr_len_diff = float(abs(len(a1) - len(a2)))

        # 3. Country features
        country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
        country_missing = 1.0 if (not c1 or not c2) else 0.0

        # 4. Source features
        is_s2 = 1.0 if cid.startswith("S2") else 0.0
        is_s3 = 1.0 if cid.startswith("S3") else 0.0

        matrix[i, 0] = exact_match
        matrix[i, 1] = first_w_match
        matrix[i, 2] = tok_jaccard
        matrix[i, 3] = tok_overlap
        matrix[i, 4] = tok_dice
        matrix[i, 5] = char_3g_jaccard
        matrix[i, 6] = char_4g_jaccard
        matrix[i, 7] = len_diff
        matrix[i, 8] = len_ratio
        matrix[i, 9] = prefix_3
        matrix[i, 10] = prefix_5
        matrix[i, 11] = char_overlap
        matrix[i, 12] = tok_containment
        matrix[i, 13] = addr_tok_jaccard
        matrix[i, 14] = addr_tok_overlap
        matrix[i, 15] = addr_char_jaccard
        matrix[i, 16] = addr_num_match
        matrix[i, 17] = addr_num_jaccard
        matrix[i, 18] = addr_len_diff
        matrix[i, 19] = country_match
        matrix[i, 20] = country_missing
        matrix[i, 21] = is_s2
        matrix[i, 22] = is_s3

    df_feats = pd.DataFrame(matrix, columns=FEATURE_COLUMNS)

    # 5. Semantic similarity from Amazon Bedrock Titan Embeddings
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
