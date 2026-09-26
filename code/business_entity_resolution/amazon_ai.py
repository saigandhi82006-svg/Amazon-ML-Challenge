"""
Amazon / AWS AI Module — Business Entity Resolution
Integrates Amazon Bedrock Titan Text Embeddings for semantic representation and feature generation.

Architecture & Optimizations:
1. Embeddings computed exclusively per UNIQUE entity text (Name + Address + Country).
2. Uses standard AWS SDK / boto3 credential provider chain (no hardcoded keys).
3. Memory and disk cache in artifacts/amazon_ai_cache/ to prevent duplicate requests.
4. Fast vectorized batch cosine similarity computation.
5. High-speed deterministic fallback representation (character n-gram hashing) when
   offline or AWS Bedrock credentials are unconfigured.
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

from .config import AWS_CONFIG, CACHE_DIR

logger = logging.getLogger("amazon_ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AmazonBedrockEmbeddingClient:
    """
    Amazon Bedrock Client for semantic embedding extraction using Amazon Titan Text Embeddings.
    """

    def __init__(
        self,
        model_id: str = AWS_CONFIG["model_id"],
        region_name: str = AWS_CONFIG["region_name"],
        embedding_dimension: int = AWS_CONFIG["embedding_dimension"],
        cache_path: Optional[Path] = AWS_CONFIG["cache_file"],
    ):
        self.model_id = model_id
        self.region_name = region_name
        self.embedding_dimension = embedding_dimension
        self.cache_path = cache_path
        self.cache: Dict[str, np.ndarray] = {}
        self.client = None
        self.is_bedrock_available = False

        self._load_cache()
        self._init_client()

    def _load_cache(self):
        """Loads cached embeddings from disk if available."""
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}. Initializing empty cache.")
                self.cache = {}

    def save_cache(self):
        """Persists current cache to disk."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved {len(self.cache)} embeddings to cache at {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save embedding cache: {e}")

    def _init_client(self):
        """Initializes boto3 Bedrock client using standard environment configuration."""
        if not AWS_CONFIG["enabled"]:
            logger.info("Amazon Bedrock integration is disabled via configuration.")
            return

        try:
            import boto3
            from botocore.config import Config

            boto_config = Config(
                retries={"max_attempts": AWS_CONFIG["max_retries"], "mode": "standard"},
                region_name=self.region_name,
                connect_timeout=3,
                read_timeout=5,
            )
            # Standard AWS credential provider chain
            self.client = boto3.client("bedrock-runtime", config=boto_config)
            self.is_bedrock_available = True
            logger.info(
                f"Amazon Bedrock Runtime client initialized (Region: {self.region_name}, Model: {self.model_id})."
            )
        except Exception as e:
            logger.info(
                f"Amazon Bedrock client not initialized ({e}). "
                "Operating in deterministic local semantic mode."
            )
            self.is_bedrock_available = False

    @staticmethod
    def format_entity_text(business_name: str, business_address: str, country: str) -> str:
        """
        Constructs standard formatted text representation for embedding.
        Strictly contains competition-provided data only.
        """
        name = str(business_name or "").strip()
        addr = str(business_address or "").strip()
        cntry = str(country or "").strip()
        return f"Business: {name} | Address: {addr} | Country: {cntry}"

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Retrieves normalized embedding vector for a single text.
        """
        if not text:
            return np.zeros(self.embedding_dimension, dtype=np.float32)

        if text in self.cache:
            return self.cache[text]

        if self.is_bedrock_available and self.client:
            try:
                request_body = {
                    "inputText": text[:2048],
                    "dimensions": self.embedding_dimension,
                    "normalize": True,
                }
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body),
                )
                response_body = json.loads(response["body"].read())
                embedding = np.array(response_body["embedding"], dtype=np.float32)
                self.cache[text] = embedding
                return embedding
            except Exception as e:
                logger.debug(f"Bedrock API call unavailable ({e}), using local semantic embedding.")
                # Mark unavailable to avoid repeated connection timeouts
                self.is_bedrock_available = False

        embedding = self._generate_offline_embedding(text)
        self.cache[text] = embedding
        return embedding

    def _generate_offline_embedding(self, text: str) -> np.ndarray:
        """
        Deterministic fast character 3-gram hashing embedding for local execution.
        """
        vec = np.zeros(self.embedding_dimension, dtype=np.float32)
        clean_text = text.lower()
        for i in range(len(clean_text) - 2):
            trigram = clean_text[i : i + 3]
            idx = hash(trigram) % self.embedding_dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def compute_pairwise_similarities(
        self,
        texts1: List[str],
        texts2: List[str],
    ) -> np.ndarray:
        """
        Vectorized computation of pairwise cosine similarities between two lists of entity texts.
        """
        n = len(texts1)
        if n == 0:
            return np.array([], dtype=np.float32)

        # Pre-fetch embeddings for all unique texts
        unique_texts = set(texts1) | set(texts2)
        for t in unique_texts:
            if t not in self.cache:
                self.get_embedding(t)

        # Build 2D embedding matrices
        dim = self.embedding_dimension
        mat1 = np.empty((n, dim), dtype=np.float32)
        mat2 = np.empty((n, dim), dtype=np.float32)

        for i in range(n):
            mat1[i] = self.cache.get(texts1[i], np.zeros(dim, dtype=np.float32))
            mat2[i] = self.cache.get(texts2[i], np.zeros(dim, dtype=np.float32))

        # Vectorized dot product (since vectors are unit normalized, dot product == cosine similarity)
        similarities = np.sum(mat1 * mat2, axis=1)
        return np.clip(similarities, 0.0, 1.0)


# Module-level singleton instance
_bedrock_client: Optional[AmazonBedrockEmbeddingClient] = None


def get_bedrock_client() -> AmazonBedrockEmbeddingClient:
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = AmazonBedrockEmbeddingClient()
    return _bedrock_client


def generate_semantic_features(
    s1_names: List[str],
    s1_addresses: List[str],
    s1_countries: List[str],
    cand_names: List[str],
    cand_addresses: List[str],
    cand_countries: List[str],
) -> np.ndarray:
    """
    Computes pairwise semantic similarity vector for candidate pairs using Amazon Bedrock / semantic module.
    """
    client = get_bedrock_client()
    texts1 = [
        client.format_entity_text(n, a, c)
        for n, a, c in zip(s1_names, s1_addresses, s1_countries)
    ]
    texts2 = [
        client.format_entity_text(n, a, c)
        for n, a, c in zip(cand_names, cand_addresses, cand_countries)
    ]
    return client.compute_pairwise_similarities(texts1, texts2)
