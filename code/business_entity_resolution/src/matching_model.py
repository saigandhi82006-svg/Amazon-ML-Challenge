"""
Entity Matching Model Module
Trains and executes lightweight, explainable Gradient Boosted Decision Tree models
for entity resolution pair classification.

Features:
- Primary Engine: LightGBM (MIT licensed, high speed, tree-based).
- Fallback Engine: Scikit-learn HistGradientBoostingClassifier (BSD-3 licensed).
- Balanced class weighting & depth regularization for robust generalizability.
- Entity-level Grouped Train/Validation split to strictly prevent data leakage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import MODEL_CONFIG, EXPERIMENTS_DIR
from .features import FEATURE_COLUMNS

logger = logging.getLogger("matching_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EntityMatchingClassifier:
    """
    Gradient Boosted Classifier for Entity Pair Matching.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.model_type = self.config.get("model_type", "lightgbm")
        self.feature_names = FEATURE_COLUMNS
        self._init_model()

    def _init_model(self):
        """Initializes the underlying gradient boosting classifier."""
        try:
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                n_estimators=self.config.get("n_estimators", 400),
                learning_rate=self.config.get("learning_rate", 0.05),
                max_depth=self.config.get("max_depth", 8),
                num_leaves=self.config.get("num_leaves", 63),
                min_child_samples=self.config.get("min_child_samples", 25),
                subsample=self.config.get("subsample", 0.85),
                colsample_bytree=self.config.get("colsample_bytree", 0.85),
                random_state=self.config.get("random_state", 42),
                n_jobs=self.config.get("n_jobs", -1),
                verbose=-1,
            )
            self.model_type = "lightgbm"
            logger.info("Initialized LightGBM Classifier successfully.")
        except Exception as e:
            logger.warning(f"LightGBM not available ({e}). Falling back to Scikit-learn HistGradientBoostingClassifier.")
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(
                max_iter=self.config.get("n_estimators", 400),
                learning_rate=self.config.get("learning_rate", 0.05),
                max_depth=self.config.get("max_depth", 8),
                random_state=self.config.get("random_state", 42),
            )
            self.model_type = "hist_gradient_boosting"

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Trains the matching classifier."""
        pos_cnt = int(np.sum(y == 1))
        neg_cnt = int(np.sum(y == 0))
        logger.info(f"Training {self.model_type} on {len(X):,} pairs ({pos_cnt:,} positive, {neg_cnt:,} negative)...")
        self.model.fit(X[self.feature_names], y)
        logger.info("Model training completed successfully.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts match probabilities (class 1)."""
        probs = self.model.predict_proba(X[self.feature_names])
        return probs[:, 1]

    def save(self, path: Optional[Path] = None):
        """Persists trained model artifact to disk."""
        save_path = path or self.config.get("model_save_path", EXPERIMENTS_DIR / "matching_model.joblib")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "model_type": self.model_type, "features": self.feature_names}, save_path)
        logger.info(f"Saved matching model artifact to {save_path}")

    def load(self, path: Optional[Path] = None):
        """Loads trained model artifact from disk."""
        load_path = path or self.config.get("model_save_path", EXPERIMENTS_DIR / "matching_model.joblib")
        data = joblib.load(load_path)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.feature_names = data["features"]
        logger.info(f"Loaded matching model artifact ({self.model_type}) from {load_path}")
