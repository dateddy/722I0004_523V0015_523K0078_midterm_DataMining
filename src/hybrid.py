"""
Hybrid Recommendation System for MovieLens.

This module implements four complementary hybrid recommendation strategies:
  1. WeightedHybrid: Linear combination of multiple models with normalized scores.
  2. SwitchingHybrid: Routes users to different models based on interaction history.
  3. CascadeHybrid: Two-stage pipeline (candidate generation + reranking).
  4. HybridEnsemble: Meta-hybrid combining all three strategies.

Each hybrid strategy is model-agnostic—accepts any model that implements
the standard .recommend(user_idx, n, seen_items) interface.

Usage Example:
    from src.hybrid import WeightedHybrid, SwitchingHybrid, CascadeHybrid

    # Create a weighted hybrid combining three models
    hybrid = WeightedHybrid(
        models={
            "svd":     (svd_model,  0.5),
            "item_cf": (item_cf,    0.3),
            "content": (cb_model,   0.2),
        },
        n_items=3706,
        normalize=True,
    )
    
    # Get recommendations
    recs = hybrid.recommend(user_idx=0, n=10)
    scored = hybrid.recommend_with_scores(user_idx=0, n=10)
    
    # Inspect model weights
    weights = hybrid.get_weights()
    print(weights)

    # Switch between collaborative and content-based based on user history
    switching = SwitchingHybrid(
        cf_model=collaborative_model,
        cb_model=content_model,
        train_df=train_df,
        cold_threshold=5,
    )
    recs = switching.recommend(user_idx=0, n=10)

    # Two-stage cascade: fast retrieval + precise re-ranking
    cascade = CascadeHybrid(
        candidate_model=item_cf,
        reranker_model=svd_model,
        candidate_n=200,
    )
    recs = cascade.recommend(user_idx=0, n=10)

    # Full meta-hybrid
    ensemble = HybridEnsemble(
        weighted=hybrid,
        cb_model=content_model,
        train_df=train_df,
        cold_threshold=5,
        use_cascade=True,
        candidate_model=item_cf,
        candidate_n=200,
    )
    recs = ensemble.recommend(user_idx=0, n=10)
    print(ensemble.describe())
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Optional, Dict, List, Tuple, Set, Any

logger = logging.getLogger(__name__)


class WeightedHybrid:
    """
    Linearly combines recommendations from multiple models using weighted averaging.
    
    Each model's score vector is normalized (min-max) independently, then scaled
    by its weight and summed to produce a final score vector. This ensures that
    models with different score ranges do not unfairly dominate.
    
    Note: This class does NOT store train_df internally, so it can only mask
    seen_items passed in via the seen_items parameter. Models requiring
    train_df knowledge for masking should be pre-masked before instantiation.
    
    Attributes:
        models (dict): {model_name: (model_instance, weight)}.
        n_items (int): Total number of items in the dataset.
        normalize (bool): Whether to apply min-max normalization per model.
        _weights_normalized (dict): Normalized weights (sum to 1.0).
    """

    def __init__(
        self,
        models: Dict,
        n_items: int,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the WeightedHybrid.
        
        Args:
            models: Dictionary of {model_name: (model_instance, weight)}.
                    Weights do not need to sum to 1—they are normalized internally.
            n_items: Total number of items. Must match across all component models.
            normalize: If True, apply min-max normalization to each model's scores
                       before combining to prevent score-scale domination.
        
        Example:
            hybrid = WeightedHybrid(
                models={
                    "svd": (svd_model, 0.5),
                    "cf": (cf_model, 0.3),
                    "content": (cb_model, 0.2),
                },
                n_items=3706,
            )
        """
        self.models = models
        self.n_items = n_items
        self.normalize = normalize

        # Extract and normalize weights
        raw_weights = {name: weight for name, (_, weight) in models.items()}
        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0:
            raise ValueError("Sum of model weights must be positive.")
        self._weights_normalized = {
            name: w / weight_sum for name, w in raw_weights.items()
        }

        logger.debug(
            f"WeightedHybrid initialized with {len(models)} models, "
            f"n_items={n_items}, normalize={normalize}"
        )

    def fit(self, train_matrix: Any, train_df: pd.DataFrame) -> "WeightedHybrid":
        """
        Fit component models if they have not been fitted yet.
        
        Detects whether a model is already fitted by checking for a _fitted
        attribute or by attempting to call fit() and catching exceptions.
        Models that are already fitted are skipped.
        
        Args:
            train_matrix: Training matrix (format depends on component models).
            train_df: Training interaction DataFrame with columns [user_idx, item_idx, ...].
        
        Returns:
            self (for chaining).
        
        Raises:
            Logs warnings if any model's fit() fails; does not raise.
        """
        for name, (model, _) in self.models.items():
            if hasattr(model, "_fitted"):
                logger.debug(f"Model '{name}' already fitted; skipping.")
                continue

            try:
                if hasattr(model, "fit"):
                    model.fit(train_matrix, train_df)
                    logger.debug(f"Successfully fitted model '{name}'.")
                else:
                    logger.debug(f"Model '{name}' has no fit() method; skipping.")
            except Exception as e:
                logger.warning(f"Failed to fit model '{name}': {e}")

        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        seen_items: Optional[Set] = None,
    ) -> List[int]:
        """
        Generate top-n recommendations by weighted combination of model scores.
        
        Args:
            user_idx: User index.
            n: Number of recommendations to return.
            seen_items: Set of item indices to exclude (already rated by user, etc.).
                        If None, no masking is applied beyond what component models do.
        
        Returns:
            List of up to n recommended item indices, sorted by combined score descending.
        
        Example:
            recs = hybrid.recommend(user_idx=5, n=10)
        """
        if seen_items is None:
            seen_items = set()

        # Collect scores from all models
        combined_scores = np.zeros(self.n_items, dtype=np.float32)

        for model_name, (model, _) in self.models.items():
            try:
                scores = self._get_scores(model, user_idx)
                if self.normalize:
                    scores = self._minmax(scores)
                scores *= self._weights_normalized[model_name]
                combined_scores += scores
            except Exception as e:
                logger.warning(
                    f"Model '{model_name}' failed for user {user_idx}: {e}. "
                    "Skipping its contribution."
                )

        # Mask seen items
        if seen_items:
            combined_scores[list(seen_items)] = -np.inf

        # Return top-n
        top_indices = np.argsort(-combined_scores)[:n]
        result = [int(idx) for idx in top_indices if np.isfinite(combined_scores[idx])]

        logger.debug(
            f"WeightedHybrid.recommend(user={user_idx}, n={n}): "
            f"returned {len(result)} items"
        )
        return result

    def recommend_with_scores(
        self,
        user_idx: int,
        n: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Generate top-n recommendations with combined scores.
        
        Args:
            user_idx: User index.
            n: Number of recommendations to return.
        
        Returns:
            List of (item_idx, combined_score) tuples, sorted by score descending.
        
        Example:
            scored_recs = hybrid.recommend_with_scores(user_idx=5, n=10)
            for item_idx, score in scored_recs:
                print(f"Item {item_idx}: {score:.4f}")
        """
        # Collect scores from all models
        combined_scores = np.zeros(self.n_items, dtype=np.float32)

        for model_name, (model, _) in self.models.items():
            try:
                scores = self._get_scores(model, user_idx)
                if self.normalize:
                    scores = self._minmax(scores)
                scores *= self._weights_normalized[model_name]
                combined_scores += scores
            except Exception as e:
                logger.warning(
                    f"Model '{model_name}' failed for user {user_idx}: {e}. "
                    "Skipping contribution."
                )

        # Return top-n with scores
        top_indices = np.argsort(-combined_scores)[:n]
        result = [
            (int(idx), float(combined_scores[idx]))
            for idx in top_indices
            if np.isfinite(combined_scores[idx])
        ]

        return result

    def _get_scores(self, model: Any, user_idx: int) -> np.ndarray:
        """
        Extract a score vector for a user from a model.
        
        Attempts three strategies in order:
          1. Call model.recommend_with_scores(user_idx, n=n_items) if available.
          2. Call model.recommend(user_idx, n=n_items) and convert ranks to reciprocal scores.
          3. Return zeros if both fail.
        
        For very large n_items (>50k), caps n at 10k to avoid memory/time overhead.
        This is a trade-off: we only score the top 10k candidates from the model,
        then place those scores in a zero-filled array. Items not in the top 10k
        are implicitly scored as 0.
        
        Args:
            model: A model instance with .recommend() or .recommend_with_scores().
            user_idx: User index.
        
        Returns:
            Score vector of shape (n_items,), dtype float32.
        """
        scores = np.zeros(self.n_items, dtype=np.float32)

        # Cap n to avoid excessive computation on huge item sets
        n_query = min(self.n_items, 10000)

        # Strategy 1: recommend_with_scores
        if hasattr(model, "recommend_with_scores"):
            try:
                scored_items = model.recommend_with_scores(user_idx, n=n_query)
                for item_idx, score in scored_items:
                    if 0 <= item_idx < self.n_items:
                        scores[item_idx] = score
                logger.debug(
                    f"_get_scores(model): scored {len(scored_items)} items via recommend_with_scores"
                )
                return scores
            except Exception as e:
                logger.debug(
                    f"_get_scores: recommend_with_scores failed: {e}"
                )

        # Strategy 2: recommend → reciprocal rank
        try:
            recommended = model.recommend(user_idx, n=n_query, seen_items=set())
            for rank, item_idx in enumerate(recommended):
                if 0 <= item_idx < self.n_items:
                    scores[item_idx] = 1.0 / (rank + 1)
            logger.debug(
                f"_get_scores(model): scored {len(recommended)} items via reciprocal rank"
            )
            return scores
        except Exception as e:
            logger.debug(f"_get_scores: recommend failed: {e}")

        # Strategy 3: return zeros
        logger.warning(
            f"_get_scores: both strategies failed for user {user_idx}; returning zeros"
        )
        return scores

    def _minmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to a score vector.
        
        Normalizes to [0, 1] over finite values only. If all non-masked values
        are equal or all masked, returns a zero vector (avoids division by zero).
        
        Args:
            scores: Score vector (may contain -inf for masked items).
        
        Returns:
            Normalized score vector with same shape.
        """
        finite_mask = np.isfinite(scores)
        if not np.any(finite_mask):
            return np.zeros_like(scores)

        finite_scores = scores[finite_mask]
        s_min, s_max = finite_scores.min(), finite_scores.max()

        if s_min == s_max:
            # All finite values equal; avoid division by zero
            result = np.zeros_like(scores)
            result[finite_mask] = 0.5
            return result

        result = np.zeros_like(scores, dtype=np.float32)
        result[finite_mask] = (finite_scores - s_min) / (s_max - s_min)
        return result

    def get_weights(self) -> Dict[str, float]:
        """
        Return normalized model weights.
        
        Returns:
            Dictionary {model_name: normalized_weight} where weights sum to 1.0.
        
        Example:
            weights = hybrid.get_weights()
            for name, w in weights.items():
                print(f"{name}: {w:.2%}")
        """
        return self._weights_normalized.copy()


class SwitchingHybrid:
    """
    Routes users to different models based on interaction history.
    
    Users with few interactions (cold-start) are routed to a content-based model;
    users with many interactions (warm-start) are routed to a collaborative model.
    This improves accuracy by using the most appropriate strategy per user.
    
    Attributes:
        cf_model: Collaborative filtering model (warm-start users).
        cb_model: Content-based model (cold-start users).
        train_df: Training interactions DataFrame [user_idx, item_idx, ...].
        cold_threshold: Interaction count threshold (below → cold-start).
        _user_counts: Cached {user_idx: interaction_count} dict.
        _user_rated_items: Cached {user_idx: set(item_idxs)} for masking.
    """

    def __init__(
        self,
        cf_model: Any,
        cb_model: Any,
        train_df: pd.DataFrame,
        cold_threshold: int = 5,
    ) -> None:
        """
        Initialize the SwitchingHybrid.
        
        Args:
            cf_model: Collaborative filtering model (e.g., UserBasedCF, SVD).
            cb_model: Content-based model.
            train_df: Training DataFrame with at least columns [user_idx, item_idx].
            cold_threshold: Users with < this many interactions use cb_model.
        
        Example:
            switching = SwitchingHybrid(
                cf_model=svd_model,
                cb_model=content_model,
                train_df=train_df,
                cold_threshold=5,
            )
        """
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.train_df = train_df
        self.cold_threshold = cold_threshold

        # Pre-compute user interaction counts and rated items
        self._user_counts = train_df.groupby("user_idx").size().to_dict()
        self._user_rated_items = (
            train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )

        logger.debug(
            f"SwitchingHybrid initialized: cold_threshold={cold_threshold}, "
            f"{len(self._user_counts)} users"
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        seen_items: Optional[Set] = None,
    ) -> List[int]:
        """
        Route user to appropriate model and generate recommendations.
        
        Args:
            user_idx: User index.
            n: Number of recommendations.
            seen_items: Additional items to exclude (if any).
        
        Returns:
            List of up to n recommended item indices.
        
        Example:
            recs = switching.recommend(user_idx=5, n=10)
        """
        if seen_items is None:
            seen_items = set()

        # Union with training interactions
        all_seen = seen_items | self._user_rated_items.get(user_idx, set())

        model = self.cf_model if self.which_model(user_idx) == "cf" else self.cb_model

        try:
            return model.recommend(user_idx, n=n, seen_items=all_seen)
        except Exception as e:
            logger.warning(
                f"Model {self.which_model(user_idx)} failed for user {user_idx}: {e}"
            )
            return []

    def which_model(self, user_idx: int) -> str:
        """
        Determine which model to use for a user.
        
        Args:
            user_idx: User index.
        
        Returns:
            "cf" for collaborative filtering (warm-start), "content" for content-based (cold-start).
        
        Example:
            model_name = switching.which_model(user_idx=5)
            print(f"User 5 uses {model_name} model")
        """
        count = self._user_counts.get(user_idx, 0)
        return "cf" if count >= self.cold_threshold else "content"

    def user_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the user population split.
        
        Returns:
            Dictionary with keys:
              - n_cold: Number of cold-start users.
              - n_warm: Number of warm-start users.
              - threshold: The cold_threshold used.
              - cold_pct: Percentage of users that are cold-start.
        
        Example:
            stats = switching.user_stats()
            print(f"{stats['cold_pct']:.1%} of users are cold-start")
        """
        counts = list(self._user_counts.values())
        n_cold = sum(1 for c in counts if c < self.cold_threshold)
        n_warm = len(counts) - n_cold
        cold_pct = n_cold / len(counts) if counts else 0

        return {
            "n_cold": n_cold,
            "n_warm": n_warm,
            "threshold": self.cold_threshold,
            "cold_pct": cold_pct,
        }


class CascadeHybrid:
    """
    Two-stage recommendation pipeline: candidate generation + re-ranking.
    
    Stage 1 generates a pool of high-recall candidates (fast, broad coverage).
    Stage 2 re-ranks only those candidates for precision (expensive, but limited scope).
    
    Final recommendations are the top items from the re-ranked list, with any
    remaining candidates appended to ensure we return n items.
    
    Attributes:
        candidate_model: Fast model for stage 1 (e.g., ItemBasedCF).
        reranker_model: Precise model for stage 2 (e.g., SVD, ContentBased).
        candidate_n: Size of the candidate pool generated in stage 1.
    """

    def __init__(
        self,
        candidate_model: Any,
        reranker_model: Any,
        candidate_n: int = 200,
    ) -> None:
        """
        Initialize the CascadeHybrid.
        
        Args:
            candidate_model: Fast retrieval model (stage 1).
            reranker_model: Precise re-ranking model (stage 2).
            candidate_n: Number of candidates to generate in stage 1.
        
        Example:
            cascade = CascadeHybrid(
                candidate_model=item_cf,
                reranker_model=svd_model,
                candidate_n=200,
            )
        """
        self.candidate_model = candidate_model
        self.reranker_model = reranker_model
        self.candidate_n = candidate_n

        logger.debug(
            f"CascadeHybrid initialized with candidate_n={candidate_n}"
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        seen_items: Optional[Set] = None,
    ) -> List[int]:
        """
        Generate recommendations via two-stage cascade.
        
        Args:
            user_idx: User index.
            n: Number of final recommendations.
            seen_items: Items to exclude.
        
        Returns:
            List of up to n recommended item indices.
        
        Example:
            recs = cascade.recommend(user_idx=5, n=10)
        """
        if seen_items is None:
            seen_items = set()

        stages = self.recommend_with_stages(user_idx, n=n)
        return stages["final"]

    def recommend_with_stages(
        self,
        user_idx: int,
        n: int = 10,
    ) -> Dict[str, List[int]]:
        """
        Generate recommendations and return intermediate outputs.
        
        Useful for understanding what each stage contributes and for
        debugging the cascade pipeline.
        
        Args:
            user_idx: User index.
            n: Number of final recommendations.
        
        Returns:
            Dictionary with keys:
              - "candidates": Stage 1 output (candidate_n items).
              - "reranked": Stage 2 output (candidate_n items).
              - "final": Merged final list (up to n items).
        
        Example:
            stages = cascade.recommend_with_stages(user_idx=5, n=10)
            print(f"Stage 1 returned {len(stages['candidates'])} candidates")
            print(f"Final: {stages['final']}")
        """
        candidates = []
        reranked = []
        final = []

        # Stage 1: Generate candidates
        try:
            candidates = self.candidate_model.recommend(
                user_idx, n=self.candidate_n, seen_items=set()
            )
            logger.debug(f"Stage 1 (candidate): {len(candidates)} items")
        except Exception as e:
            logger.warning(f"Stage 1 (candidate) failed: {e}")
            return {
                "candidates": [],
                "reranked": [],
                "final": [],
            }

        # Stage 2: Re-rank candidates
        try:
            reranked = self.reranker_model.recommend(
                user_idx, n=self.candidate_n, seen_items=set()
            )
            logger.debug(f"Stage 2 (reranker): {len(reranked)} items")
        except Exception as e:
            logger.warning(
                f"Stage 2 (reranker) failed: {e}. Returning candidates as fallback."
            )
            return {
                "candidates": candidates,
                "reranked": [],
                "final": candidates[:n],
            }

        # Merge: reranked items first (in order), then remaining candidates
        candidate_set = set(candidates)
        final = []
        for item in reranked:
            if item in candidate_set:
                final.append(item)
        for item in candidates:
            if item not in final:
                final.append(item)

        final = final[:n]

        logger.debug(
            f"Cascade: stage1={len(candidates)}, stage2={len(reranked)}, final={len(final)}"
        )

        return {
            "candidates": candidates,
            "reranked": reranked,
            "final": final,
        }


class HybridEnsemble:
    """
    Meta-hybrid combining switching and cascade strategies.
    
    Routes warm-start users to a weighted hybrid, cold-start users to content-based,
    and optionally wraps everything in a cascade for production deployment.
    
    This provides the most sophisticated routing: SwitchingHybrid decides whether to
    use the weighted ensemble or content-based, and CascadeHybrid optionally refines
    the final ranking via a second model.
    
    Attributes:
        weighted: WeightedHybrid instance for warm-start routing.
        cb_model: Content-based model for cold-start routing.
        train_df: Training DataFrame (for cold-start cutoffs).
        _switching: Internal SwitchingHybrid instance.
        _cascade: Internal CascadeHybrid if use_cascade=True, else None.
        use_cascade: Whether to apply cascade re-ranking.
    """

    def __init__(
        self,
        weighted: "WeightedHybrid",
        cb_model: Any,
        train_df: pd.DataFrame,
        cold_threshold: int = 5,
        use_cascade: bool = False,
        candidate_model: Optional[Any] = None,
        candidate_n: int = 200,
    ) -> None:
        """
        Initialize the HybridEnsemble.
        
        Args:
            weighted: WeightedHybrid instance for warm-start users.
            cb_model: Content-based model for cold-start users.
            train_df: Training DataFrame [user_idx, item_idx, ...].
            cold_threshold: Interaction count for switching decision.
            use_cascade: If True, wrap switcher in a cascade pipeline.
            candidate_model: Model for candidate generation (stage 1 of cascade).
                            Required if use_cascade=True.
            candidate_n: Size of candidate pool for cascade.
        
        Example:
            ensemble = HybridEnsemble(
                weighted=weighted_hybrid,
                cb_model=content_model,
                train_df=train_df,
                cold_threshold=5,
                use_cascade=True,
                candidate_model=item_cf,
                candidate_n=200,
            )
            recs = ensemble.recommend(user_idx=5, n=10)
            print(ensemble.describe())
        """
        self.weighted = weighted
        self.cb_model = cb_model
        self.train_df = train_df
        self.use_cascade = use_cascade

        # Build switching hybrid: warm users → weighted, cold users → content
        self._switching = SwitchingHybrid(
            cf_model=weighted,
            cb_model=cb_model,
            train_df=train_df,
            cold_threshold=cold_threshold,
        )

        # Optionally wrap in cascade
        if use_cascade:
            if candidate_model is None:
                raise ValueError(
                    "candidate_model required when use_cascade=True"
                )
            self._cascade = CascadeHybrid(
                candidate_model=candidate_model,
                reranker_model=self._switching,
                candidate_n=candidate_n,
            )
        else:
            self._cascade = None

        logger.debug(
            f"HybridEnsemble initialized: cold_threshold={cold_threshold}, "
            f"use_cascade={use_cascade}"
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        seen_items: Optional[Set] = None,
    ) -> List[int]:
        """
        Generate recommendations using the full ensemble pipeline.
        
        Args:
            user_idx: User index.
            n: Number of recommendations.
            seen_items: Additional items to exclude.
        
        Returns:
            List of up to n recommended item indices.
        
        Example:
            recs = ensemble.recommend(user_idx=5, n=10)
        """
        if seen_items is None:
            seen_items = set()

        if self._cascade is not None:
            return self._cascade.recommend(user_idx, n=n, seen_items=seen_items)
        else:
            return self._switching.recommend(user_idx, n=n, seen_items=seen_items)

    def describe(self) -> str:
        """
        Return a human-readable description of the ensemble architecture.
        
        Returns:
            String describing the full model stack and routing logic.
        
        Example:
            print(ensemble.describe())
            # Output: CascadeHybrid(candidate=ItemCF → reranker=Switching(cold→Content,
            #         warm→Weighted[SVD:0.50, ItemCF:0.30, Content:0.20]))
        """
        weighted_str = self._format_weighted()
        switching_str = f"Switching(cold→Content, warm→{weighted_str})"

        if self._cascade is not None:
            return f"CascadeHybrid(candidate={self._cascade.candidate_model.__class__.__name__} → reranker={switching_str})"
        else:
            return switching_str

    def _format_weighted(self) -> str:
        """Helper to format the weighted hybrid description."""
        weights = self.weighted.get_weights()
        weight_strs = [
            f"{name}:{w:.2f}"
            for name, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)
        ]
        return f"Weighted[{', '.join(weight_strs)}]"


def evaluate_hybrid_configs(
    configs: Dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k: int = 10,
    sample_users: int = 200,
) -> pd.DataFrame:
    """
    Evaluate multiple hybrid configurations on a test set.
    
    Runs ranking evaluation on each configuration (e.g., Precision@K, Recall@K, NDCG@K)
    using the Evaluator from src.evaluation. Results are returned in a DataFrame
    sorted by NDCG@K descending.
    
    Args:
        configs: Dictionary {config_name: model_instance} of configurations to evaluate.
        train_df: Training interactions DataFrame [user_idx, item_idx, ...].
        test_df: Test interactions DataFrame [user_idx, item_idx, ...].
        n_items: Total number of items (for Recall and Coverage).
        k: Ranking cutoff (default 10).
        sample_users: Number of test users to evaluate (if < all test users, randomly samples).
    
    Returns:
        DataFrame with columns:
          - model: Configuration name.
          - Precision@K: Precision at cutoff k.
          - Recall@K: Recall at cutoff k.
          - NDCG@K: Normalized Discounted Cumulative Gain.
          - MRR: Mean Reciprocal Rank.
          - HitRate: Percentage of users with >= 1 relevant item in top-k.
          - Coverage: Percentage of unique items recommended across all users.
        Sorted by NDCG@K descending.
    
    Example:
        results = evaluate_hybrid_configs(
            configs={
                "weighted": weighted_hybrid,
                "switching": switching_hybrid,
                "cascade": cascade_hybrid,
            },
            train_df=train_df,
            test_df=test_df,
            n_items=3706,
            k=10,
        )
        print(results)
    """
    # Deferred import to avoid circular dependency
    from src.evaluation import Evaluator

    # Pre-compute training items per user for masking
    train_user_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # Determine test users
    test_users = test_df["user_idx"].unique()
    if sample_users < len(test_users):
        np.random.seed(42)
        test_users = np.random.choice(test_users, size=sample_users, replace=False)

    evaluator = Evaluator(train_df=train_df, test_df=test_df, n_items=n_items)

    results = []
    for model_name, model in configs.items():
        logger.info(f"Evaluating {model_name}...")
        start = time.time()

        try:
            scores = evaluator.evaluate_ranking(
                model=model,
                test_users=test_users,
                k=k,
            )
            elapsed = time.time() - start
            scores["model"] = model_name
            scores["time_s"] = elapsed
            results.append(scores)
            logger.info(
                f"  {model_name}: NDCG@{k}={scores[f'NDCG@{k}']:.4f} ({elapsed:.1f}s)"
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(f"NDCG@{k}", ascending=False)

    return results_df


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == "__main__":
    """Minimal test to verify module imports and basic functionality."""
    logging.basicConfig(level=logging.DEBUG)

    # Create mock models with minimal interface
    class MockModel:
        def recommend(self, user_idx, n, seen_items=None):
            """Return a dummy list of items."""
            return list(range(1, n + 1))

    # Test WeightedHybrid
    print("Testing WeightedHybrid...")
    weighted = WeightedHybrid(
        models={
            "model_a": (MockModel(), 0.6),
            "model_b": (MockModel(), 0.4),
        },
        n_items=100,
        normalize=True,
    )
    recs = weighted.recommend(user_idx=0, n=10)
    print(f"  WeightedHybrid.recommend(user=0, n=10): {recs}")
    assert len(recs) > 0, "Expected non-empty recommendations"

    # Test SwitchingHybrid
    print("\nTesting SwitchingHybrid...")
    train_df = pd.DataFrame(
        {"user_idx": [0, 0, 1, 1, 1, 1, 1, 1, 1], "item_idx": range(9)}
    )
    switching = SwitchingHybrid(
        cf_model=MockModel(),
        cb_model=MockModel(),
        train_df=train_df,
        cold_threshold=5,
    )
    recs = switching.recommend(user_idx=0, n=5)
    print(f"  SwitchingHybrid.recommend(user=0, n=5): {recs}")
    stats = switching.user_stats()
    print(f"  User stats: {stats}")

    # Test CascadeHybrid
    print("\nTesting CascadeHybrid...")
    cascade = CascadeHybrid(
        candidate_model=MockModel(),
        reranker_model=MockModel(),
        candidate_n=50,
    )
    recs = cascade.recommend(user_idx=0, n=10)
    print(f"  CascadeHybrid.recommend(user=0, n=10): {recs}")
    stages = cascade.recommend_with_stages(user_idx=0, n=10)
    print(f"  Stages: candidates={len(stages['candidates'])}, final={len(stages['final'])}")

    # Test HybridEnsemble
    print("\nTesting HybridEnsemble...")
    ensemble = HybridEnsemble(
        weighted=weighted,
        cb_model=MockModel(),
        train_df=train_df,
        cold_threshold=5,
        use_cascade=False,
    )
    recs = ensemble.recommend(user_idx=0, n=10)
    print(f"  HybridEnsemble.recommend(user=0, n=10): {recs}")
    desc = ensemble.describe()
    print(f"  Architecture: {desc}")

    print("\n✓ All smoke tests passed!")
