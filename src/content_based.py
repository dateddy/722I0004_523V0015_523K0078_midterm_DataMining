"""
content_based.py
----------------
Content-Based Filtering recommender using genre and tag features.

Algorithm:
  1. User genre profile: normalized user preferences per genre
  2. Movie feature vectors: genre multi-hot encoding or TF-IDF tag vectors
  3. Similarity computation: cosine similarity between user profiles and
     movie feature vectors
  4. Graceful fallback to BaselineModel for cold-start cases
  5. Vectorised Top-K ranking with seen-item masking

Design constraints enforced:
  - Features computed from train data ONLY → no data leakage
  - test.csv is NEVER read or touched inside this module
  - Index consistency: always maps via movie_to_idx
  - Supports both genre-based and tag-based (TF-IDF) content features

Inputs  : user_genre_profile (DataFrame), movie_genre_matrix (array),
          baseline model, movies metadata
Outputs : rating predictions (float), Top-K recommendation lists
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Default number of neighbors to use in weighted prediction
DEFAULT_N_NEIGHBORS = 20

# Similarity threshold below which neighbors are not considered
DEFAULT_SIM_THRESHOLD = 0.0

# Default weight for blending baseline and content-based predictions
DEFAULT_BLEND_WEIGHT = 0.5


# ─────────────────────────────────────────────────────────────
# CONTENT-BASED FILTERING CLASS
# ─────────────────────────────────────────────────────────────

class ContentBasedCF:
    """
    Content-Based Collaborative Filtering using genre and/or tag features.

    Uses user genre preferences and movie feature vectors (genre multi-hot
    encoding or TF-IDF tag vectors) to compute similarity and make predictions.

    Attributes
    ----------
    user_genre_profile : pd.DataFrame
        User-genre preference matrix (long format or pivoted).
        For wide format: shape (n_users, n_genres)
    movie_features : np.ndarray or csr_matrix
        Movie feature vectors (shape: n_movies x n_features).
        Can be multi-hot genre encoding or TF-IDF tag vectors.
    movie_to_idx : dict
        Maps movieId -> column index in feature matrix.
    idx_to_movie : dict
        Reverse map: column index -> movieId.
    movie_idx_to_genre_id : dict
        Maps movie feature index to original movieId for reference.
    baseline : BaselineModel
        Fallback predictor for cold-start cases.
    n_neighbors : int
        Maximum number of similar movies to use per prediction.
    sim_threshold : float
        Minimum cosine similarity to include a neighbor.
    blend_weight : float
        Weight for blending content-based and baseline predictions.
        blend_weight * content_based + (1 - blend_weight) * baseline
    is_fitted : bool
        Whether fit() has been called.
    """

    def __init__(
        self,
        baseline=None,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        blend_weight: float = DEFAULT_BLEND_WEIGHT,
    ):
        """
        Parameters
        ----------
        baseline      : fitted BaselineModel instance (used for fallback)
        n_neighbors   : max similar movies to use per prediction
        sim_threshold : minimum cosine similarity to include a neighbor
        blend_weight  : weight for content-based in blend with baseline
        """
        self.baseline      = baseline
        self.n_neighbors   = n_neighbors
        self.sim_threshold = sim_threshold
        self.blend_weight  = blend_weight

        self.user_genre_profile = None
        self.movie_features     = None
        self.movie_to_idx       = {}
        self.idx_to_movie       = {}
        self.movie_idx_to_genre_id = {}
        self.movie_similarity   = None
        self.is_fitted          = False

    # ── FIT ────────────────────────────────────────────────────

    def fit(
        self,
        user_genre_profile: pd.DataFrame,
        movie_features: np.ndarray,
        movie_to_idx: dict,
        movie_ids: list = None,
    ) -> "ContentBasedCF":
        """
        Fit the content-based model on user genre profile and movie features.

        Steps:
        1. Store user genre profile and movie features
        2. Store index mappings
        3. Compute movie-movie cosine similarity from feature vectors

        Parameters
        ----------
        user_genre_profile : pd.DataFrame
            User-genre preference matrix. Can be:
            - Wide format (users × genres): shape (n_users, n_genres)
            - Long format: will be converted to wide format internally
            If long format with columns [userId, genre, combined_score],
            will be pivoted automatically.
        movie_features : np.ndarray or csr_matrix
            Feature vectors for movies (shape: n_movies, n_features).
            Can be multi-hot genre encoding or TF-IDF tag vectors.
        movie_to_idx : dict
            Maps movieId -> feature vector row index.
        movie_ids : list, optional
            List of movieIds in order. If None, uses sorted keys from movie_to_idx.

        Returns
        -------
        self
        """
        # ── Store movie features and index mappings ─────────────
        self.movie_features = movie_features
        self.movie_to_idx   = movie_to_idx
        self.idx_to_movie   = {v: k for k, v in movie_to_idx.items()}

        # Ensure movie_ids matches feature matrix shape
        if movie_ids is None:
            movie_ids = sorted(movie_to_idx.keys(), key=lambda x: movie_to_idx[x])
        self.movie_idx_to_genre_id = {i: mid for i, mid in enumerate(movie_ids)}

        n_movies, n_features = (
            movie_features.shape[0],
            movie_features.shape[1]
        )
        print(f"[ContentBasedCF.fit] Movie feature matrix shape: {movie_features.shape}")

        # Handle sparsity
        if issparse(movie_features):
            print(f"  Sparse density: {movie_features.nnz / (n_movies * n_features):.4%}")
        else:
            print(f"  Dense array")

        # ── Convert user genre profile to wide format if needed ──
        if isinstance(user_genre_profile, pd.DataFrame):
            if "genre" in user_genre_profile.columns:
                # Long format: pivot to wide
                profile_wide = user_genre_profile.pivot_table(
                    index="userId",
                    columns="genre",
                    values="combined_score",
                    fill_value=0.0
                )
            elif "userId" in user_genre_profile.columns:
                # Already wide format
                profile_wide = user_genre_profile
            else:
                raise ValueError(
                    "user_genre_profile must have 'userId' and either "
                    "'genre' (long) or genre columns (wide format)"
                )
        else:
            raise TypeError("user_genre_profile must be a pandas DataFrame")

        self.user_genre_profile = profile_wide
        print(f"  User-genre profile shape: {profile_wide.shape}")

        # ── Compute movie-movie cosine similarity ────────────────
        # This captures content similarity between movies
        if issparse(movie_features):
            # For sparse matrices, compute similarity in two steps
            self.movie_similarity = cosine_similarity(
                movie_features, movie_features
            )
        else:
            # For dense arrays
            self.movie_similarity = cosine_similarity(movie_features)

        print(f"  Movie-movie similarity computed: shape {self.movie_similarity.shape}")
        self.is_fitted = True
        print(f"[ContentBasedCF.fit] Done.")
        return self

    # ── PREDICT ────────────────────────────────────────────────

    def predict(self, userId, movieId) -> float:
        """
        Predict the rating using content-based similarity and genre profile.

        Strategy:
        1. Get user's genre preference vector
        2. For target movie, find similar movies using feature similarity
        3. Compute weighted average of similar movies the user has seen
        4. Blend with baseline predictor for robustness

        Parameters
        ----------
        userId  : user identifier
        movieId : movie identifier

        Returns
        -------
        float : predicted rating in [0.5, 5.0], never NaN
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Fallback for unknown items
        if movieId not in self.movie_to_idx:
            if self.baseline:
                return self.baseline.predict(userId, movieId)
            return 3.0  # Default middle rating

        movie_idx = self.movie_to_idx[movieId]

        # Get user's genre preference (profile)
        if userId not in self.user_genre_profile.index:
            # User not in genre profile: use baseline
            if self.baseline:
                return self.baseline.predict(userId, movieId)
            return 3.0

        user_profile = self.user_genre_profile.loc[userId].values

        # Get movie feature vector
        if issparse(self.movie_features):
            target_feature = self.movie_features[movie_idx].toarray().ravel()
        else:
            target_feature = self.movie_features[movie_idx]

        # Compute similarity between target movie and user profile
        # (simple dot product as proxy for preference)
        if len(user_profile) == len(target_feature):
            content_score = float(np.dot(user_profile, target_feature))
            # Normalize to rating scale
            # Assuming features are normalized 0-1
            content_pred = 2.5 + (content_score * 2.5)  # Map to [2.5, 5.0] range
        else:
            content_pred = 3.0

        # Blend with baseline
        if self.baseline:
            baseline_pred = self.baseline.predict(userId, movieId)
            blended = (
                self.blend_weight * content_pred +
                (1.0 - self.blend_weight) * baseline_pred
            )
        else:
            blended = content_pred

        return float(np.clip(blended, 0.5, 5.0))

    def predict_batch(self, pairs: pd.DataFrame) -> np.ndarray:
        """
        Vectorised batch prediction for multiple (userId, movieId) pairs.

        Parameters
        ----------
        pairs : pd.DataFrame
            Must contain columns [userId, movieId].

        Returns
        -------
        np.ndarray of float predictions, shape (len(pairs),)
        """
        predictions = np.array([
            self.predict(row["userId"], row["movieId"])
            for _, row in pairs.iterrows()
        ])
        return predictions

    # ── TOP-K RANKING ─────────────────────────────────────────

    def recommend_top_k(
        self,
        userId,
        k: int = 10,
        seen_items: set = None,
    ) -> pd.DataFrame:
        """
        Return top-K recommendations based on content similarity.

        For each unseen movie, computes a content-based score from
        the target movie's feature vector and the user's genre preference.

        Parameters
        ----------
        userId     : user identifier
        k          : number of recommendations to return
        seen_items : set of movieIds already rated by this user in train.
                     If None, no masking is applied.

        Returns
        -------
        pd.DataFrame with columns [movieId, score, rank]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get user profile
        if userId not in self.user_genre_profile.index:
            # User not in profile: fallback to baseline
            if self.baseline:
                return self.baseline.recommend_top_k(userId, k, seen_items)
            return pd.DataFrame(columns=["movieId", "score", "rank"])

        user_profile = self.user_genre_profile.loc[userId].values

        # Compute content scores for all movies
        if issparse(self.movie_features):
            # For sparse: convert to dense for dot product
            features_dense = self.movie_features.toarray()
        else:
            features_dense = self.movie_features

        # Score each movie
        scores = features_dense @ user_profile  # (n_movies,)

        # Create recommendation DataFrame
        recommendations = pd.DataFrame({
            "movieId": [self.idx_to_movie[i] for i in range(len(scores))],
            "score": scores
        })

        # Filter unseen items
        if seen_items:
            recommendations = recommendations[
                ~recommendations["movieId"].isin(seen_items)
            ]

        # Sort and rank
        recommendations = (
            recommendations
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        recommendations["rank"] = range(1, len(recommendations) + 1)

        return recommendations[["movieId", "score", "rank"]].head(k)

    # ── SIMILARITY LOOKUP ──────────────────────────────────────

    def get_similar_movies(
        self,
        movieId,
        k: int = 10,
        threshold: float = None,
    ) -> pd.DataFrame:
        """
        Find similar movies to a given movie based on content features.

        Parameters
        ----------
        movieId   : target movie identifier
        k         : number of similar movies to return
        threshold : similarity threshold (uses self.sim_threshold if None)

        Returns
        -------
        pd.DataFrame with columns [similar_movieId, similarity, rank]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if movieId not in self.movie_to_idx:
            return pd.DataFrame(columns=["similar_movieId", "similarity", "rank"])

        if threshold is None:
            threshold = self.sim_threshold

        movie_idx = self.movie_to_idx[movieId]
        similarities = self.movie_similarity[movie_idx].copy()

        # Create DataFrame
        similar = pd.DataFrame({
            "similar_movieId": [self.idx_to_movie[i] for i in range(len(similarities))],
            "similarity": similarities
        })

        # Exclude self
        similar = similar[similar["similar_movieId"] != movieId]

        # Filter by threshold
        if threshold > 0:
            similar = similar[similar["similarity"] >= threshold]

        # Sort and rank
        similar = (
            similar
            .sort_values("similarity", ascending=False)
            .reset_index(drop=True)
        )
        similar["rank"] = range(1, len(similar) + 1)

        return similar[["similar_movieId", "similarity", "rank"]].head(k)

    # ── DIAGNOSTICS ───────────────────────────────────────────

    def coverage_stats(self) -> dict:
        """
        Compute coverage statistics for the content-based model.

        Returns
        -------
        dict with coverage metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_users = len(self.user_genre_profile)
        n_movies = len(self.movie_to_idx)

        return {
            "n_users":   n_users,
            "n_movies":  n_movies,
            "n_features": self.movie_features.shape[1],
        }
