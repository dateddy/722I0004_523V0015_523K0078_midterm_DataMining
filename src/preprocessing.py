import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import joblib


# ─────────────────────────────────────────────────────────────
# 1. LOADERS
# ─────────────────────────────────────────────────────────────

def load_ratings(path: str, sample: int = None) -> pd.DataFrame:
    """
    Load ratings.csv.
    Args:
        path   : full path to ratings.csv
        sample : number of rows to load (None = full 32M)
    Returns:
        DataFrame with columns [userId, movieId, rating, timestamp]
    """
    df = pd.read_csv(path, nrows=sample)
    print(f"[load_ratings] Loaded {len(df):,} rows")
    return df


def load_movies(path: str) -> pd.DataFrame:
    """Load movies.csv."""
    df = pd.read_csv(path)
    print(f"[load_movies] Loaded {len(df):,} movies")
    return df


def load_tags(path: str) -> pd.DataFrame:
    """Load tags.csv."""
    df = pd.read_csv(path)
    print(f"[load_tags] Loaded {len(df):,} tags")
    return df


def load_links(path: str) -> pd.DataFrame:
    """Load links.csv."""
    df = pd.read_csv(path)
    print(f"[load_links] Loaded {len(df):,} links")
    return df


# ─────────────────────────────────────────────────────────────
# 2. CLEAN RATINGS
# ─────────────────────────────────────────────────────────────

def clean_ratings(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 5
) -> pd.DataFrame:
    """
    Clean the ratings DataFrame:
      - Drop duplicates
      - Convert timestamp to datetime
      - Filter users with fewer than min_user_ratings
      - Filter movies with fewer than min_movie_ratings

    Args:
        ratings           : raw ratings DataFrame
        min_user_ratings  : minimum ratings a user must have (default 20)
        min_movie_ratings : minimum ratings a movie must have (default 5)
    Returns:
        Cleaned ratings DataFrame
    """
    print(f"[clean_ratings] Start: {len(ratings):,} rows")

    # Drop duplicates
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"])
    print(f"  After drop_duplicates   : {len(ratings):,}")

    # Convert timestamp
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["year"]     = ratings["datetime"].dt.year
    ratings["month"]    = ratings["datetime"].dt.month

    # Filter inactive users
    user_counts = ratings["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings["userId"].isin(valid_users)]
    print(f"  After user filter (≥{min_user_ratings}) : {len(ratings):,} | users: {ratings['userId'].nunique():,}")

    # Filter unpopular movies
    movie_counts = ratings["movieId"].value_counts()
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    ratings = ratings[ratings["movieId"].isin(valid_movies)]
    print(f"  After movie filter (≥{min_movie_ratings}): {len(ratings):,} | movies: {ratings['movieId'].nunique():,}")

    ratings = ratings.reset_index(drop=True)
    print(f"[clean_ratings] Done: {len(ratings):,} rows")
    return ratings


# ─────────────────────────────────────────────────────────────
# 3. CLEAN MOVIES
# ─────────────────────────────────────────────────────────────

def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the movies DataFrame:
      - Extract release year from title
      - Split genres into a list
      - Flag movies with no genre
      - Multi-hot encode genres

    Args:
        movies : raw movies DataFrame
    Returns:
        Enriched movies DataFrame with extra columns
    """
    print(f"[clean_movies] Start: {len(movies):,} movies")

    # Extract release year
    movies["release_year"] = (
        movies["title"]
        .str.extract(r"\((\d{4})\)$")
        .astype(float)
    )

    # Split genres
    movies["genre_list"] = movies["genres"].str.split("|")

    # Flag no-genre
    movies["no_genre"] = movies["genres"] == "(no genres listed)"

    # Multi-hot encode genres
    all_genres = sorted({
        g for genres in movies["genre_list"]
        for g in genres
        if g != "(no genres listed)"
    })

    for genre in all_genres:
        col = "genre_" + genre.replace("-", "_").replace("'", "").replace(" ", "_")
        movies[col] = movies["genre_list"].apply(lambda x: int(genre in x))

    print(f"  Release year extracted  : {movies['release_year'].notnull().sum():,} / {len(movies):,}")
    print(f"  No-genre movies         : {movies['no_genre'].sum():,}")
    print(f"  Genres encoded          : {len(all_genres)}")
    print(f"[clean_movies] Done: {movies.shape}")
    return movies


# ─────────────────────────────────────────────────────────────
# 4. CLEAN TAGS
# ─────────────────────────────────────────────────────────────

def clean_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the tags DataFrame:
      - Drop rows with null tags
      - Lowercase and strip whitespace
      - Drop duplicates (same user, movie, tag)

    Args:
        tags : raw tags DataFrame
    Returns:
        Cleaned tags DataFrame
    """
    print(f"[clean_tags] Start: {len(tags):,} rows")

    tags = tags.dropna(subset=["tag"])
    tags["tag"] = tags["tag"].str.lower().str.strip()
    tags = tags.drop_duplicates(subset=["userId", "movieId", "tag"])

    print(f"[clean_tags] Done: {len(tags):,} rows")
    return tags


def build_movie_tag_profile(tags: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all tags per movie into a single string (for TF-IDF).

    Args:
        tags : cleaned tags DataFrame
    Returns:
        DataFrame with [movieId, tag_profile]
    """
    tag_profile = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"tag": "tag_profile"})
    )
    print(f"[build_movie_tag_profile] {len(tag_profile):,} movies have tags")
    return tag_profile


# ─────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────

def temporal_train_test_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: for each user, hold out their LAST test_ratio
    ratings (by timestamp) as the test set.

    This is more realistic than random split because it simulates
    recommending to users based on their past behaviour.

    Args:
        ratings    : cleaned ratings DataFrame (must have 'timestamp' col)
        test_ratio : fraction of each user's ratings to hold out (default 0.2)
    Returns:
        (train_df, test_df)
    """
    print(f"[temporal_split] Splitting {len(ratings):,} rows | test_ratio={test_ratio}")

    ratings = ratings.sort_values(["userId", "timestamp"])

    def split_user(group):
        n = len(group)
        cutoff = max(1, int(n * (1 - test_ratio)))
        group = group.reset_index(drop=True)
        group["split"] = "train"
        group.loc[cutoff:, "split"] = "test"
        return group

    ratings = ratings.groupby("userId", group_keys=False).apply(split_user)

    train = ratings[ratings["split"] == "train"].drop(columns="split").reset_index(drop=True)
    test  = ratings[ratings["split"] == "test"].drop(columns="split").reset_index(drop=True)

    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"  Train users: {train['userId'].nunique():,} | Test users: {test['userId'].nunique():,}")
    return train, test


# ─────────────────────────────────────────────────────────────
# 6. USER-ITEM MATRIX
# ─────────────────────────────────────────────────────────────

def build_user_item_matrix(
    ratings: pd.DataFrame
) -> tuple[csr_matrix, dict, dict]:
    """
    Build a sparse user-item rating matrix from ratings DataFrame.

    Args:
        ratings : DataFrame with [userId, movieId, rating]
    Returns:
        (sparse_matrix, user_to_idx, movie_to_idx)
        - sparse_matrix : scipy csr_matrix of shape (n_users, n_movies)
        - user_to_idx   : dict mapping userId -> row index
        - movie_to_idx  : dict mapping movieId -> col index
    """
    users  = ratings["userId"].unique()
    movies = ratings["movieId"].unique()

    user_to_idx  = {u: i for i, u in enumerate(users)}
    movie_to_idx = {m: i for i, m in enumerate(movies)}

    row = ratings["userId"].map(user_to_idx).values
    col = ratings["movieId"].map(movie_to_idx).values
    data = ratings["rating"].values

    matrix = csr_matrix((data, (row, col)), shape=(len(users), len(movies)))

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"[build_user_item_matrix] Shape: {matrix.shape} | Density: {density:.4%}")
    return matrix, user_to_idx, movie_to_idx


# ─────────────────────────────────────────────────────────────
# 7. SAVE / LOAD HELPERS
# ─────────────────────────────────────────────────────────────

def save_processed(
    ratings_clean: pd.DataFrame,
    movies_clean: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    matrix: csr_matrix,
    user_to_idx: dict,
    movie_to_idx: dict,
    processed_dir: str = "data/processed/",
    splits_dir: str = "data/splits/"
) -> None:
    """Save all processed artifacts to disk."""
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    ratings_clean.to_csv(processed_dir + "ratings_clean.csv", index=False)
    movies_clean.to_csv(processed_dir  + "movies_clean.csv",  index=False)
    train.to_csv(splits_dir + "train.csv", index=False)
    test.to_csv(splits_dir  + "test.csv",  index=False)

    save_npz(processed_dir + "user_item_matrix.npz", matrix)
    joblib.dump({"user_to_idx": user_to_idx, "movie_to_idx": movie_to_idx},
                processed_dir + "index_maps.pkl")

    print("[save_processed] All artifacts saved:")
    print(f"  {processed_dir}ratings_clean.csv")
    print(f"  {processed_dir}movies_clean.csv")
    print(f"  {splits_dir}train.csv")
    print(f"  {splits_dir}test.csv")
    print(f"  {processed_dir}user_item_matrix.npz")
    print(f"  {processed_dir}index_maps.pkl")


def load_processed(
    processed_dir: str = "data/processed/",
    splits_dir: str = "data/splits/"
) -> dict:
    """Load all saved processed artifacts."""
    from scipy.sparse import load_npz

    matrix    = load_npz(processed_dir + "user_item_matrix.npz")
    idx_maps  = joblib.load(processed_dir + "index_maps.pkl")

    return {
        "ratings_clean" : pd.read_csv(processed_dir + "ratings_clean.csv"),
        "movies_clean"  : pd.read_csv(processed_dir + "movies_clean.csv"),
        "train"         : pd.read_csv(splits_dir + "train.csv"),
        "test"          : pd.read_csv(splits_dir + "test.csv"),
        "matrix"        : matrix,
        "user_to_idx"   : idx_maps["user_to_idx"],
        "movie_to_idx"  : idx_maps["movie_to_idx"],
    }


# ─────────────────────────────────────────────────────────────
# 8. FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    raw_dir: str = "data/raw/",
    processed_dir: str = "data/processed/",
    splits_dir: str = "data/splits/",
    sample: int = 2_000_000,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 5,
    test_ratio: float = 0.2
) -> dict:
    """
    Run the full preprocessing pipeline end-to-end.

    Args:
        raw_dir           : path to raw CSV files
        processed_dir     : output path for processed files
        splits_dir        : output path for train/test splits
        sample            : rows to load from ratings (None = all)
        min_user_ratings  : minimum ratings per user to keep
        min_movie_ratings : minimum ratings per movie to keep
        test_ratio        : fraction of ratings held out for test
    Returns:
        dict with all processed artifacts
    """
    print("=" * 50)
    print("  PREPROCESSING PIPELINE")
    print("=" * 50)

    # Load
    ratings = load_ratings(raw_dir + "ratings.csv", sample=sample)
    movies  = load_movies(raw_dir  + "movies.csv")
    tags    = load_tags(raw_dir    + "tags.csv")

    print()

    # Clean
    ratings_clean = clean_ratings(ratings, min_user_ratings, min_movie_ratings)
    movies_clean  = clean_movies(movies)
    tags_clean    = clean_tags(tags)
    tag_profiles  = build_movie_tag_profile(tags_clean)

    # Merge tag profiles into movies
    movies_clean = movies_clean.merge(tag_profiles, on="movieId", how="left")
    movies_clean["tag_profile"] = movies_clean["tag_profile"].fillna("")

    print()

    # Split
    train, test = temporal_train_test_split(ratings_clean, test_ratio)

    print()

    # User-item matrix (from train only)
    matrix, user_to_idx, movie_to_idx = build_user_item_matrix(train)

    print()

    # Save
    save_processed(ratings_clean, movies_clean, train, test,
                   matrix, user_to_idx, movie_to_idx,
                   processed_dir, splits_dir)

    print("\n✓ Pipeline complete.")

    return {
        "ratings_clean" : ratings_clean,
        "movies_clean"  : movies_clean,
        "tags_clean"    : tags_clean,
        "train"         : train,
        "test"          : test,
        "matrix"        : matrix,
        "user_to_idx"   : user_to_idx,
        "movie_to_idx"  : movie_to_idx,
    }


if __name__ == "__main__":
    run_pipeline()