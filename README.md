movie-recommender/
│
├── data/
│   ├── raw/                       
│   │   ├── ratings.csv
│   │   ├── movies.csv
│   │   ├── tags.csv
│   │   └── links.csv
│   ├── processed/                  # Cleaned & transformed data
│   │   ├── ratings_clean.csv
│   │   ├── movies_clean.csv
│   │   ├── tags_tfidf.npz
│   │   └── user_item_matrix.npz   # Sparse matrix
│   └── splits/                     # Train/test splits
│       ├── train.csv
│       └── test.csv
│
├── notebooks/                      # Jupyter notebooks (exploration & demos)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_collaborative_filtering.ipynb
│   ├── 04_matrix_factorization.ipynb
│   ├── 05_content_based.ipynb
│   └── 06_evaluation.ipynb
│
├── src/                            # Reusable Python modules
│   ├── __init__.py
│   ├── preprocessing.py           # Cleaning, encoding, splitting
│   ├── collaborative_filtering.py # User-based & Item-based CF
│   ├── matrix_factorization.py    # SVD, ALS
│   ├── content_based.py           # Genre + tag similarity
│   ├── hybrid.py                  # blended model
│   └── evaluation.py              # RMSE, MAE, Precision@K, NDCG@K
│
├── models/                         # Saved trained models
│   ├── svd_model.pkl
│   ├── als_model.pkl
│   └── cosine_sim_matrix.npz
│
├── results/                        # Outputs from experiments
│   ├── evaluation_summary.csv     # Model comparison table
│   └── figures/                   # EDA & result plots
│       ├── rating_distribution.png
│       ├── genre_distribution.png
│       └── model_comparison.png
│
├── report/                         # Final report & slides
│   ├── midterm_report.pdf
│   └── slides.pptx
│
├── requirements.txt                # All dependencies
├── README.md                       # Project overview & how to run