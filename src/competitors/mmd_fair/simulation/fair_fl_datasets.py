# ABOUTME: COMPAS dataset loader matching Fair-FL reference implementation.
# ABOUTME: Loads and preprocesses COMPAS dataset with client partitioning by age_cat.

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class CompasDataset:
    """
    COMPAS dataset loader matching Fair-FL's preprocessing.

    Dataset: https://www.kaggle.com/danofer/compass
    Partitioning: By age_cat (creates ~3 clients: 25-45, Greater than 45, Less than 25)
    Features: race, c_charge_degree, sex (categorical) + age, priors_count (continuous)
    Target: two_year_recid
    Sensitive attribute: race (African-American vs Caucasian)
    """

    def __init__(self):
        """Initialize COMPAS dataset loader."""
        pass

    def load_data(
        self, homefolder: str = "/"
    ) -> list[tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """
        Load and preprocess COMPAS dataset.

        Matches Fair-FL's preprocessing exactly:
        - Filters data by days_b_screening_arrest, is_recid, c_charge_degree, score_text, race
        - One-hot encodes categorical features
        - Partitions by age_cat

        Args:
            homefolder: Path to Fair-FL root directory containing datasets/

        Returns:
            List of (X, Y, A) tuples, one per client (age_cat group)
            - X: Features (one-hot encoded + continuous)
            - Y: Target (two_year_recid)
            - A: Sensitive attribute (1 if African-American, 0 if Caucasian)
        """
        # Features to use
        continuous_features = ["age", "priors_count"]
        categorical_features = ["race", "c_charge_degree", "sex"]
        label = "two_year_recid"
        sensitive_attribute = "race"
        client_attribute = "age_cat"

        # Load data
        df = pd.read_csv(
            os.path.join(homefolder, "datasets/compas-scores-two-years.csv")
        )

        # Data filtering (matching Fair-FL exactly)
        df = df.dropna(subset=["days_b_screening_arrest"])
        df = df[
            (df["days_b_screening_arrest"] <= 30)
            & (df["days_b_screening_arrest"] >= -30)
        ]
        df = df[df["is_recid"] != -1]
        df = df[df["c_charge_degree"] != "O"]
        df = df[df["score_text"] != "NA"]
        df = df[(df["race"] == "African-American") | (df["race"] == "Caucasian")]
        df = df.reset_index()

        # One-hot encoding
        encoder = OneHotEncoder(handle_unknown="ignore").fit(df[categorical_features])

        # Dataset generation (partition by age_cat)
        datasets = []
        for client in df[client_attribute].unique():
            client_df = df[df[client_attribute] == client]
            X = pd.DataFrame(
                np.hstack(
                    (
                        encoder.transform(client_df[categorical_features]).todense(),
                        client_df[continuous_features],
                    )
                )
            )
            Y = client_df[label]
            A = (client_df[sensitive_attribute] == "African-American") * 1.0
            datasets.append((X, Y, A))

        return datasets
