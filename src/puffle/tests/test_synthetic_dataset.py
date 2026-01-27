import pandas as pd
import torch

from puffle.examples.data_preparation.synthetic import (
    SyntheticDataset,
    generate_synthetic_data,
)


class TestSyntheticDataset:
    def test_generate_synthetic_data_shape(self):
        num_samples = 100
        num_features = 5
        df = generate_synthetic_data(num_samples=num_samples, num_features=num_features)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == num_samples
        assert "sensitive_attribute" in df.columns
        assert "target" in df.columns
        assert len(df.columns) == num_features + 2

    def test_synthetic_dataset_items(self):
        df = generate_synthetic_data(num_samples=20)
        dataset = SyntheticDataset(df)

        assert len(dataset) == 20

        # Check first item
        features, sensitive, target, index, idx = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert features.shape == (5,)
        assert isinstance(sensitive, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(index, torch.Tensor)
        assert isinstance(idx, int)

    def test_bias_controlled(self):
        # High bias
        df_biased = generate_synthetic_data(
            num_samples=1000, bias_strength=5.0, seed=42
        )
        corr_biased = df_biased["sensitive_attribute"].corr(df_biased["target"])

        # Zero bias
        df_unbiased = generate_synthetic_data(
            num_samples=1000, bias_strength=0.0, seed=42
        )
        corr_unbiased = df_unbiased["sensitive_attribute"].corr(df_unbiased["target"])

        # Biased should have higher absolute correlation than unbiased
        assert abs(corr_biased) > abs(corr_unbiased)
