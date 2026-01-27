import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from puffle.examples.synthetic.synthetic_dataset import (
    SyntheticDataset,
    generate_synthetic_data,
)
from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss
from puffle.Regularization.mix_loss import MixLoss


def train_example():
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # 1. Generate synthetic data
    print("Generating synthetic data...")

    df = generate_synthetic_data(num_samples=2000, bias_strength=5.0, seed=seed)
    dataset = SyntheticDataset(df)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Define simple model
    input_dim = 5
    output_dim = 2

    def create_model():
        return nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, output_dim)
        )

    # 3. Train Standard Model (no fairness)
    print("\nTraining Standard Model...")
    model_std = create_model()
    optimizer_std = torch.optim.Adam(model_std.parameters(), lr=0.01)

    # Simple wrapper for standard CrossEntropyLoss to match PUFFLE signature
    class StandardLoss(nn.Module):
        def forward(self, inputs, targets):
            outputs, _, _ = inputs
            return nn.CrossEntropyLoss()(outputs, targets)

    puffle_std = PUFFLEModel(
        model=model_std,
        optimizer=optimizer_std,
        criterion=StandardLoss(),
        lambda_regularization=0.0,
    )

    std_metrics = puffle_std.train(
        train_loader, epochs=10, val_loader=val_loader, verbose=False
    )

    final_acc_std = std_metrics["val_accuracy"][-1]
    final_disp_std = std_metrics["val_disparity"][-1]
    print(
        f"Standard Model - Accuracy: {final_acc_std:.4f}, Disparity: {final_disp_std:.4f}"
    )

    # 4. Train Fair Model (with Puffle)
    print("\nTraining Fair Model...")
    model_fair = create_model()
    optimizer_fair = torch.optim.Adam(model_fair.parameters(), lr=0.01)

    fair_criterion = MixLoss(
        model_loss=nn.CrossEntropyLoss(),
        unfairness_loss=DisparityRegularizationLoss(),
        possible_sensitive_attributes=[0, 1],
        possible_targets=[0, 1],
    )

    puffle_fair = PUFFLEModel(
        model=model_fair,
        optimizer=optimizer_fair,
        criterion=fair_criterion,
        lambda_regularization=0.8,
        tunable_lambda=False,
    )

    fair_metrics = puffle_fair.train(
        train_loader,
        epochs=10,
        val_loader=val_loader,
        verbose=False,
    )

    final_acc_fair = fair_metrics["val_accuracy"][-1]
    final_disp_fair = fair_metrics["val_disparity"][-1]
    print(
        f"Fair Model     - Accuracy: {final_acc_fair:.4f}, Disparity: {final_disp_fair:.4f}"
    )

    return std_metrics, fair_metrics


if __name__ == "__main__":
    train_example()
