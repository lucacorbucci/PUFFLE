# ABOUTME: Demonstrates training a fairness-aware model with Puffle on synthetic data.
# ABOUTME: Compares a standard model with a Puffle-regularized model to show fairness improvement.

import torch
from torch import nn
from torch.utils.data import DataLoader

from puffle.examples.data_preparation.synthetic import (
    SyntheticDataset,
    generate_synthetic_data,
)
from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss


def train_example():
    # 1. Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_samples=1000, bias_strength=1.0)
    dataset = SyntheticDataset(df)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Define simple model
    input_dim = 5
    output_dim = 2
    model = nn.Sequential(
        nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, output_dim)
    )

    # 3. Train Standard Model (no fairness)
    print("\nTraining Standard Model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # We need a custom criterion that handles the PUFFLE tuple format
    # OR we use a simple wrapper for standard training
    class SimplePUFFLELoss(nn.Module):
        def forward(self, inputs, targets):
            outputs, _, _ = inputs
            return nn.CrossEntropyLoss()(outputs, targets)

    puffle_std = PUFFLEModel(
        model=model,
        optimizer=optimizer,
        criterion=SimplePUFFLELoss(),
        lambda_regularization=0.0,
    )

    std_metrics = puffle_std.train(
        train_loader, epochs=5, val_loader=val_loader, verbose=False
    )
    print(
        f"Standard Model - Final Accuracy: {std_metrics['train_accuracy'][-1]:.4f}, Disparity: {std_metrics['train_disparity'][-1]:.4f}"
    )

    # 4. Train Fair Model (with Puffle)
    print("\nTraining Fair Model...")
    # Reset model
    for layer in model.children():
        reset_parameters = getattr(layer, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Use MixLoss to combine accuracy and fairness
    from puffle.Regularization.mix_loss import MixLoss

    fair_criterion = MixLoss(
        model_loss=nn.CrossEntropyLoss(),
        unfairness_loss=DisparityRegularizationLoss(),
        possible_sensitive_attributes=[0, 1],
        possible_targets=[0, 1],
    )

    puffle_fair = PUFFLEModel(
        model=model,
        optimizer=optimizer,
        criterion=fair_criterion,
        lambda_regularization=0.5,
    )

    fair_metrics = puffle_fair.train(
        train_loader,
        epochs=5,
        val_loader=val_loader,
        verbose=False,
    )
    print(
        f"Fair Model - Final Accuracy: {fair_metrics['train_accuracy'][-1]:.4f}, Disparity: {fair_metrics['train_disparity'][-1]:.4f}"
    )


if __name__ == "__main__":
    train_example()
