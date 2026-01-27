# PUFFLE: Precision-based UnFairness Loss Expression

PUFFLE is a library designed to reduce machine learning model unfairness during training through a differentiable regularization term.

## 🚀 Overview

PUFFLE implements a group fairness regularization approach that can be integrated into standard training loops. It focuses on reducing disparities between sensitive groups while maintaining high utility.

Currently supported metrics:
- **Demographic Parity**: Reducing the difference in positive prediction rates between groups.
- **Error Rate Parity**: Reducing the difference in error rates between groups.

## 📦 Project Structure

The project is organized into the `puffle` package:

-   `puffle.PUFFLEModel`: A powerful wrapper for training and evaluation.
-   `puffle.Regularization`: Implementation of various fairness-aware loss functions.
-   `puffle.Utils`: Utility functions for metric computation and data handling.
-   `puffle.examples`: Demonstration scripts and dataset preparation guides.

## 🛠️ Installation & Development

This project uses `uv` for fast, reliable dependency management.

### Setup
```bash
# Install dependencies
uv sync
```

### Running Tests
```bash
# Run all unit tests
uv run pytest src/puffle/tests/
```

### Formatting & Linting
```bash
# Format code
uv run ruff format ./src/

# Lint code
uv run ruff check ./src/
```

## 📖 Usage Example

Here is a quick example of how to use PUFFLE with a standard PyTorch model:

```python
import torch
from torch import nn
from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss
from puffle.Regularization.mix_loss import MixLoss

# 1. Define your model, optimizer, and criterion
model = MyNeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
base_criterion = nn.CrossEntropyLoss()

# 2. Setup PUFFLE fairness regularization
# Use MixLoss to combine your standard loss with fairness regularization
fair_criterion = MixLoss(
    model_loss=base_criterion,
    unfairness_loss=DisparityRegularizationLoss(),
    possible_sensitive_attributes=[0, 1],
    possible_targets=[0, 1]
)

# 3. Wrap with PUFFLEModel
puffle = PUFFLEModel(
    model=model,
    optimizer=optimizer,
    criterion=fair_criterion,
    lambda_regularization=0.5  # Strength of fairness regularization
)

# 4. Train
metrics = puffle.train(
    train_loader=my_dataloader,
    epochs=10,
    val_loader=my_val_loader
)

print(f"Final Disparity: {metrics['train_disparity'][-1]}")
```

## 🧪 Examples

Check the `src/puffle/examples` directory for more detailed examples:
-   `synthetic_example.py`: A complete walkthrough using a biased synthetic dataset.
-   `centralised/`: Examples for standard centralized training.

## 📄 License

[Add License Info]