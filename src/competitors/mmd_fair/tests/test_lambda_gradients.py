import copy

import torch
import torch.nn as nn
from puffle.Utils.config import PUFFLEConfig

from competitors.mmd_fair.model import MMDFairModel


def test_lambda_affects_gradients():
    # Setup simple model, inputs, and tracking sets
    torch.manual_seed(42)
    model = nn.Linear(5, 1)
    # create two copies to compare gradients
    model_zero = copy.deepcopy(model)
    model_large = copy.deepcopy(model)
    
    x = torch.randn(10, 5)
    z = torch.tensor([0,0,0,0,0, 1,1,1,1,1])
    y = torch.tensor([1,1,1,1,1, 0,0,0,0,0], dtype=torch.float32)
    
    # Tracking sets (just random numbers like server would broadcast)
    # The fix ensures these are in probability space [0, 1]
    # We use disjoint sets to guarantee the penalty is non-zero
    y_0 = torch.zeros(100)
    y_1 = torch.ones(100)

    # 1. Train with lambda=0
    optimizer_zero = torch.optim.SGD(model_zero.parameters(), lr=0.1)
    mmd_zero = MMDFairModel(
        model=model_zero,
        optimizer=optimizer_zero,
        criterion=nn.BCEWithLogitsLoss(),
        device='cpu',
        config=PUFFLEConfig(lambda_regularization=0.0)
    )
    mmd_zero.set_tracking_function(y_0, y_1)
    mmd_zero.set_server_predictions(y_0, y_1)
    mmd_zero.set_client_weights(alpha_0=1.0, alpha_1=1.0)
    mmd_zero._train_batch((x, z, y), model_zero, optimizer_zero, nn.BCEWithLogitsLoss())
    
    grad_zero = model_zero.weight.grad.clone()

    # 2. Train with lambda=100
    optimizer_large = torch.optim.SGD(model_large.parameters(), lr=0.1)
    mmd_large = MMDFairModel(
        model=model_large,
        optimizer=optimizer_large,
        criterion=nn.BCEWithLogitsLoss(),
        device='cpu',
        config=PUFFLEConfig(lambda_regularization=0.0)
    )
    mmd_large.lambda_regularization = 100.0  # Set high lambda
    mmd_large.set_tracking_function(y_0, y_1)
    mmd_large.set_server_predictions(y_0, y_1)
    mmd_large.set_client_weights(alpha_0=1.0, alpha_1=1.0)
    mmd_large._train_batch((x, z, y), model_large, optimizer_large, nn.BCEWithLogitsLoss())
    
    grad_large = model_large.weight.grad.clone()

    # The gradients MUST be different if lambda has an effect
    assert not torch.allclose(grad_zero, grad_large), "Lambda penalty did not affect gradients!"
