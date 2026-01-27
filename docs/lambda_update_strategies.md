# Lambda Update Strategies - Usage Guide

This guide explains how to use the three lambda update strategies for tunable fairness regularization in PUFFLE.

## Available Strategies

### 1. Gradient (Default - Recommended)
**Simple proportional control** - Direct response to unfairness error.

```python
from puffle.PUFFLEModel.puffle_model import PUFFLEModel

model = PUFFLEModel(
    model=your_model,
    optimizer=optimizer,
    criterion=criterion,
    lambda_regularization=0.5,
    target=0.1,  # Target unfairness
    alpha=0.01,  # Learning rate
    tunable_lambda=True,
    lambda_update_strategy="gradient",  # Default
)
```

**Characteristics:**
- ✅ Smooth, predictable updates
- ✅ Fast convergence
- ✅ No oscillations
- ❌ May have small steady-state error

### 2. Momentum (Original Algorithm)
**Velocity-based updates** - Accumulates past gradients for momentum.

```python
model = PUFFLEModel(
    model=your_model,
    optimizer=optimizer,
    criterion=criterion,
    lambda_regularization=0.5,
    target=0.1,
    alpha=0.01,
    momentum=0.9,  # Momentum coefficient
    tunable_lambda=True,
    lambda_update_strategy="momentum",
)
```

**Characteristics:**
- ✅ Can overcome local minima
- ✅ Smooths noisy gradients
- ❌ Can overshoot target
- ❌ May oscillate around equilibrium
- ❌ Slower to stabilize

### 3. PID Controller (New - Best of Both Worlds)
**Proportional-Integral-Derivative control** - Industry-standard control algorithm.

```python
model = PUFFLEModel(
    model=your_model,
    optimizer=optimizer,
    criterion=criterion,
    lambda_regularization=0.5,
    target=0.1,
    tunable_lambda=True,
    lambda_update_strategy="pid",
    # PID gains (tune these for your problem)
    lambda_kp=0.01,  # Proportional gain - responsiveness
    lambda_ki=0.001,  # Integral gain - eliminates steady-state error
    lambda_kd=0.005,  # Derivative gain - dampens oscillations
)
```

**Characteristics:**
- ✅ Fast response to changes (P term)
- ✅ Eliminates steady-state error (I term)
- ✅ Dampens oscillations (D term)
- ✅ Best balance of speed and stability
- ⚠️ Requires tuning PID gains for optimal performance

## Comparison Example

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss

# Setup
model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = DisparityRegularizationLoss()

# Experiment 1: Gradient (smooth, stable)
puffle_gradient = PUFFLEModel(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    lambda_regularization=0.5,
    target=0.1,
    alpha=0.01,
    tunable_lambda=True,
    lambda_update_strategy="gradient",
    wandb_run=wandb.init(name="gradient_strategy"),
)

# Experiment 2: Momentum (may oscillate)
puffle_momentum = PUFFLEModel(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    lambda_regularization=0.5,
    target=0.1,
    alpha=0.01,
    momentum=0.9,
    tunable_lambda=True,
    lambda_update_strategy="momentum",
    wandb_run=wandb.init(name="momentum_strategy"),
)

# Experiment 3: PID (best performance)
puffle_pid = PUFFLEModel(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    lambda_regularization=0.5,
    target=0.1,
    tunable_lambda=True,
    lambda_update_strategy="pid",
    lambda_kp=0.01,
    lambda_ki=0.001,
    lambda_kd=0.005,
    wandb_run=wandb.init(name="pid_strategy"),
)

# Train and compare
for puffle in [puffle_gradient, puffle_momentum, puffle_pid]:
    puffle.train(train_loader, epochs=50)
```

## Tuning PID Parameters

### Quick Start Values
- **Aggressive**: `kp=0.05, ki=0.005, kd=0.01` - Fast response, may overshoot
- **Balanced**: `kp=0.01, ki=0.001, kd=0.005` - Default, good for most cases
- **Conservative**: `kp=0.005, ki=0.0005, kd=0.002` - Slow but very stable

### Manual Tuning Process
1. **Start with P only**: Set `ki=0, kd=0`, tune `kp` until response is reasonable
2. **Add I term**: Increase `ki` gradually to eliminate steady-state error
3. **Add D term**: Increase `kd` to reduce oscillations

### Observing Behavior in WandB
- **Lambda oscillates**: Reduce `kp`, increase `kd`
- **Slow convergence**: Increase `kp`
- **Steady-state error**: Increase `ki`
- **Overshoot**: Reduce `kp`, increase `kd`

## Algorithm Details

### Gradient Update
```
λ[t+1] = λ[t] + α * (unfairness - target)
```

### Momentum Update
```
v[t] = momentum * v[t-1] + (target - unfairness)
λ[t+1] = λ[t] - α * v[t]
```

### PID Update
```
error = unfairness - target
P = kp * error
I = ki * Σ(error)
D = kd * (error - error_prev)
λ[t+1] = λ[t] + P + I + D
```

All strategies constrain λ ∈ [0, 1].

## Research Paper Recommendations

For your ablation study, we recommend:

1. **Baseline**: Momentum (original algorithm)
2. **Improved**: Gradient (simpler, more stable)
3. **Advanced**: PID (best performance with tuning)

Compare on metrics:
- Lambda trajectory smoothness (std dev of lambda over epochs)
- Convergence speed (epochs to reach target ± ε)
- Steady-state error (|unfairness - target| at convergence)
- Fairness-accuracy tradeoff (Pareto frontier)
