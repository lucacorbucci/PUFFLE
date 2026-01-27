# Lambda Update Strategy Comparison Experiment

This experiment compares three lambda update strategies under distribution shift scenarios:

1. **Momentum** - Original velocity-based algorithm
2. **Gradient** - Simple proportional control (current default)
3. **PID** - Proportional-Integral-Derivative controller

## Experiment Design

### Distribution Shift Scenario

- **Training**: 20 epochs total
- **Distribution Shift**: Injected at epoch 10
- **Shift Type**: Increase bias (makes sensitive attribute more predictive of label)
- **Shift Magnitude**: 0.3 (30% of training data affected)

### Metrics Tracked

- Lambda trajectory over time
- Adaptation speed after distribution shift
- Steady-state error
- Fairness-accuracy tradeoff
- Overshoot and oscillation

## Running the Experiment

### Option 1: Hyperparameter Tuning (Recommended)

Run WandB sweeps to find optimal hyperparameters for each strategy:

```bash
# Momentum strategy
cd src/puffle/examples/centralised/lambda_strategy_comparison
wandb sweep momentum_strategy.yaml
wandb agent <sweep-id>

# Gradient strategy
wandb sweep gradient_strategy.yaml
wandb agent <sweep-id>

# PID strategy
wandb sweep pid_strategy.yaml
wandb agent <sweep-id>
```

### Option 2: Quick Test with Default Parameters

```bash
# Test momentum strategy
uv run python main.py \
  --lr 0.05 \
  --epochs 20 \
  --batch_size 732 \
  --optimizer adam \
  --project_name LambdaStrategyTest \
  --target 0.05 \
  --lambda_update_strategy momentum \
  --momentum 0.9 \
  --alpha 0.5 \
  --shift_epoch 10 \
  --shift_type increase_bias \
  --shift_magnitude 0.3 \
  --csv_path ../../data/dutch/

# Test gradient strategy
uv run python main.py \
  --lr 0.05 \
  --epochs 20 \
  --batch_size 732 \
  --optimizer adam \
  --project_name LambdaStrategyTest \
  --target 0.05 \
  --lambda_update_strategy gradient \
  --alpha 0.01 \
  --shift_epoch 10 \
  --shift_type increase_bias \
  --shift_magnitude 0.3 \
  --csv_path ../../data/dutch/

# Test PID strategy
uv run python main.py \
  --lr 0.05 \
  --epochs 20 \
  --batch_size 732 \
  --optimizer adam \
  --project_name LambdaStrategyTest \
  --target 0.05 \
  --lambda_update_strategy pid \
  --lambda_kp 0.01 \
  --lambda_ki 0.001 \
  --lambda_kd 0.005 \
  --shift_epoch 10 \
  --shift_type increase_bias \
  --shift_magnitude 0.3 \
  --csv_path ../../data/dutch/
```

### Option 3: No Distribution Shift (Baseline)

Omit `--shift_epoch` to run without distribution shift:

```bash
uv run python main.py \
  --lr 0.05 \
  --epochs 20 \
  --batch_size 732 \
  --optimizer adam \
  --project_name LambdaStrategyBaseline \
  --target 0.05 \
  --lambda_update_strategy gradient \
  --alpha 0.01 \
  --csv_path ../../data/dutch/
```

## Configuration Files

### momentum_strategy.yaml

Tunes:
- `momentum` (0.1 to 0.99)
- `alpha` (0.001 to 2.0)
- `weight_decay_alpha` (0.5 to 0.99)
- Plus standard training hyperparameters

### gradient_strategy.yaml

Tunes:
- `alpha` (0.001 to 2.0)
- Plus standard training hyperparameters

### pid_strategy.yaml

Tunes:
- `lambda_kp` (0.001 to 0.1) - Proportional gain
- `lambda_ki` (0.0001 to 0.01) - Integral gain
- `lambda_kd` (0.001 to 0.05) - Derivative gain
- Plus standard training hyperparameters

## Expected Results

### Momentum Strategy
- **Pros**: Fast initial response
- **Cons**: May overshoot and oscillate after distribution shift
- **Expected**: Higher variance in lambda trajectory

### Gradient Strategy
- **Pros**: Stable, predictable
- **Cons**: Slower adaptation to distribution shift
- **Expected**: Smooth lambda trajectory, moderate adaptation speed

### PID Strategy
- **Pros**: Fast adaptation + minimal overshoot + zero steady-state error
- **Cons**: Requires tuning 3 parameters
- **Expected**: Best overall performance with proper tuning

## Analyzing Results in WandB

1. **Lambda Trajectory**: Plot `lambda_regularization` over epochs
   - Look for smoothness, overshoot, oscillations
   - Mark epoch 10 (distribution shift point)

2. **Adaptation Metrics**:
   - Time to re-converge after shift
   - Maximum deviation from target after shift
   - Final steady-state error

3. **Fairness-Accuracy Tradeoff**:
   - Plot test accuracy vs test disparity
   - Compare Pareto frontiers across strategies

4. **Statistical Comparison**:
   - Run multiple seeds for each strategy
   - Compare mean ± std for key metrics

## Distribution Shift Types

You can experiment with different shift types:

- `increase_bias`: Makes sensitive attribute more predictive (default)
- `decrease_bias`: Makes sensitive attribute less predictive
- `flip_labels`: Flips labels to create sudden accuracy drop

Example:
```bash
--shift_type decrease_bias --shift_magnitude 0.5
```

## For Research Paper

This experiment provides empirical evidence for:

1. **Robustness to distribution shift**: Which strategy adapts best?
2. **Hyperparameter sensitivity**: How much tuning does each need?
3. **Practical recommendations**: When to use which strategy?

Include in paper:
- Lambda trajectories (Figure)
- Adaptation metrics table
- Statistical significance tests
- Ablation study on shift magnitude
