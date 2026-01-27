# Lambda Update Strategy Comparison Experiment

This experiment compares three lambda update strategies under distribution shift scenarios:

1. **Momentum** - Original velocity-based algorithm
2. **Gradient** - Simple proportional control (current default)
3. **PID** - Proportional-Integral-Derivative controller

## Experiment Design

### Distribution Shift Using Pre-Generated Datasets

Instead of injecting shifts at runtime, this experiment uses **pre-generated CSV datasets** for maximum control and reproducibility.

**Workflow:**
1. Generate shifted datasets using `generate_shifted_datasets.py`
2. Train on original dataset for N epochs
3. Switch to shifted dataset and continue training
4. Compare how each strategy adapts

## Step 1: Generate Shifted Datasets

First, create datasets with distribution shifts:

```bash
cd src/puffle/examples

# Generate dataset with increased bias
uv run python generate_shifted_datasets.py \
  --input_path ../data/dutch/ \
  --output_path ../data/dutch_shifted/ \
  --shift_type increase_bias \
  --shift_magnitude 0.3 \
  --seed 42
```

This creates:
- `dutch_shifted/dutch_census_2001_original.csv` - Copy of original
- `dutch_shifted/dutch_census_2001_shifted_increase_bias_0.3.csv` - Shifted version
- `dutch_shifted/shift_metadata_increase_bias_0.3.json` - Metrics and metadata

**Shift Types:**
- `increase_bias` - Makes sensitive attribute more predictive of label
- `decrease_bias` - Makes sensitive attribute less predictive
- `flip_labels` - Flips labels to create accuracy drop
- `class_imbalance` - Creates class imbalance

## Step 2: Run Experiments

### Option 1: Quick Test

```bash
cd src/puffle/examples/centralised/lambda_strategy_comparison

# Test gradient strategy with distribution shift
uv run python main.py \
  --lr 0.05 \
  --epochs_before_shift 10 \
  --epochs_after_shift 10 \
  --batch_size 732 \
  --optimizer adam \
  --project_name LambdaStrategyTest \
  --run_name "gradient_with_shift" \
  --target 0.05 \
  --lambda_update_strategy gradient \
  --alpha 0.01 \
  --csv_path_before ../../data/dutch_shifted/ \
  --csv_path_after ../../data/dutch_shifted/ \
  --dataset_name_before dutch_census_2001_original.csv \
  --dataset_name_after dutch_census_2001_shifted_increase_bias_0.3.csv
```

### Option 2: Hyperparameter Tuning

Update the YAML files with your dataset paths, then run:

```bash
wandb sweep momentum_strategy.yaml
wandb agent <sweep-id>
```

## Configuration Files

The YAML files need to be updated with your specific dataset paths. Example:

```yaml
command:
  - ${env}
  - uv 
  - run 
  - python
  - ${program}
  - ${args}
  - --project_name 
  - LambdaStrategyComparison
  - --target
  - "0.05"
  - --epochs_before_shift
  - "10"
  - --epochs_after_shift
  - "10"
  - --lambda_update_strategy
  - gradient
  - --csv_path_before
  - ../../data/dutch_shifted/
  - --csv_path_after
  - ../../data/dutch_shifted/
  - --dataset_name_before
  - dutch_census_2001_original.csv
  - --dataset_name_after
  - dutch_census_2001_shifted_increase_bias_0.3.csv
```

## Command Line Arguments

### Required Arguments

- `--lr` - Learning rate
- `--epochs_before_shift` - Epochs on original dataset
- `--epochs_after_shift` - Epochs on shifted dataset
- `--batch_size` - Batch size
- `--optimizer` - Optimizer (adam/sgd)
- `--project_name` - WandB project name
- `--target` - Target unfairness value
- `--lambda_update_strategy` - Strategy (momentum/gradient/pid)
- `--csv_path_before` - Path to original dataset directory
- `--csv_path_after` - Path to shifted dataset directory (optional, omit for no shift)
- `--dataset_name_before` - Filename of original dataset
- `--dataset_name_after` - Filename of shifted dataset (optional)

### Strategy-Specific Parameters

**Momentum:**
- `--momentum` (default: 0.9)
- `--alpha` (default: 0.01)
- `--weight_decay_alpha` (default: 0.99)

**Gradient:**
- `--alpha` (default: 0.01)

**PID:**
- `--lambda_kp` (default: 0.01) - Proportional gain
- `--lambda_ki` (default: 0.001) - Integral gain
- `--lambda_kd` (default: 0.005) - Derivative gain

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
   - Mark the shift point (end of `epochs_before_shift`)

2. **Adaptation Metrics**:
   - Time to re-converge after shift
   - Maximum deviation from target after shift
   - Final steady-state error

3. **Fairness-Accuracy Tradeoff**:
   - Plot test accuracy vs test disparity
   - Compare Pareto frontiers across strategies

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
