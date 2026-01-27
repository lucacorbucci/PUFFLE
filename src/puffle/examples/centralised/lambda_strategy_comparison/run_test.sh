#!/bin/bash

# Quick test of all three lambda update strategies with distribution shift

echo "Testing Momentum Strategy..."
uv run python main.py \
  --lr 0.05 \
  --epochs 20 \
  --batch_size 732 \
  --optimizer adam \
  --project_name LambdaStrategyTest \
  --run_name "momentum_shift" \
  --target 0.05 \
  --lambda_update_strategy momentum \
  --momentum 0.85 \
  --alpha 1.0 \
  --shift_epoch 10 \
  --shift_type increase_bias \
  --shift_magnitude 0.3 \
  --csv_path ../../data/dutch/

echo "Testing Gradient Strategy..."
uv run python main.py \
  --lr 0.05 \
  --epochs 20 \
  --batch_size 732 \
  --optimizer adam \
  --project_name LambdaStrategyTest \
  --run_name "gradient_shift" \
  --target 0.05 \
  --lambda_update_strategy gradient \
  --alpha 0.01 \
  --shift_epoch 10 \
  --shift_type increase_bias \
  --shift_magnitude 0.3 \
  --csv_path ../../data/dutch/

echo "Testing PID Strategy..."
uv run python main.py \
  --lr 0.05 \
  --epochs 20 \
  --batch_size 732 \
  --optimizer adam \
  --project_name LambdaStrategyTest \
  --run_name "pid_shift" \
  --target 0.05 \
  --lambda_update_strategy pid \
  --lambda_kp 0.01 \
  --lambda_ki 0.001 \
  --lambda_kd 0.005 \
  --shift_epoch 10 \
  --shift_type increase_bias \
  --shift_magnitude 0.3 \
  --csv_path ../../data/dutch/

echo "All tests complete! Check WandB project 'LambdaStrategyTest' for results."
