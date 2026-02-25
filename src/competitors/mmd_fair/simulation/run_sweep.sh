#!/bin/bash
# Run WandB hyperparameter sweep for MMD-Fair on Dutch dataset.
# Phase 1: tune lr, batch_size, num_epochs, momentum with low lambda.

PROJECT_NAME="MMDFairValidation"

run_sweep_and_agent () {
  SWEEP_NAME="$1"
  COUNT="${2:-30}"

  uv run wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "${SWEEP_NAME}.yaml" >temp_output.txt 2>&1

  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

  echo "Running sweep: $SWEEP_ID"
  uv run wandb agent "$SWEEP_ID" --project "$PROJECT_NAME" --count "$COUNT"
}

run_sweep_and_agent "mmd_fair_dutch_sweep"
