#!/bin/bash
# Phase 1: tune lr, batch_size, num_epochs, momentum with low lambda.
# Phase 2: sweep lambda to generate the Pareto frontier.

PROJECT_NAME="MMDFairValidation"
COUNT=100

run_sweep_and_agent () {
  SWEEP_NAME="$1"

  uv run wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "${SWEEP_NAME}.yaml" >temp_output.txt 2>&1

  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

  echo "Running sweep: $SWEEP_ID"
  uv run wandb agent "$SWEEP_ID" --project "$PROJECT_NAME" --count "$COUNT"
}

# Phase 1 — baseline hyperparameter tuning (low lambda)
# run_sweep_and_agent "mmd_fair_dutch_sweep"

# Phase 2 — lambda sweep for Pareto frontier (grid, fixed hyperparameters)
# run_sweep_and_agent "mmd_fair_dutch_lambda_sweep"

# Phase 3 — joint sweep for Pareto frontier (random search over all hyperparams + lambda)
run_sweep_and_agent "mmd_fair_dutch_joint_sweep"
