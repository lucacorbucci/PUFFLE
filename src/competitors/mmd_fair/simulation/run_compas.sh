#!/bin/bash
# Run COMPAS experiment matching Fair-FL's configuration.
# Results saved to a separate directory for side-by-side comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results/ours"

uv run python -m competitors.mmd_fair.simulation.fair_fl_experiment \
  --method ours \
  --home /home/lcorbucci/Fair-FL \
  --output "${OUTPUT_DIR}" \
  --numSeeds 10 \
  --numComRnds 100 \
  --numLambdas 50 \
  --runName "PUFFLE_MMDFair_COMPAS" \
  --dataset compas
