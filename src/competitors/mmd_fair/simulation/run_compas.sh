#!/bin/bash
# Run COMPAS experiment matching Fair-FL's configuration

uv run python -m competitors.mmd_fair.simulation.fair_fl_experiment \
  --method ours \
  --home /home/lcorbucci/Fair-FL \
  --numSeeds 10 \
  --numComRnds 100 \
  --numLambdas 50 \
  --runName "PUFFLE_MMDFair_COMPAS" \
  --dataset compas
