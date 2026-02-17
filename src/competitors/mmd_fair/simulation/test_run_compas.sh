#!/bin/bash
# Quick test run with minimal parameters to verify the experiment runner works

uv run python -m competitors.mmd_fair.simulation.fair_fl_experiment \
  --method ours \
  --home /home/lcorbucci/Fair-FL \
  --numSeeds 1 \
  --numComRnds 2 \
  --numLambdas 2 \
  --runName "PUFFLE_MMDFair_COMPAS_Test" \
  --dataset compas
