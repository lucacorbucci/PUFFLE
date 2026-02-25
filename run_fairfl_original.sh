#!/bin/bash
# Run the original Fair-FL script with full sweeps to reproduce their plot
# This will take several hours (approx 500 runs at ~1 min/run)

uv run python Fair-FL/run_compas.py \
  --method ours \
  --home /home/lcorbucci/PUFFLE/Fair-FL \
  --numSeeds 10 \
  --numComRnds 100 \
  --numLambdas 50 \
  --runName "FairFL_Original_COMPAS"
