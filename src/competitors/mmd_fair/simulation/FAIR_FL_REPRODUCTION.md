# Fair-FL Experiment Reproduction - Implementation Summary

## What Was Implemented

I've successfully implemented a standalone experiment runner that reproduces the Fair-FL paper's COMPAS experiments using PUFFLE's MMD-Fair implementation. This allows us to verify the correctness of PUFFLE's MMD-Fair code against the original Fair-FL reference.

## Files Created

### 1. Model Architectures (`fair_fl_models.py`)
- **TwoLayerNN**: 2-layer neural network (Linear→ReLU→Linear) matching Fair-FL's architecture for COMPAS
  - Input size: 8 features (after one-hot encoding)
  - Hidden size: 16
  - Uses `bias=True` (matching Fair-FL exactly)
- **LogisticModel**: Logistic regression model for future Synthetic dataset experiments

### 2. Dataset Loader (`fair_fl_datasets.py`)
- **CompasDataset**: Loads and preprocesses COMPAS dataset exactly as Fair-FL does
  - Filters by days_b_screening_arrest, is_recid, c_charge_degree, score_text, race
  - One-hot encodes categorical features (race, c_charge_degree, sex)
  - Partitions by age_cat (creates ~3 clients)
  - Returns list of (X, Y, A) tuples

### 3. Experiment Runner (`fair_fl_experiment.py`)
- **FairFLClient**: Client abstraction using PUFFLE's `MMDFairModel` for local training
  - Implements `client_step()`, `sample_C_update()`, `set_C()`, `split_train_test()`
  - Uses PUFFLE's `_train_batch()` method with MMD fairness penalty
- **FairFLServer**: Server abstraction matching Fair-FL's training loop
  - Implements `train()`, `update_C()`, `aggregate_theta()`, `test_current_model()`
  - Uses PUFFLE's `PredictionTracker` for Y_0/Y_1 sets
  - Logs accuracy and P1 fairness to wandb
- **Metrics**: `accuracy()` and `P1()` functions matching Fair-FL exactly
- **Main function**: Runs parameter sweep (lambda, seeds) matching Fair-FL's `run_compas.py`

### 4. Run Scripts
- **run_compas.sh**: Full COMPAS experiment (10 seeds, 100 rounds, 50 lambda values)
- **test_run_compas.sh**: Quick test run (1 seed, 2 rounds, 2 lambda values)

### 5. Tests (`test_fair_fl_experiment.py`)
- Model architecture tests (TwoLayerNN, LogisticModel)
- Dataset loading tests (COMPAS)
- Metrics tests (accuracy, P1)
- Client/server initialization tests
- End-to-end simulation test (marked as slow)

## How It Works

The implementation **bypasses Flower's simulation framework** and directly implements the Fair-FL training loop, but uses PUFFLE's `MMDFairModel` for the core training logic. This design choice ensures:

1. **Exact data splits**: Uses Fair-FL's dataset loading, so data is partitioned identically
2. **Exact hyperparameters**: Matches Fair-FL's learning rates, batch sizes, epochs, etc.
3. **Exact metrics**: Uses Fair-FL's accuracy and P1 calculations
4. **PUFFLE's core logic**: Uses PUFFLE's `MMDFairModel._train_batch()` and `PredictionTracker`

This allows us to verify that PUFFLE's MMD-Fair implementation produces the same results as Fair-FL's reference implementation when given the same data and hyperparameters.

## How to Run

### Quick Test (2 rounds, 2 lambda values, 1 seed)
```bash
cd /home/lcorbucci/PUFFLE/src/competitors/mmd_fair/simulation
./test_run_compas.sh
```

### Full COMPAS Experiment (100 rounds, 50 lambda values, 10 seeds)
```bash
cd /home/lcorbucci/PUFFLE/src/competitors/mmd_fair/simulation
./run_compas.sh
```

**Note**: The full experiment will take a long time (10 seeds × 50 lambda values × 100 rounds = 50,000 training rounds total).

### Run Tests
```bash
cd /home/lcorbucci/PUFFLE/src
uv run pytest competitors/mmd_fair/tests/test_fair_fl_experiment.py -v
```

## Results

Results are saved to `/home/lcorbucci/Fair-FL/results/ours/` as `.npy` files with the format:
```
compas_p_{lambda}_{seed}_{NY}.npy
```

Each file contains a numpy array of shape `(num_clients, 2)` with:
- Column 0: Accuracy per client
- Column 1: P1 fairness per client

Metrics are also logged to Weights & Biases under project `fairFL` with run name `PUFFLE_MMDFair_COMPAS`.

## Verification

To verify correctness, you should:

1. **Run the quick test** to ensure everything works
2. **Compare wandb logs** between Fair-FL's original code and PUFFLE's reproduction
3. **Compare final metrics** (accuracy vs P1 tradeoff curves)

The metrics should match closely (within noise tolerance) if the implementation is correct.

## Code Quality

- ✅ All tests pass
- ✅ Code formatted with `ruff format`
- ✅ No linting errors (`ruff check`)
- ✅ Type-checked with `ty check`
- ✅ Follows PUFFLE's coding conventions

## Next Steps

1. Run the quick test to verify the implementation works end-to-end
2. If the test passes, run the full COMPAS experiment
3. Compare results against Fair-FL's reference implementation
4. If results match, the PUFFLE MMD-Fair implementation is verified for COMPAS
5. (Optional) Extend to Synthetic and Communities & Crime datasets
