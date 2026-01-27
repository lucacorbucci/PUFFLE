# PUFFLE Code Improvement Plan

This plan outlines the specific steps to implement the performance and code quality improvements identified in `report.md`.

## Task 1: Optimizing Metric Calculation (`metric.py`)
Goal: Vectorize `compute_demographic_disparity` to avoid nested Python loops.

- [x] **1.1: Enhance Test Coverage**
    - [x] Create `src/puffle/tests/test_metric_performance.py` (or add to `test_metric.py`) with a large randomized dataset test case to verify correctness stays identical after vectorization.
    - [x] Ensure edge cases (empty groups) are covered.
- [x] **1.2: Refactor `compute_demographic_disparity`**
    - [x] Replace nested loops with `torch` scatter/mask operations.
    - [x] Verify that `statistics` dictionary is still populated correctly (or optionally).
- [x] **1.3: Verification**
    - [x] Run `uv run pytest src/puffle/tests/test_metric.py`
    - [x] Run `uv run ruff check` & `uv run ty check`

## Task 2: Optimizing Loss Calculation (`ErrorRateRegularizationLoss`)
Goal: Vectorize `ErrorRateRegularizationLoss` to remove inefficient `zip` and Python loops.

- [x] **2.1: Enhance Test Coverage**
    - [x] Add specific tests in `src/puffle/tests/test_error_rate_loss.py` for `compute_counters` outputs with exact expected values.
    - [x] Verify `compute_formula_components` output structure matches current expectation.
- [x] **2.2: Refactor `ErrorRateRegularizationLoss`**
    - [x] Rewrite `compute_counters` to use vectorized boolean masking: `((pred==p) & (target==y) & (sensitive==z)).sum()`.
    - [x] Remove `_compute_analysis_dict` if it becomes redundant, or optimize it.
    - [x] Ensure `forward` pass remains differentiable where applicable (though error rate is usually non-differentiable counting, the gradient routing uses the mask).
- [x] **2.3: Verification**
    - [x] Run `uv run pytest src/puffle/tests/test_error_rate_loss.py`
    - [x] Run `uv run ruff check` & `uv run ty check`

## Task 3: Refactoring `PUFFLEModel`
Goal: Improve code readability and reduce training loop overhead.

- [x] **3.1: Introduce `TrainingBatchResult`**
    - [x] Define `class TrainingBatchResult(NamedTuple)` in `puffle_model.py`.
    - [x] Update `_train_batch` to return this NamedTuple.
    - [x] Update `_train_one_epoch` to unpack this NamedTuple.
- [x] **3.2: Optimize Training Loop Data Collection**
    - [x] In `_train_one_epoch`, avoid `cpu().numpy()` on every batch.
    - [x] Collect tensors in a list and `torch.cat` + `cpu().numpy()` once at the end of the epoch.
- [x] **3.3: Metrics Refactoring**
    - [x] Refactor `_initialize_metrics_dict` to use `TypedDict` or just cleaner initialization.
    - [x] Consolidate `update_alpha` and `exp_lr_scheduler` if possible.
- [x] **3.4: Verification**
    - [x] Run `uv run pytest src/puffle/tests/test_fair_model.py` (Update tests to expect NamedTuple).
    - [x] Run integration test `src/puffle/tests/test_synthetic_example.py`.
    - [x] Run full project verification.

## Final Status

All tasks have been completed and verified:
- ✅ `uv run ruff check ./src/puffle/` - All checks passed
- ✅ `uv run ruff format ./src/puffle/` - 40 files unchanged
- ✅ `uv run ty check ./src/puffle/` - All checks passed
- ✅ `uv run pytest src/puffle/tests/` - 74 tests passed
