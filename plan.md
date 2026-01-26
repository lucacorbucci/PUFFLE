# PUFFLE Library Improvement Plan

This plan focuses on improving the `./src/puffle/` folder. Tasks are ordered by dependency and priority.

---

## Phase 1: Code Quality & Static Analysis

For these, do not add new ignored rules to obtain zero errors please.

### Task 1.1: Fix Ruff Errors
**Goal**: Ensure all code passes `ruff check` and `ruff format`. **Iterate until zero errors.**

- [ ] **1.1.1** Run `uv run ruff check ./src/puffle/` and capture output
- [ ] **1.1.2** Fix linting errors in `PUFFLEModel/puffle_model.py`
- [ ] **1.1.3** Fix linting errors in `Regularization/disparity_loss.py`
- [ ] **1.1.4** Fix linting errors in `Regularization/ErrorRateRegularizationLoss.py`
- [ ] **1.1.5** Fix linting errors in `Regularization/mix_loss.py`
- [ ] **1.1.6** Fix linting errors in `Utils/` directory files
- [ ] **1.1.7** Run `uv run ruff format ./src/puffle/` to auto-format
- [ ] **1.1.8** Verify no remaining errors with `uv run ruff check ./src/puffle/`

### Task 1.2: Fix Type Errors (ty)
**Goal**: Ensure all code passes `ty check`. **Iterate until zero errors.**

- [ ] **1.2.1** Run `uv run ty check ./src/puffle/` and capture output
- [ ] **1.2.2** Add/fix type annotations in `PUFFLEModel/puffle_model.py`
- [ ] **1.2.3** Add/fix type annotations in `Regularization/` files
- [ ] **1.2.4** Add/fix type annotations in `Utils/` files
- [ ] **1.2.5** Verify no remaining errors with `uv run ty check ./src/puffle/`

### Task 1.3: Reduce Cognitive Complexity (complexipy)
**Goal**: Ensure functions are short and cognitive load is manageable. **Iterate until zero high-complexity warnings.**

- [ ] **1.3.1** Run `uv run complexipy ./src/puffle/` and capture output
- [ ] **1.3.2** Identify functions with high complexity scores
- [ ] **1.3.3** Refactor `DisparityRegularizationLoss.forward()` (likely candidate)
- [ ] **1.3.4** Refactor any other flagged functions by extracting helper methods
- [ ] **1.3.5** Verify reduced complexity with `uv run complexipy ./src/puffle/`

---

## Phase 2: Test Infrastructure

### Task 2.1: Fix Existing Test Imports
**Goal**: Make existing tests runnable by fixing broken imports.

- [ ] **2.1.1** Update imports in `tests/test_fair_model.py`:
    - Change `from puffle.FairModel.fair_model import PUFFLEModel` to `from puffle.PUFFLEModel.puffle_model import PUFFLEModel`
    - Change `from puffle.FairReg.Regularization.RegularizationLoss import ...` to correct path
- [ ] **2.1.2** Update imports in `tests/test_metric.py`:
    - Change `from puffle.FairReg.Utils.metric import ...` to `from puffle.Utils.metric import ...`
- [ ] **2.1.3** Update imports in `tests/test_regularization_loss.py`:
    - Change `from FairReg.RegularizationLoss import RegularizationLoss` to `from puffle.Regularization.disparity_loss import DisparityRegularizationLoss`
- [ ] **2.1.4** Fix API mismatches (e.g., `binary_sensitive_value` parameter removed)
- [ ] **2.1.5** Run `uv run pytest ./src/puffle/tests/ -v` and verify all tests pass

### Task 2.2: Add Comprehensive Test Coverage
**Goal**: Add tests for all functions not yet covered. **Target: 90% test coverage minimum.**

- [ ] **2.2.1** Audit `PUFFLEModel/puffle_model.py` and list untested methods
- [ ] **2.2.2** Add tests for `PUFFLEModel._train_one_epoch()`
- [ ] **2.2.3** Add tests for `PUFFLEModel._train_batch()`
- [ ] **2.2.4** Add tests for `PUFFLEModel.update_lambda()`
- [ ] **2.2.5** Add tests for `PUFFLEModel.exp_lr_scheduler()`
- [ ] **2.2.6** Add tests for `PUFFLEModel.update_alpha()`
- [ ] **2.2.7** Audit `Regularization/` files and add tests for any untested functions
- [ ] **2.2.8** Audit `Utils/` files and add tests for untested functions
- [ ] **2.2.9** Run full test suite to confirm coverage

---

## Phase 3: Synthetic Dataset & Integration Test

### Task 3.1: Create Synthetic Unfair Dataset
**Goal**: Provide a lightweight, fast-running example for testing fairness reduction.

- [ ] **3.1.1** Create `src/puffle/examples/synthetic/synthetic_dataset.py`:
    - Generate a small dataset (e.g., 1000 samples)
    - Include a binary sensitive attribute `z` (e.g., 0/1)
    - Include a binary target `y` correlated with `z` to create unfairness
    - Include simple features `X`
- [ ] **3.1.2** Create `src/puffle/examples/synthetic/main.py`:
    - Load synthetic dataset
    - Train a simple linear model with `PUFFLEModel`
    - Demonstrate disparity reduction
- [ ] **3.1.3** Create `src/puffle/tests/test_synthetic_example.py`:
    - Test that training on synthetic data reduces disparity
    - Test that accuracy remains reasonable
    - Test that tunable lambda mode works correctly

---

## Phase 4: Documentation

### Task 4.1: Add ABOUTME Comments
**Goal**: All source files start with a 2-line ABOUTME comment.

- [ ] **4.1.1** Add ABOUTME to `puffle/__init__.py`
- [ ] **4.1.2** Add ABOUTME to `PUFFLEModel/puffle_model.py`
- [ ] **4.1.3** Add ABOUTME to `Regularization/disparity_loss.py`
- [ ] **4.1.4** Add ABOUTME to `Regularization/ErrorRateRegularizationLoss.py`
- [ ] **4.1.5** Add ABOUTME to `Regularization/mix_loss.py`
- [ ] **4.1.6** Add ABOUTME to `Utils/metric.py`
- [ ] **4.1.7** Add ABOUTME to `Utils/regularization_config.py`
- [ ] **4.1.8** Add ABOUTME to `Utils/tabular_datasets_utils.py`
- [ ] **4.1.9** Add ABOUTME to `Utils/utils.py`
- [ ] **4.1.10** Add ABOUTME to all files in `examples/` subdirectories

### Task 4.2: Document Functions & Fix Comment Inconsistencies
**Goal**: Ensure all public functions have accurate docstrings.

- [ ] **4.2.1** Review and update docstrings in `PUFFLEModel/puffle_model.py`
- [ ] **4.2.2** Review and update docstrings in `Regularization/disparity_loss.py`
- [ ] **4.2.3** Review and update docstrings in `Regularization/ErrorRateRegularizationLoss.py`
- [ ] **4.2.4** Review and update docstrings in `Utils/metric.py`
- [ ] **4.2.5** Remove any comments that are outdated or inconsistent with current code

### Task 4.3: Update README.md
**Goal**: README reflects current project state.

- [ ] **4.3.1** Replace all mentions of `Poetry` with `uv`
- [ ] **4.3.2** Update folder structure section to reflect current hierarchy (`src/puffle/`, not `DPL/`)
- [ ] **4.3.3** Rename all references from `DPL` to `PUFFLE`
- [ ] **4.3.4** Update code examples to use `PUFFLEModel` instead of old class names
- [ ] **4.3.5** Add section on running tests with `uv run pytest`
- [ ] **4.3.6** Add section on the synthetic example for quick testing

### Task 4.4: Standardize Naming (DPL → PUFFLE)
**Goal**: Consistent naming throughout codebase.

- [ ] **4.4.1** Search for remaining `DPL` references in code and comments
- [ ] **4.4.2** Replace `DPL` with `PUFFLE` or appropriate alternative
- [ ] **4.4.3** Verify no broken references after renaming

---

## Execution Order

1. **Phase 1** (Code Quality) - Must be done first to establish clean baseline
2. **Phase 2** (Test Infrastructure) - Fix tests before adding new ones
3. **Phase 3** (Synthetic Dataset) - Creates a lightweight validation mechanism
4. **Phase 4** (Documentation) - Can be parallelized with Phase 3

---

## Verification Checklist

After all tasks are complete, verify:
- [ ] `uv run ruff check ./src/puffle/` passes
- [ ] `uv run ruff format ./src/puffle/` makes no changes
- [ ] `uv run ty check ./src/puffle/` passes
- [ ] `uv run complexipy ./src/puffle/` shows acceptable complexity
- [ ] `uv run pytest ./src/puffle/tests/ -v` passes all tests
- [ ] Synthetic example runs successfully
- [ ] README accurately describes the project
