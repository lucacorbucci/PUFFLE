# PUFFLE Cleanup Changelog

This file documents the changes made during the code quality improvement phase.

## 2026-01-26: Phase 1 - Ruff Error Fixes

### Deleted Dead Code Files
- `src/puffle/Utils/tabular_datasets_utils.py` - Entirely commented-out code (610 lines), never imported
- `src/puffle/Utils/utils.py` - Entirely commented-out code, never imported  
- `src/puffle/Utils/regularization_config.py` - Entirely commented-out code, never imported

### Added Missing `__init__.py` Files
Created package markers for proper Python imports:
- `src/puffle/tests/__init__.py`
- `src/puffle/PUFFLEModel/__init__.py`
- `src/puffle/Regularization/__init__.py`
- `src/puffle/Utils/__init__.py`
- `src/puffle/examples/__init__.py`
- `src/puffle/examples/utils/__init__.py`
- `src/puffle/examples/models/__init__.py`
- `src/puffle/examples/data_preparation/__init__.py`
- `src/puffle/examples/centralised/__init__.py`
- `src/puffle/examples/centralised/dutch/__init__.py`
- `src/puffle/examples/centralised/celeba/__init__.py`

### Removed Commented-Out Code (ERA001)
Cleaned up scattered commented code from:
- `src/puffle/Regularization/ErrorRateRegularizationLoss.py` - Removed old debug prints, unused variable assignments
- `src/puffle/Regularization/disparity_loss.py` - Removed unused variable declarations
- `src/puffle/Utils/metric.py` - Removed commented validation code
- `src/puffle/examples/data_preparation/dataset_preparation.py` - Removed debug prints
- `src/puffle/examples/models/models.py` - Removed commented line
- `src/puffle/tests/test_regularization_loss.py` - Removed commented assertion

### Progress
- Started with 991 ruff errors
- Reduced to 212 errors after ERA001 fixes
- Remaining categories: S101 (assert in tests), FBT (boolean args), N806 (variable naming), etc.
