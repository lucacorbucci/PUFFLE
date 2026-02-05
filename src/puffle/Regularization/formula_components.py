"""
Data structures for fairness formula components.

This module defines the FormulaComponents dataclass, which holds confusion matrix
statistics for both privileged and unprivileged groups in fairness computations.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FormulaComponents:
    """
    Components computed for error rate parity formula.

    Contains confusion matrix components for privileged and unprivileged groups,
    plus analysis metadata.
    """

    # Unprivileged group confusion matrix (differentiable scores)
    fp_unprivileged: float
    tp_unprivileged: float
    tn_unprivileged: float
    fn_unprivileged: float

    # Privileged group confusion matrix (differentiable scores)
    fp_privileged: float
    tp_privileged: float
    tn_privileged: float
    fn_privileged: float

    # Argmax-based counts (for hard predictions)
    fp_unprivileged_argmax: float
    tp_unprivileged_argmax: float
    tn_unprivileged_argmax: float
    fn_unprivileged_argmax: float
    fp_privileged_argmax: float
    tp_privileged_argmax: float
    tn_privileged_argmax: float
    fn_privileged_argmax: float

    # Analysis dictionary for debugging/logging
    analysis_dict: dict = field(default_factory=dict)
