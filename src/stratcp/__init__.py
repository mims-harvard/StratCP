"""
StratCP: Stratified Conformal Prediction

A Python package for post-selection conformal inference with FDR-controlled stratification.

This package provides tools for making confident predictions with statistical guarantees
by stratifying data based on prediction confidence and applying different conformal
prediction procedures to each stratum.

Key Features
------------
- FDR-controlled selection procedures (single, multiple, survival)
- Post-selection conformal inference (JOMI)
- Multiple nonconformity scores (TPS, APS, RAPS)
- Support for ordinal labels with consecutive prediction sets
- Evaluation metrics for coverage and efficiency

Basic Usage
-----------
>>> from stratcp import StratifiedCP
>>> from stratcp.conformal import compute_score_raps
>>> from stratcp.selection import get_sel_single
>>>
>>> # Make FDR-controlled selection
>>> sel_idx, unsel_idx, tau = get_sel_single(
...     cal_scores, cal_eligs, cal_labels,
...     test_scores, test_eligs, alpha=0.1
... )
>>>
>>> # Compute nonconformity scores
>>> cal_scores, test_scores = compute_score_raps(
...     cal_probs, test_probs, cal_labels, test_labels
... )

Modules
-------
selection
    FDR-controlled selection methods (single, multiple, survival)
conformal
    Conformal prediction methods (core, scores)
metrics
    Evaluation metrics for coverage and efficiency
utils
    Utility functions for data manipulation
"""

# Version info
__version__ = "0.1.0"
__author__ = "Zitnik Lab"
__email__ = "marinka@hms.harvard.edu"

# Import main modules for convenient access
from stratcp import conformal, metrics, selection, utils

# Import high-level API
from stratcp.stratified import StratifiedCP

# Import key functions for direct access
from stratcp.conformal import (
    compute_score_aps,
    compute_score_raps,
    compute_score_tps,
    compute_score_utility,
    conformal,
    eval_similarity,
)
from stratcp.metrics import eval_consec, label_cond_cov, size_cond_cov
from stratcp.selection import (
    get_jomi_survival_lcb,
    get_reference_sel_multiple,
    get_reference_sel_single,
    get_sel_multiple,
    get_sel_single,
    get_sel_survival,
)
from stratcp.utils import combine_data

__all__ = [
    # High-level API
    "StratifiedCP",
    # Modules
    "selection",
    "conformal",
    "metrics",
    "utils",
    # Key selection functions
    "get_sel_single",
    "get_sel_multiple",
    "get_sel_survival",
    "get_reference_sel_single",
    "get_reference_sel_multiple",
    "get_jomi_survival_lcb",
    # Key conformal functions
    "conformal",
    "compute_score_tps",
    "compute_score_aps",
    "compute_score_raps",
    "compute_score_utility",
    "eval_similarity",
    # Utility functions
    "combine_data",
    "size_cond_cov",
    "label_cond_cov",
    "eval_consec",
]
