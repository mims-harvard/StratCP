"""
Conformal prediction methods for post-selection inference.

This module provides conformal prediction functions including nonconformity scores,
JOMI reference sets, and prediction set construction.
"""

from stratcp.conformal.core import conformal
from stratcp.conformal.scores import (
    compute_score_aps,
    compute_score_aps_consec,
    compute_score_raps,
    compute_score_raps_consec,
    compute_score_tps,
    compute_score_tps_consec,
)
from stratcp.conformal.utility import compute_score_utility, eval_similarity

__all__ = [
    # Core conformal prediction
    "conformal",
    # Score functions
    "compute_score_tps",
    "compute_score_aps",
    "compute_score_raps",
    # Consecutive score functions (for ordered labels)
    "compute_score_tps_consec",
    "compute_score_aps_consec",
    "compute_score_raps_consec",
    # Utility-aware score functions
    "compute_score_utility",
    "eval_similarity",
]
