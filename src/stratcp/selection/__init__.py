"""
FDR-controlled selection of directly actionable predictions for stratified conformal prediction.

This module provides functions for making FDR-controlled selections based on predictions,
supporting single property selection, multiple property selection, and survival analysis.
"""

from stratcp.selection.multiple import get_reference_sel_multiple, get_sel_multiple
from stratcp.selection.single import get_reference_sel_single, get_sel_single
from stratcp.selection.survival import get_jomi_survival_lcb, get_sel_survival

__all__ = [
    # Single selection (CASE A)
    "get_sel_single",
    "get_reference_sel_single",
    # Multiple selection (CASE B)
    "get_sel_multiple",
    "get_reference_sel_multiple",
    # Survival selection (CASE C)
    "get_sel_survival",
    "get_jomi_survival_lcb",
]
