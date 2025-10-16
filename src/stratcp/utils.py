"""
Utility functions for stratified conformal prediction.

This module provides helper functions for data manipulation and common operations.
"""

import numpy as np


def combine_data(
    sel_idx: np.ndarray, unsel_idx: np.ndarray, sel_res: np.ndarray, unsel_res: np.ndarray
) -> np.ndarray:
    """
    Combine results from selected and unselected samples.

    Parameters
    ----------
    sel_idx : np.ndarray
        Indices of selected samples
    unsel_idx : np.ndarray
        Indices of unselected samples
    sel_res : np.ndarray
        Results for selected samples
    unsel_res : np.ndarray
        Results for unselected samples

    Returns
    -------
    com_res : np.ndarray
        Combined results in original order
    """
    com_res = np.zeros(len(sel_idx) + len(unsel_idx))
    com_res[sel_idx] = sel_res
    com_res[unsel_idx] = unsel_res
    return com_res
