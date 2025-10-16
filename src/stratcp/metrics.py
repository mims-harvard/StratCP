"""
Evaluation metrics for conformal prediction.

This module provides functions for evaluating coverage, prediction set sizes,
and decision-specific metrics for medical diagnosis applications.
"""

import numpy as np
import pandas as pd


def size_cond_cov(pred_set: np.ndarray, test_labels: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """
    Compute conditional coverage by prediction set size.

    Parameters
    ----------
    pred_set : np.ndarray
        Prediction set (m, nclass)

    y : np.ndarray
        True labels (m,)  

    Returns
    -------
    cond_cov : pd.Series
        Conditional coverage for each set size
    cond_freq : pd.Series
        Frequency of each set size
    """

    cov = np.array([pred_set[i, test_labels[i]] for i in range(pred_set.shape[0])])
    size = np.sum(pred_set, axis=1)
    num_class = pred_set.shape[1]
    cond_cov = (
        pd.Series(cov * 1).groupby(pd.Series(size)).mean().reindex(range(0, num_class + 1), fill_value=np.nan)
    )
    cond_freq = pd.Series(cov * 1).groupby(pd.Series(size)).count().reindex(range(0, num_class + 1), fill_value=0)
    return cond_cov, cond_freq


def label_cond_cov(pred_set: np.ndarray, test_labels: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """
    Compute conditional coverage by true label.

    Parameters
    ----------
    pred_set : np.ndarray
        Binary prediction set matrix (m, nclass)
    test_labels : np.ndarray
        True labels (m,)

    Returns
    -------
    cond_cov : pd.Series
        Conditional coverage for each label
    cond_freq : pd.Series
        Frequency of each label
    """
    cov = np.array([pred_set[i, test_labels[i]] for i in range(pred_set.shape[0])])
    num_class = pred_set.shape[1]

    cond_cov = pd.Series(cov * 1).groupby(pd.Series(test_labels)).mean().reindex(range(0, num_class), fill_value=np.nan)
    cond_freq = pd.Series(cov * 1).groupby(pd.Series(test_labels)).count().reindex(range(0, num_class), fill_value=np.nan)
    return cond_cov, cond_freq


def eval_singleton_cover(pred_set: np.ndarray, test_labels: np.ndarray) -> tuple[float, float]:
    """
    Evaluate coverage separately for singleton and non-singleton prediction sets.

    Parameters
    ----------
    pred_set : np.ndarray
        Binary prediction set matrix (m, nclass)
    test_labels : np.ndarray
        True labels (m,)

    Returns
    -------
    singleton_cover : float
        Coverage for singleton prediction sets
    non_singleton_cover : float
        Coverage for non-singleton prediction sets
    """
    set_size = np.sum(pred_set, axis=1)
    singleton_cover = np.mean(
        [pred_set[i, test_labels[i]] for i in range(pred_set.shape[0]) if set_size[i] == 1]
    )
    non_singleton_cover = np.mean(
        [pred_set[i, test_labels[i]] for i in range(pred_set.shape[0]) if set_size[i] != 1]
    )
    return singleton_cover, non_singleton_cover


def eval_consec(pred_set: np.ndarray) -> np.ndarray:
    """
    Check if prediction sets are consecutive (for ordinal labels).

    Parameters
    ----------
    pred_set : np.ndarray
        Binary prediction set matrix (m, nclass)

    Returns
    -------
    if_consec : np.ndarray
        Binary indicators for consecutiveness (m,)
    """
    if_consec = np.zeros((pred_set.shape[0]))
    for i in range(pred_set.shape[0]):
        true_indices = np.flatnonzero(pred_set[i, :])

        if true_indices.size <= 1:
            if_consec[i] = True
        else:
            if_consec[i] = np.all(pred_set[i, true_indices[0] : true_indices[-1] + 1])

    return if_consec
