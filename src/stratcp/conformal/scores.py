"""
Nonconformity score functions for conformal prediction.

This module provides various score functions (TPS, APS, RAPS) for computing
nonconformity scores from predicted class probabilities.
"""

import numpy as np


def compute_score_tps(
    cal_smx: np.ndarray,
    test_smx: np.ndarray,
    cal_labels: np.ndarray,
    nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Threshold-based Prediction Set (TPS) scores.

    TPS uses simple threshold-based scores: 1 - p(y).

    Parameters
    ----------
    cal_smx : np.ndarray
        Predicted probabilities for calibration data (n, nclass)
    test_smx : np.ndarray
        Predicted probabilities for test data (m, nclass)
    cal_labels : np.ndarray
        True labels for calibration data (n,)
    test_labels : np.ndarray
        True labels for test data (m,)
    nonempty : bool, default=True
        If True, ensure top predicted class has score 0

    Returns
    -------
    cal_scores : np.ndarray
        Nonconformity scores for calibration data (n,)
    val_all_scores : np.ndarray
        Nonconformity scores for all test labels (m, nclass)
    """
    n = len(cal_labels)
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]

    val_all_scores = 1 - test_smx

    if nonempty:
        cal_scores = cal_scores * (cal_labels != cal_smx.argmax(axis=1))
        test_max_id = np.argmax(test_smx, axis=1)
        val_all_scores[np.arange(test_smx.shape[0]), test_max_id] = 0

    return cal_scores, val_all_scores


def compute_score_aps(
    cal_smx: np.ndarray,
    test_smx: np.ndarray,
    cal_labels: np.ndarray,
    nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Adaptive Prediction Set (APS) scores.

    APS uses cumulative probability up to and including the true class.

    Parameters
    ----------
    cal_smx : np.ndarray
        Predicted probabilities for calibration data (n, nclass)
    test_smx : np.ndarray
        Predicted probabilities for test data (m, nclass)
    cal_labels : np.ndarray
        True labels for calibration data (n,)
    test_labels : np.ndarray
        True labels for test data (m,)
    nonempty : bool, default=True
        If True, ensure top predicted class has score 0

    Returns
    -------
    cal_scores : np.ndarray
        Nonconformity scores for calibration data (n,)
    val_all_scores : np.ndarray
        Nonconformity scores for all test labels (m, nclass)
    """
    n = len(cal_labels)
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[range(n), cal_labels]

    test_max_id = np.argmax(test_smx, axis=1)
    val_pi = test_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(test_smx, val_pi, axis=1).cumsum(axis=1)
    val_all_scores = np.take_along_axis(val_srt, val_pi.argsort(axis=1), axis=1)

    if nonempty:
        cal_scores = cal_scores * (cal_labels != cal_smx.argmax(axis=1))
        val_all_scores[np.arange(test_smx.shape[0]), test_max_id] = 0

    return cal_scores, val_all_scores


def compute_score_raps(
    cal_smx: np.ndarray,
    test_smx: np.ndarray,
    cal_labels: np.ndarray,
    lam_reg: float = 0.01,
    nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Regularized Adaptive Prediction Set (RAPS) scores.

    RAPS adds regularization to APS to encourage smaller prediction sets.

    Parameters
    ----------
    cal_smx : np.ndarray
        Predicted probabilities for calibration data (n, nclass)
    test_smx : np.ndarray
        Predicted probabilities for test data (m, nclass)
    cal_labels : np.ndarray
        True labels for calibration data (n,)
    test_labels : np.ndarray
        True labels for test data (m,)
    lam_reg : float, default=0.01
        Regularization parameter
    nonempty : bool, default=True
        If True, ensure top predicted class has score 0

    Returns
    -------
    cal_scores : np.ndarray
        Nonconformity scores for calibration data (n,)
    val_all_scores : np.ndarray
        Nonconformity scores for all test labels (m, nclass)
    """
    n = len(cal_labels)

    k_reg = min(5, cal_smx.shape[1])
    reg_vec = np.array(k_reg * [0] + (cal_smx.shape[1] - k_reg) * [lam_reg])[None, :]

    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:, None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L]

    n_val = test_smx.shape[0]
    val_pi = test_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(test_smx, val_pi, axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    val_all_scores = np.take_along_axis(val_srt_reg_cumsum, val_pi.argsort(axis=1), axis=1)

    if nonempty:
        cal_scores = cal_scores * (cal_labels != cal_smx.argmax(axis=1))
        test_max_id = np.argmax(test_smx, axis=1)
        val_all_scores[np.arange(test_smx.shape[0]), test_max_id] = 0

    return cal_scores, val_all_scores


def get_consec_ordering(smx: np.ndarray) -> list[int]:
    """
    Get consecutive ordering of classes for ordinal labels.

    Starting from the highest probability class, alternately adds adjacent
    classes to create a consecutive ordering.

    Parameters
    ----------
    smx : np.ndarray
        Class probabilities (nclass,)

    Returns
    -------
    ordering : list[int]
        Consecutive ordering of class indices
    """
    nclass = len(smx)
    ordering = [int(np.argmax(smx))]
    left = int(np.argmax(smx))
    right = int(np.argmax(smx))

    for _ in range(nclass - 1):
        # Consecutively add left-1 or right+1
        if left == 0:
            ordering.append(right + 1)
            right += 1
        elif right == (nclass - 1):
            ordering.append(left - 1)
            left -= 1
        elif smx[left - 1] <= smx[right + 1]:
            ordering.append(right + 1)
            right += 1
        elif smx[left - 1] > smx[right + 1]:
            ordering.append(left - 1)
            left -= 1

    return ordering


def compute_score_tps_consec(
    cal_smx: np.ndarray,
    test_smx: np.ndarray,
    cal_labels: np.ndarray,
    nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute TPS scores ensuring consecutive prediction sets for ordinal labels.

    Parameters
    ----------
    cal_smx : np.ndarray
        Predicted probabilities for calibration data (n, nclass)
    test_smx : np.ndarray
        Predicted probabilities for test data (m, nclass)
    cal_labels : np.ndarray
        True labels for calibration data (n,)
    test_labels : np.ndarray
        True labels for test data (m,)
    nonempty : bool, default=True
        If True, ensure top predicted class has score 0

    Returns
    -------
    cal_scores : np.ndarray
        Nonconformity scores for calibration data (n,)
    val_all_scores : np.ndarray
        Nonconformity scores for all test labels (m, nclass)
    """
    n = len(cal_labels)
    m = test_smx.shape[0]
    nclass = cal_smx.shape[1]
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]

    for i in range(n):
        # Determine ordering in a consecutive fashion
        i_order = get_consec_ordering(cal_smx[i, :])
        true_idx = i_order.index(cal_labels[i])
        cal_scores[i] = 1 - np.min(cal_smx[i, np.array(i_order)[range(true_idx + 1)]])

    val_all_scores = 1 - test_smx
    test_max_id = np.argmax(test_smx, axis=1)

    for i in range(m):
        # Determine ordering in a consecutive fashion
        i_order = get_consec_ordering(test_smx[i, :])
        for j in range(nclass):
            val_all_scores[i, i_order[j]] = 1 - np.min(test_smx[i, np.array(i_order)[range(j + 1)]])

    if nonempty:
        cal_scores = cal_scores * (cal_labels != cal_smx.argmax(axis=1))
        test_max_id = np.argmax(test_smx, axis=1)
        val_all_scores[np.arange(test_smx.shape[0]), test_max_id] = 0

    return cal_scores, val_all_scores


def compute_score_aps_consec(
    cal_smx: np.ndarray,
    test_smx: np.ndarray,
    cal_labels: np.ndarray,
    nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute APS scores ensuring consecutive prediction sets for ordinal labels.

    Parameters
    ----------
    cal_smx : np.ndarray
        Predicted probabilities for calibration data (n, nclass)
    test_smx : np.ndarray
        Predicted probabilities for test data (m, nclass)
    cal_labels : np.ndarray
        True labels for calibration data (n,)
    test_labels : np.ndarray
        True labels for test data (m,)
    nonempty : bool, default=True
        If True, ensure top predicted class has score 0

    Returns
    -------
    cal_scores : np.ndarray
        Nonconformity scores for calibration data (n,)
    val_all_scores : np.ndarray
        Nonconformity scores for all test labels (m, nclass)
    """
    n = len(cal_labels)
    m = test_smx.shape[0]
    nclass = cal_smx.shape[1]
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[range(n), cal_labels]

    for i in range(n):
        # Determine ordering in a consecutive fashion
        i_order = get_consec_ordering(cal_smx[i, :])
        true_idx = i_order.index(cal_labels[i])
        cal_scores[i] = np.sum(cal_smx[i, np.array(i_order)[range(true_idx + 1)]])

    test_max_id = np.argmax(test_smx, axis=1)
    val_all_scores = np.ones((m, nclass))

    for i in range(m):
        # Determine ordering in a consecutive fashion
        i_order = get_consec_ordering(test_smx[i, :])
        for j in range(nclass):
            val_all_scores[i, i_order[j]] = np.sum(test_smx[i, np.array(i_order)[range(j + 1)]])

    if nonempty:
        cal_scores = cal_scores * (cal_labels != cal_smx.argmax(axis=1))
        val_all_scores[np.arange(test_smx.shape[0]), test_max_id] = 0

    return cal_scores, val_all_scores


def compute_score_raps_consec(
    cal_smx: np.ndarray,
    test_smx: np.ndarray,
    cal_labels: np.ndarray,
    lam_reg: float = 0.05,
    nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RAPS scores ensuring consecutive prediction sets for ordinal labels.

    Parameters
    ----------
    cal_smx : np.ndarray
        Predicted probabilities for calibration data (n, nclass)
    test_smx : np.ndarray
        Predicted probabilities for test data (m, nclass)
    cal_labels : np.ndarray
        True labels for calibration data (n,)
    test_labels : np.ndarray
        True labels for test data (m,)
    lam_reg : float, default=0.05
        Regularization parameter
    nonempty : bool, default=True
        If True, ensure top predicted class has score 0

    Returns
    -------
    cal_scores : np.ndarray
        Nonconformity scores for calibration data (n,)
    val_all_scores : np.ndarray
        Nonconformity scores for all test labels (m, nclass)
    """
    n = len(cal_labels)
    m = test_smx.shape[0]
    nclass = cal_smx.shape[1]

    k_reg = min(2, cal_smx.shape[1])
    reg_vec = np.array(k_reg * [0] + (cal_smx.shape[1] - k_reg) * [lam_reg])[None, :]

    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:, None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L]

    for i in range(n):
        # Determine ordering in a consecutive fashion
        i_order = get_consec_ordering(cal_smx[i, :])
        true_idx = i_order.index(cal_labels[i])
        cal_scores[i] = np.sum(cal_smx[i, np.array(i_order)[range(true_idx + 1)]]) + np.sum(
            reg_vec[0, range(1 + true_idx)]
        )

    test_max_id = np.argmax(test_smx, axis=1)
    val_all_scores = np.ones((m, nclass))

    for i in range(m):
        # Determine ordering in a consecutive fashion
        i_order = get_consec_ordering(test_smx[i, :])
        for j in range(nclass):
            val_all_scores[i, i_order[j]] = np.sum(test_smx[i, np.array(i_order)[range(j + 1)]]) + np.sum(
                reg_vec[0, range(1 + j)]
            )

    if nonempty:
        cal_scores = cal_scores * (cal_labels != cal_smx.argmax(axis=1))
        test_max_id = np.argmax(test_smx, axis=1)
        val_all_scores[np.arange(test_smx.shape[0]), test_max_id] = 0

    return cal_scores, val_all_scores
