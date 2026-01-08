"""
Single property selection with FDR control

This module implements selection procedures for identifying test samples where a binary
property L==1 can be confidently claimed, controlling the False Discovery Rate (FDR).
"""

from typing import Optional

import numpy as np
import pandas as pd


def get_sel_single(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
    cal_eligs: Optional[np.ndarray] = None,
    test_eligs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Select test samples with L==1 among eligible samples with FDR control.

    Ranks test samples by score f(X) and selects those with high enough scores
    to guarantee FDR <= alpha.

    Parameters
    ----------
    cal_scores : np.ndarray
        Scores f(X_i) for calibration data (n,)
    cal_eligs : np.ndarray
        Eligibility E_i = E(X_i) for calibration data (n,)
    cal_labels : np.ndarray
        Labels L_i = L(X_i, Y_i) for calibration data (n,)
    test_scores : np.ndarray
        Scores f(X_{n+j}) for test data (m,)
    test_eligs : np.ndarray
        Eligibility E_{n+j} = E(X_{n+j}) for test data (m,)
    alpha : float
        FDR nominal level (e.g., 0.1 for 10% FDR)

    Returns
    -------
    sel_idx : np.ndarray
        Indices of selected test samples
    unsel_idx : np.ndarray
        Indices of unselected test samples
    hat_tau : float
        Selection threshold on scores

    Notes
    -----
    Guarantees that E[sum_{j in sel_idx} 1(L_{n+j}==0) / |sel_idx|] <= alpha
    """
    n_e = int(np.sum(cal_eligs))
    m_e = int(np.sum(test_eligs))

    # Combine calibration and test data for ranking
    df = pd.DataFrame({
        "mu": np.concatenate([cal_scores[cal_eligs == 1], test_scores[test_eligs == 1]]),
        "L": np.concatenate([cal_labels[cal_eligs == 1], np.zeros(m_e)]),
        "if_test": np.concatenate([np.zeros(n_e), np.ones(m_e)]),
        "id": range(n_e + m_e),
    }).sort_values(by="mu", ascending=False)

    # Compute FDR estimate for each threshold
    df["RR"] = (
        (1 + np.cumsum((df["L"] == 0) * (1 - df["if_test"])))
        / np.maximum(1, np.cumsum(df["if_test"]))
        * np.sum(df["if_test"])
        / (1 + np.sum(1 - df["if_test"]))
    )

    # Find largest threshold satisfying FDR control
    idx_sm = np.where(df["RR"] <= alpha)[0]

    if len(idx_sm) > 0:
        hat_tau = np.min(df["mu"].iloc[idx_sm])
        sel_idx = np.where((test_scores >= hat_tau) & (test_eligs == 1))[0]
    else:
        sel_idx = np.array([])
        hat_tau = 1.0

    unsel_idx = np.setdiff1d(np.arange(len(test_eligs)), sel_idx)
    return sel_idx, unsel_idx, hat_tau


def get_reference_sel_single(
    unsel_idx: np.ndarray,
    cal_conf_labels: np.ndarray,
    cal_conf_scores: np.ndarray,
    test_conf_scores: np.ndarray,
    test_imputed_conf_labels: np.ndarray,
    alpha: float,
    cal_eligs: Optional[np.ndarray] = None,
    test_eligs: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """
    Generate reference sets R(y) for unselected test instances.

    For each unselected test sample j and each potential label y, computes which
    calibration samples would be in the reference set R_j(y) for JOMI conformal prediction.

    Parameters
    ----------
    unsel_idx : np.ndarray
        Indices of unselected test samples
    cal_conf_labels : np.ndarray
        Labels L_i for calibration data (n,)
    cal_eligs : np.ndarray
        Eligibility E_i for calibration data (n,)
    cal_conf_scores : np.ndarray
        Scores f(X_i) for calibration data (n,)
    test_eligs : np.ndarray
        Eligibility E_{n+j} for test data (m,)
    test_conf_scores : np.ndarray
        Scores f(X_{n+j}) for test data (m,)
    test_imputed_conf_labels : np.ndarray
        Imputed labels L(X_{n+j}, y) for all test samples and labels (m, nclass)
    alpha : float
        FDR nominal level used in selection

    Returns
    -------
    ref_mats : list[np.ndarray]
        List of nclass binary matrices R(y), where ref_mats[y][j,i] indicates
        whether calibration sample i is in the reference set for test sample j
        assuming label y (m, n)

    Notes
    -----
    Reference sets are computed by checking whether calibration sample i would NOT
    be selected if we swapped it with test sample j assuming j has label y.
    """
    n = len(cal_conf_labels)
    m = len(test_conf_scores)
    n_e = int(np.sum(cal_eligs))
    m_e = int(np.sum(test_eligs))
    nclass = test_imputed_conf_labels.shape[1]

    # Initialize reference sets (all ones for selected, will modify for unselected)
    ref_mats = [np.ones((m, n)) for _ in range(nclass)]

    # If all samples are selected, return
    if len(unsel_idx) == m:
        return ref_mats

    # Set to zero for unselected samples (will compute properly below)
    for s in range(nclass):
        ref_mats[s][unsel_idx, :] = 0

    # Compute reference sets for each unselected sample
    for j in unsel_idx:
        if test_eligs[j] == 1:
            # Case: test sample j is eligible
            df = pd.DataFrame({
                "mu": np.concatenate([cal_conf_scores[cal_eligs == 1], test_conf_scores[test_eligs == 1]]),
                "L": np.concatenate([cal_conf_labels[cal_eligs == 1], np.zeros(m_e)]),
                "if_test": np.concatenate([np.zeros(n_e), np.ones(m_e)]),
                "id": range(n_e + m_e),
            }).sort_values(by="mu", ascending=False)

            # Compute FDR estimates for different swap scenarios
            df["R11"] = (
                (1 + np.cumsum((1 - df["if_test"]) * (df["L"] == 0)) + 1 * (df["mu"] <= test_conf_scores[j]))
                * (m_e - test_eligs[j] + 1)
                / (n_e + 1)
                / (np.cumsum(df["if_test"]) - 1 * (df["mu"] <= test_conf_scores[j]))
            )
            df["R10"] = (
                (1 + np.cumsum((1 - df["if_test"]) * (df["L"] == 0)))
                * (m_e - test_eligs[j] + 1)
                / (n_e + 1)
                / (np.cumsum(df["if_test"]) - 1 * (df["mu"] <= test_conf_scores[j]))
            )
            df["R01"] = (
                (np.cumsum((1 - df["if_test"]) * (df["L"] == 0)) + 1 * (df["mu"] <= test_conf_scores[j]))
                * (m_e - test_eligs[j] + 1)
                / (n_e + 1)
                / (np.cumsum(df["if_test"]) - 1 * (df["mu"] <= test_conf_scores[j]))
            )
            df["R00"] = (
                (np.cumsum((1 - df["if_test"]) * (df["L"] == 0)))
                * (m_e - test_eligs[j] + 1)
                / (n_e + 1)
                / (np.cumsum(df["if_test"]) - 1 * (df["mu"] <= test_conf_scores[j]))
            )

            # Find thresholds for each scenario
            idx_sm_00 = np.where(df["R00"] <= alpha)[0]
            idx_sm_01 = np.where(df["R01"] <= alpha)[0]
            idx_sm_10 = np.where(df["R10"] <= alpha)[0]
            idx_sm_11 = np.where(df["R11"] <= alpha)[0]

            tau_00 = np.min(df["mu"].iloc[idx_sm_00]) if len(idx_sm_00) > 0 else 1.0
            tau_01 = np.min(df["mu"].iloc[idx_sm_01]) if len(idx_sm_01) > 0 else 1.0
            tau_10 = np.min(df["mu"].iloc[idx_sm_10]) if len(idx_sm_10) > 0 else 1.0
            tau_11 = np.min(df["mu"].iloc[idx_sm_11]) if len(idx_sm_11) > 0 else 1.0

            # Construct reference sets based on whether calibration samples would be selected
            for y in range(nclass):
                ref_mats[y][j, cal_eligs == 0] = 1
                if test_imputed_conf_labels[j, y] == 1:
                    ref_mats[y][j, (cal_eligs == 1) & (cal_conf_labels == 1) & (cal_conf_scores < tau_10)] = 1
                    ref_mats[y][j, (cal_eligs == 1) & (cal_conf_labels == 0) & (cal_conf_scores < tau_00)] = 1
                if test_imputed_conf_labels[j, y] == 0:
                    ref_mats[y][j, (cal_eligs == 1) & (cal_conf_labels == 1) & (cal_conf_scores < tau_11)] = 1
                    ref_mats[y][j, (cal_eligs == 1) & (cal_conf_labels == 0) & (cal_conf_scores < tau_01)] = 1

        if test_eligs[j] == 0:
            # Case: test sample j is not eligible
            df = pd.DataFrame({
                "mu": np.concatenate([cal_conf_scores[cal_eligs == 1], test_conf_scores[test_eligs == 1]]),
                "L": np.concatenate([cal_conf_labels[cal_eligs == 1], np.zeros(m_e)]),
                "if_test": np.concatenate([np.zeros(n_e), np.ones(m_e)]),
                "id": range(n_e + m_e),
            }).sort_values(by="mu", ascending=False)

            df["R1"] = (
                (1 + np.cumsum((1 - df["if_test"]) * (df["L"] == 0)))
                * (m_e - test_eligs[j] + 1)
                / n_e
                / (1 + np.cumsum(df["if_test"]))
            )
            df["R0"] = (
                (np.cumsum((1 - df["if_test"]) * (df["L"] == 0)))
                * (m_e - test_eligs[j] + 1)
                / n_e
                / (1 + np.cumsum(df["if_test"]))
            )

            idx_sm_0 = np.where(df["R0"] <= alpha)[0]
            idx_sm_1 = np.where(df["R1"] <= alpha)[0]

            tau_0 = np.min(df["mu"].iloc[idx_sm_0]) if len(idx_sm_0) > 0 else 1.0
            tau_1 = np.min(df["mu"].iloc[idx_sm_1]) if len(idx_sm_1) > 0 else 1.0

            for y in range(nclass):
                ref_mats[y][j, cal_eligs == 0] = 1
                ref_mats[y][j, (cal_eligs == 1) & (cal_conf_labels == 1) & (cal_conf_scores < tau_1)] = 1
                ref_mats[y][j, (cal_eligs == 1) & (cal_conf_labels == 0) & (cal_conf_scores < tau_0)] = 1

    return ref_mats
