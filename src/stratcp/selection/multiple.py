"""
Multiple property selection with FDR control

This module implements selection procedures for identifying test samples across
multiple selection rules (K rules), each controlling FDR independently.
"""

import numpy as np
import pandas as pd


def get_sel_multiple(
    cal_scores: np.ndarray,
    cal_eligs: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_eligs: np.ndarray,
    alpha: float,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Select test samples with L^k==1 for each of K selection rules with FDR control.

    For each rule k, ranks test samples by score f(X,k) and selects those with
    high enough scores to guarantee FDR <= alpha for that rule.

    Parameters
    ----------
    cal_scores : np.ndarray
        Scores f(X_i, k) for calibration data (n, K)
    cal_eligs : np.ndarray
        Eligibility E_i^k = E^k(X_i) for calibration data (n, K)
    cal_labels : np.ndarray
        Labels L_i^k = L^k(X_i, Y_i) for calibration data (n, K)
    test_scores : np.ndarray
        Scores f(X_{n+j}, k) for test data (m, K)
    test_eligs : np.ndarray
        Eligibility E_{n+j}^k = E^k(X_{n+j}) for test data (m, K)
    alpha : float
        FDR nominal level (e.g., 0.1 for 10% FDR)

    Returns
    -------
    all_sel : list[np.ndarray]
        List of K+1 arrays where all_sel[k] contains indices of test samples
        selected by rule k. The last element contains unselected indices.
    tau_list : list[float]
        Selection thresholds for each rule

    Notes
    -----
    Guarantees that E[sum_{j in all_sel[k]} 1(L_{n+j}^k==0) / |all_sel[k]|] <= alpha
    for each k independently.
    """
    nclass = cal_scores.shape[1]
    m = test_scores.shape[0]
    all_sel = []
    tau_list = []

    for k in range(nclass):
        n_e = int(np.sum(cal_eligs[:, k]))
        m_e = int(np.sum(test_eligs[:, k]))

        df = pd.DataFrame({
            "mu": np.concatenate([cal_scores[cal_eligs[:, k] == 1, k], test_scores[test_eligs[:, k] == 1, k]]),
            "L": np.concatenate([cal_labels[cal_eligs[:, k] == 1, k], np.zeros(m_e)]),
            "if_test": np.concatenate([np.zeros(n_e), np.ones(m_e)]),
            "id": range(n_e + m_e),
        }).sort_values(by="mu", ascending=False)

        df["RR"] = (
            (1 + np.cumsum((df["L"] == 0) * (1 - df["if_test"])))
            / np.maximum(1, np.cumsum(df["if_test"]))
            * np.sum(df["if_test"])
            / (1 + np.sum(1 - df["if_test"]))
        )

        idx_sm = np.where(df["RR"] <= alpha)[0]

        if len(idx_sm) > 0:
            hat_tau = np.min(df["mu"].iloc[idx_sm])
            sel_idx = np.where((test_scores[:, k] >= hat_tau) & (test_eligs[:, k] == 1))[0]
        else:
            sel_idx = np.array([])
            hat_tau = 1.0

        all_sel.append(sel_idx)
        tau_list.append(hat_tau)

    # Compute unselected indices (not selected by any rule)
    unique_elements = set()
    for array in all_sel:
        unique_elements.update(array)
    all_sel = all_sel + [np.array([x for x in range(m) if x not in unique_elements])]

    return all_sel, tau_list


def get_reference_sel_multiple(
    unsel_idx: np.ndarray,
    cal_labels: np.ndarray,
    cal_eligs: np.ndarray,
    cal_scores: np.ndarray,
    test_eligs: np.ndarray,
    test_scores: np.ndarray,
    test_imputed_labels: list[np.ndarray],
    alpha: float,
) -> list[np.ndarray]:
    """
    Generate reference sets R(y) for unselected test instances under multiple selection rules.

    For each unselected test sample j and each potential label y, computes which
    calibration samples would be in the reference set R_j(y) by checking if they
    would NOT be selected by ANY rule after swapping.

    Parameters
    ----------
    unsel_idx : np.ndarray
        Indices of unselected test samples
    cal_labels : np.ndarray
        Labels L_i^k for calibration data (n, K)
    cal_eligs : np.ndarray
        Eligibility E_i^k for calibration data (n, K)
    cal_scores : np.ndarray
        Scores f(X_i, k) for calibration data (n, K)
    test_eligs : np.ndarray
        Eligibility E_{n+j}^k for test data (m, K)
    test_scores : np.ndarray
        Scores f(X_{n+j}, k) for test data (m, K)
    test_imputed_labels : list[np.ndarray]
        List of K matrices where test_imputed_labels[k][j,y] = L^k(X_{n+j}, y)
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
    A calibration sample i is in R_j(y) if it would NOT be selected by any of the
    K rules after swapping with test sample j assuming j has label y.
    """
    n = cal_labels.shape[0]
    m = test_scores.shape[0]

    nclass = test_imputed_labels[0].shape[1]
    K = cal_scores.shape[1]

    # If all samples are selected, return all ones
    if len(unsel_idx) == m:
        ref_mats = [np.ones((m, n)) for _ in range(nclass)]
        return ref_mats

    # Initialize reference sets
    ref_mats = [np.ones((m, n)) for _ in range(nclass)]

    for s in range(nclass):
        ref_mats[s][unsel_idx, :] = 0

    # Compute reference sets for each unselected sample
    for j in unsel_idx:
        # Track which calibration samples are NOT selected after swap for each label
        if_not_sel_after_swap = [np.zeros((n, K)) for _ in range(nclass)]

        for k in range(K):
            n_e = int(np.sum(cal_eligs[:, k]))
            m_e = int(np.sum(test_eligs[:, k]))

            if test_eligs[j, k] == 1:
                # Case: test sample j is eligible for rule k
                df = pd.DataFrame({
                    "mu": np.concatenate([cal_scores[cal_eligs[:, k] == 1, k], test_scores[test_eligs[:, k] == 1, k]]),
                    "L": np.concatenate([cal_labels[cal_eligs[:, k] == 1, k], np.zeros(m_e)]),
                    "if_test": np.concatenate([np.zeros(n_e), np.ones(m_e)]),
                    "id": range(n_e + m_e),
                }).sort_values(by="mu", ascending=False)

                df["R11"] = (
                    (1 + np.cumsum((1 - df["if_test"]) * (df["L"] == 0)) + 1 * (df["mu"] <= test_scores[j, k]))
                    * (m_e - test_eligs[j, k] + 1)
                    / (n_e + 1)
                    / np.maximum(1, (np.cumsum(df["if_test"]) - 1 * (df["mu"] <= test_scores[j, k])))
                )
                df["R10"] = (
                    (1 + np.cumsum((1 - df["if_test"]) * (df["L"] == 0)))
                    * (m_e - test_eligs[j, k] + 1)
                    / (n_e + 1)
                    / np.maximum(1, (np.cumsum(df["if_test"]) - 1 * (df["mu"] <= test_scores[j, k])))
                )
                df["R01"] = (
                    (np.cumsum((1 - df["if_test"]) * (df["L"] == 0)) + 1 * (df["mu"] <= test_scores[j, k]))
                    * (m_e - test_eligs[j, k] + 1)
                    / (n_e + 1)
                    / np.maximum(1, (np.cumsum(df["if_test"]) - 1 * (df["mu"] <= test_scores[j, k])))
                )
                df["R00"] = (
                    (np.cumsum((1 - df["if_test"]) * (df["L"] == 0)))
                    * (m_e - test_eligs[j, k] + 1)
                    / (n_e + 1)
                    / np.maximum(1, (np.cumsum(df["if_test"]) - 1 * (df["mu"] <= test_scores[j, k])))
                )

                idx_sm_00 = np.where(df["R00"] <= alpha)[0]
                idx_sm_01 = np.where(df["R01"] <= alpha)[0]
                idx_sm_10 = np.where(df["R10"] <= alpha)[0]
                idx_sm_11 = np.where(df["R11"] <= alpha)[0]

                tau_00 = np.min(df["mu"].iloc[idx_sm_00]) if len(idx_sm_00) > 0 else 1.0
                tau_01 = np.min(df["mu"].iloc[idx_sm_01]) if len(idx_sm_01) > 0 else 1.0
                tau_10 = np.min(df["mu"].iloc[idx_sm_10]) if len(idx_sm_10) > 0 else 1.0
                tau_11 = np.min(df["mu"].iloc[idx_sm_11]) if len(idx_sm_11) > 0 else 1.0

                for y in range(nclass):
                    if_not_sel_after_swap[y][cal_eligs[:, k] == 0, k] = 1
                    if test_imputed_labels[k][j, y] == 1:
                        if_not_sel_after_swap[y][
                            (cal_eligs[:, k] == 1) & (cal_labels[:, k] == 1) & (cal_scores[:, k] < tau_10), k
                        ] = 1
                        if_not_sel_after_swap[y][
                            (cal_eligs[:, k] == 1) & (cal_labels[:, k] == 0) & (cal_scores[:, k] < tau_00), k
                        ] = 1
                    if test_imputed_labels[k][j, y] == 0:
                        if_not_sel_after_swap[y][
                            (cal_eligs[:, k] == 1) & (cal_labels[:, k] == 1) & (cal_scores[:, k] < tau_11), k
                        ] = 1
                        if_not_sel_after_swap[y][
                            (cal_eligs[:, k] == 1) & (cal_labels[:, k] == 0) & (cal_scores[:, k] < tau_01), k
                        ] = 1

            if test_eligs[j, k] == 0:
                # Case: test sample j is not eligible for rule k
                df = pd.DataFrame({
                    "mu": np.concatenate([cal_scores[cal_eligs[:, k] == 1, k], test_scores[test_eligs[:, k] == 1, k]]),
                    "L": np.concatenate([cal_labels[cal_eligs[:, k] == 1, k], np.zeros(m_e)]),
                    "if_test": np.concatenate([np.zeros(n_e), np.ones(m_e)]),
                    "id": range(n_e + m_e),
                }).sort_values(by="mu", ascending=False)

                df["R1"] = (
                    (1 + np.cumsum((1 - df["if_test"]) * (df["L"] == 0)))
                    * (m_e - test_eligs[j, k] + 1)
                    / n_e
                    / (1 + np.cumsum(df["if_test"]))
                )
                df["R0"] = (
                    (np.cumsum((1 - df["if_test"]) * (df["L"] == 0)))
                    * (m_e - test_eligs[j, k] + 1)
                    / n_e
                    / (1 + np.cumsum(df["if_test"]))
                )

                idx_sm_0 = np.where(df["R0"] <= alpha)[0]
                idx_sm_1 = np.where(df["R1"] <= alpha)[0]

                tau_0 = np.min(df["mu"].iloc[idx_sm_0]) if len(idx_sm_0) > 0 else 1.0
                tau_1 = np.min(df["mu"].iloc[idx_sm_1]) if len(idx_sm_1) > 0 else 1.0

                for y in range(nclass):
                    if_not_sel_after_swap[y][cal_eligs[:, k] == 0, k] = 1
                    if_not_sel_after_swap[y][
                        (cal_eligs[:, k] == 1) & (cal_labels[:, k] == 1) & (cal_scores[:, k] < tau_1), k
                    ] = 1
                    if_not_sel_after_swap[y][
                        (cal_eligs[:, k] == 1) & (cal_labels[:, k] == 0) & (cal_scores[:, k] < tau_0), k
                    ] = 1

        # Calibration sample is in reference set if NOT selected by any rule
        for y in range(nclass):
            ref_mats[y][j, :] = 1 * (np.sum(if_not_sel_after_swap[y], axis=1) == K)

    return ref_mats
