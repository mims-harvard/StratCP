"""
Survival analysis selection with FDR control (CASE C).

This module implements selection procedures for survival models, identifying
test samples with survival time above threshold with FDR control.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def get_sel_survival(
    cal_labels: np.ndarray,
    cal_pred: np.ndarray,
    cal_threshold: np.ndarray,
    hat_sigma: float,
    test_pred: np.ndarray,
    test_threshold: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Select test samples with survival time T > c with FDR control.

    Uses a survival model F(x,y) = Phi([log(y) - mu(x)] / sigma) to select
    samples predicted to survive beyond threshold c with FDR control.

    Parameters
    ----------
    cal_labels : np.ndarray
        Survival times T_i for calibration data (n,)
    cal_pred : np.ndarray
        Predicted values mu(X_i) for calibration data (n,)
    cal_threshold : np.ndarray
        Survival thresholds c_i for calibration data (n,)
    hat_sigma : float
        Estimated scale parameter sigma
    test_pred : np.ndarray
        Predicted values mu(X_{n+j}) for test data (m,)
    test_threshold : np.ndarray
        Survival thresholds c_{n+j} for test data (m,)
    alpha : float
        FDR nominal level (e.g., 0.1 for 10% FDR)

    Returns
    -------
    sel_idx : np.ndarray
        Indices of selected test samples (predicted long survivors)
    unsel_idx : np.ndarray
        Indices of unselected test samples
    hat_tau : float
        Selection threshold on scores

    Notes
    -----
    Guarantees that E[sum_{j in sel_idx} 1(T_{n+j} <= c_{n+j}) / |sel_idx|] <= alpha
    """
    n = len(cal_pred)
    m = len(test_pred)

    # Transform to scores (lower is better for long survival)
    cal_scores = np.log(cal_threshold) - cal_pred
    test_scores = np.log(test_threshold) - test_pred

    df = pd.DataFrame(
        {
            "mu": np.concatenate([cal_scores, test_scores]),
            "L": np.concatenate([1 * (cal_labels > cal_threshold), np.zeros(m)]),
            "if_test": np.concatenate([np.zeros(n), np.ones(m)]),
            "id": range(n + m),
        }
    ).sort_values(by="mu", ascending=True)

    df["RR"] = (
        (1 + np.cumsum((df["L"] == 0) * (1 - df["if_test"]))) / np.maximum(1, np.cumsum(df["if_test"])) * m / (1 + n)
    )

    idx_sm = np.where(df["RR"] <= alpha)[0]

    if len(idx_sm) > 0:
        hat_tau = np.max(df["mu"].iloc[idx_sm])
        sel_idx = np.where(test_scores <= hat_tau)[0]
    else:
        sel_idx = np.array([])
        hat_tau = 1.0

    unsel_idx = np.setdiff1d(np.arange(m), sel_idx)
    return sel_idx, unsel_idx, hat_tau


def get_jomi_survival_lcb(
    unsel_idx: np.ndarray,
    cal_labels: np.ndarray,
    cal_pred: np.ndarray,
    cal_threshold: np.ndarray,
    hat_sigma: float,
    test_pred: np.ndarray,
    test_threshold: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Generate JOMI lower confidence bounds for unselected survival samples.

    For each unselected test sample, computes a lower confidence bound [eta_j, +inf)
    for the survival time using JOMI post-selection inference.

    Parameters
    ----------
    unsel_idx : np.ndarray
        Indices of unselected test samples
    cal_labels : np.ndarray
        Survival times T_i for calibration data (n,)
    cal_pred : np.ndarray
        Predicted values mu(X_i) for calibration data (n,)
    cal_threshold : np.ndarray
        Survival thresholds c_i for calibration data (n,)
    hat_sigma : float
        Estimated scale parameter sigma
    test_pred : np.ndarray
        Predicted values mu(X_{n+j}) for test data (m,)
    test_threshold : np.ndarray
        Survival thresholds c_{n+j} for test data (m,)
    alpha : float
        FDR nominal level used in selection

    Returns
    -------
    hat_LCB : np.ndarray
        Lower confidence bounds for unselected samples (len(unsel_idx),)

    Notes
    -----
    The LCB is computed by finding reference sets for each unselected sample and
    using them to construct valid post-selection conformal intervals.
    """
    n = len(cal_pred)
    m = len(test_pred)
    cal_scores = np.log(cal_threshold) - cal_pred
    test_scores = np.log(test_threshold) - test_pred

    df = pd.DataFrame(
        {
            "mu": np.concatenate([cal_scores, test_scores]),
            "L": np.concatenate([1 * (cal_labels > cal_threshold), np.zeros(m)]),
            "if_test": np.concatenate([np.zeros(n), np.ones(m)]),
            "id": range(n + m),
        }
    ).sort_values(by="mu", ascending=True)

    hat_LCB = np.zeros((len(unsel_idx)))

    for jj in range(len(unsel_idx)):
        j = unsel_idx[jj]

        # Compute FDR estimates for swap scenarios
        df["R00"] = (
            (np.cumsum((df["L"] == 0) * (1 - df["if_test"])))
            / np.maximum(1, 1 + np.cumsum(df["if_test"]) - 1 * (test_scores[j] <= df["mu"]))
            * m
            / (1 + n)
        )
        df["R01"] = (
            (np.cumsum((df["L"] == 0) * (1 - df["if_test"])) + 1 * (test_scores[j] <= df["mu"]))
            / np.maximum(1, 1 + np.cumsum(df["if_test"]) - 1 * (test_scores[j] <= df["mu"]))
            * m
            / (1 + n)
        )
        df["R11"] = (
            (1 + np.cumsum((df["L"] == 0) * (1 - df["if_test"])) + 1 * (test_scores[j] <= df["mu"]))
            / np.maximum(1, 1 + np.cumsum(df["if_test"]) - 1 * (test_scores[j] <= df["mu"]))
            * m
            / (1 + n)
        )
        df["R10"] = (
            (1 + np.cumsum((df["L"] == 0) * (1 - df["if_test"])))
            / np.maximum(1, 1 + np.cumsum(df["if_test"]) - 1 * (test_scores[j] <= df["mu"]))
            * m
            / (1 + n)
        )

        idx_sm_00 = np.where(df["R00"] <= alpha)[0]
        idx_sm_01 = np.where(df["R01"] <= alpha)[0]
        idx_sm_10 = np.where(df["R10"] <= alpha)[0]
        idx_sm_11 = np.where(df["R11"] <= alpha)[0]

        tau_00 = np.max(df["mu"].iloc[idx_sm_00]) if len(idx_sm_00) > 0 else -np.inf
        tau_01 = np.max(df["mu"].iloc[idx_sm_01]) if len(idx_sm_01) > 0 else -np.inf
        tau_10 = np.max(df["mu"].iloc[idx_sm_10]) if len(idx_sm_10) > 0 else -np.inf
        tau_11 = np.max(df["mu"].iloc[idx_sm_11]) if len(idx_sm_11) > 0 else -np.inf

        # Construct reference sets
        Rj_1 = [i for i in range(n) if cal_labels[i] <= cal_threshold[i] and cal_scores[i] > tau_01] + [
            i for i in range(n) if cal_labels[i] > cal_threshold[i] and cal_scores[i] > tau_11
        ]
        Rj_0 = [i for i in range(n) if cal_labels[i] <= cal_threshold[i] and cal_scores[i] > tau_00] + [
            i for i in range(n) if cal_labels[i] > cal_threshold[i] and cal_scores[i] > tau_10
        ]

        # Compute lower confidence bounds using reference sets
        eta_1 = np.quantile(
            np.concatenate(
                [-norm.cdf(np.log(cal_labels[Rj_1]) - cal_pred[Rj_1], loc=0, scale=hat_sigma), np.array([np.inf])]
            ),
            q=1 - alpha,
            method="higher",
        )
        eta_0 = np.quantile(
            np.concatenate(
                [-norm.cdf(np.log(cal_labels[Rj_0]) - cal_pred[Rj_0], loc=0, scale=hat_sigma), np.array([np.inf])]
            ),
            q=1 - alpha,
            method="higher",
        )

        eta_1 = np.exp(test_pred[j] + hat_sigma * norm.ppf(-eta_1))
        eta_0 = np.exp(test_pred[j] + hat_sigma * norm.ppf(-eta_0))

        # Select appropriate bound based on threshold
        if eta_1 > test_threshold[j]:
            eta = np.max((test_threshold[j], eta_0))
        else:
            eta = eta_1

        hat_LCB[jj] = eta

    return hat_LCB
