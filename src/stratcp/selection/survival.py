"""
Survival analysis selection with FDR control (CASE C).

This module implements selection procedures for survival models, identifying
test samples with survival time above threshold with FDR control.
"""

# from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# def get_sel_survival(
#     cal_labels: np.ndarray,
#     cal_pred: np.ndarray,
#     cal_threshold: np.ndarray,
#     hat_sigma: float,
#     test_pred: np.ndarray,
#     test_threshold: np.ndarray,
#     alpha: float,
# ) -> tuple[np.ndarray, np.ndarray, float]:
#     """
#     Select test samples with survival time T > c with FDR control.

#     Uses a survival model F(x,y) = Phi([log(y) - mu(x)] / sigma) to select
#     samples predicted to survive beyond threshold c with FDR control.

#     Parameters
#     ----------
#     cal_labels : np.ndarray
#         Survival times T_i for calibration data (n,)
#     cal_pred : np.ndarray
#         Predicted values mu(X_i) for calibration data (n,)
#     cal_threshold : np.ndarray
#         Survival thresholds c_i for calibration data (n,)
#     hat_sigma : float
#         Estimated scale parameter sigma
#     test_pred : np.ndarray
#         Predicted values mu(X_{n+j}) for test data (m,)
#     test_threshold : np.ndarray
#         Survival thresholds c_{n+j} for test data (m,)
#     alpha : float
#         FDR nominal level (e.g., 0.1 for 10% FDR)

#     Returns
#     -------
#     sel_idx : np.ndarray
#         Indices of selected test samples (predicted long survivors)
#     unsel_idx : np.ndarray
#         Indices of unselected test samples
#     hat_tau : float
#         Selection threshold on scores

#     Notes
#     -----
#     Guarantees that E[sum_{j in sel_idx} 1(T_{n+j} <= c_{n+j}) / |sel_idx|] <= alpha
#     """
#     n = len(cal_pred)
#     m = len(test_pred)

#     # Transform to scores (lower is better for long survival)
#     cal_scores = np.log(cal_threshold) - cal_pred
#     test_scores = np.log(test_threshold) - test_pred

#     df = pd.DataFrame(
#         {
#             "mu": np.concatenate([cal_scores, test_scores]),
#             "L": np.concatenate([1 * (cal_labels > cal_threshold), np.zeros(m)]),
#             "if_test": np.concatenate([np.zeros(n), np.ones(m)]),
#             "id": range(n + m),
#         }
#     ).sort_values(by="mu", ascending=True)

#     df["RR"] = (
#         (1 + np.cumsum((df["L"] == 0) * (1 - df["if_test"]))) / np.maximum(1, np.cumsum(df["if_test"])) * m / (1 + n)
#     )

#     idx_sm = np.where(df["RR"] <= alpha)[0]

#     if len(idx_sm) > 0:
#         hat_tau = np.max(df["mu"].iloc[idx_sm])
#         sel_idx = np.where(test_scores <= hat_tau)[0]
#     else:
#         sel_idx = np.array([])
#         hat_tau = 1.0

#     unsel_idx = np.setdiff1d(np.arange(m), sel_idx)
#     return sel_idx, unsel_idx, hat_tau


def get_sel_survival(
    cal_labels: np.ndarray,  # calibration survival times T_i (n,)
    cal_scores: np.ndarray,  # calibration scores s_i (n,) — lower = more confident long survival
    cal_threshold: np.ndarray,  # calibration thresholds c_i (n,)
    test_scores: np.ndarray,  # test scores s_j (m,) — lower = more confident long survival
    test_threshold: np.ndarray,  # test thresholds c_j (m,)
    alpha: float,  # target FDR level in (0, 1)
    w_ipcw: np.ndarray | None = None,  # optional IPCW weights for calibration (n,)
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Select test samples predicted to survive beyond their thresholds with FDR control.

    Assumes **lower scores** mean higher confidence of long survival. We compute a
    data-driven cut `tau_hat` and select all test points with `score <= tau_hat`,
    controlling the expected false discovery proportion (FDP) at level `alpha`.

    Parameters
    ----------
    cal_labels : np.ndarray
        Calibration survival times T_i, shape (n,)
    cal_scores : np.ndarray
        Calibration scores s_i, shape (n,)
    cal_threshold : np.ndarray
        Calibration thresholds c_i, shape (n,)
    test_scores : np.ndarray
        Test scores s_j, shape (m,)
    test_threshold : np.ndarray
        Test thresholds c_j, shape (m,)
    alpha : float
        Nominal FDR level in (0, 1)
    w_ipcw : np.ndarray | None, optional
        IPCW weights for calibration, shape (n,). If provided, we estimate FDP
        using the weighted calibration error count.

    Returns
    -------
    sel_idx : np.ndarray
        Indices (0..m-1) of selected test samples (predicted long survivors).
    unsel_idx : np.ndarray
        Complement of `sel_idx` (unselected test samples).
    tau_hat : float
        Data-driven threshold applied to scores.

    Notes
    -----
    Under standard assumptions, this ensures E[FDP(sel_idx)] <= alpha, where FDP
    counts events {T_j <= c_j} among the selected test units.
    """
    # -------------------- Basic validation & shapes --------------------
    cal_labels = np.asarray(cal_labels)
    cal_scores = np.asarray(cal_scores)
    cal_threshold = np.asarray(cal_threshold)
    test_scores = np.asarray(test_scores)
    test_threshold = np.asarray(test_threshold)

    n = cal_scores.shape[0]
    m = test_scores.shape[0]

    if not (cal_labels.shape == cal_scores.shape == cal_threshold.shape == (n,)):
        raise ValueError("bad calibration shapes")
    if not (test_scores.shape == test_threshold.shape == (m,)):
        raise ValueError("bad test shapes")
    if np.any(cal_threshold <= 0) or np.any(test_threshold <= 0):
        raise ValueError("thresholds must be > 0")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    if w_ipcw is not None:
        w_ipcw = np.asarray(w_ipcw)
        if w_ipcw.shape != (n,):
            raise ValueError("w_ipcw must be (n,)")
        if np.any(w_ipcw < 0):
            raise ValueError("w_ipcw must be nonnegative")

    # Rank calibration + test together
    # We sort all (n + m) observations by ascending score. For any cut tau,
    # we consider selecting all test points with score <= tau. The only unknowns
    # are the test labels (we don't know yet if T_j > c_j), but calibration
    # labels are known and provide an empirical estimate of FDP at each tau.

    if w_ipcw is not None:
        # Weighted path: Z_i = w_i * 1{T_i <= c_i} for calibration rows (test rows get 0).
        df = pd.DataFrame({
            "mu": np.concatenate([cal_scores, test_scores]),
            "Z": np.concatenate([w_ipcw * (cal_labels <= cal_threshold), np.zeros(m)]),
            "if_test": np.concatenate([np.zeros(n, dtype=int), np.ones(m, dtype=int)]),
        }).sort_values("mu", ascending=True)
    else:
        # Unweighted path: L_i = 1{T_i > c_i} for calibration (success = long survival).
        df = pd.DataFrame({
            "mu": np.concatenate([cal_scores, test_scores]),
            "L": np.concatenate([(cal_labels > cal_threshold).astype(int), np.zeros(m, dtype=int)]),
            "if_test": np.concatenate([np.zeros(n, dtype=int), np.ones(m, dtype=int)]),
        }).sort_values("mu", ascending=True)

    ## Running FDP estimate at each potential tau
    # For the first k rows in the sorted list (i.e., all with score <= tau_k):
    #   - Denominator: # of test points among those k (how many we'd select)
    #   - Numerator:   Estimated # of false discoveries among those k
    #                  * Unweighted: 1 + (# calibration errors where T_i <= c_i)
    #                  * Weighted:   1 + (sum of Z_i = w_i * 1{T_i <= c_i})
    #   - The factor m / (n + 1) makes this estimate (approximately) unbiased.
    # We then pick the LARGEST tau whose estimated FDP <= alpha (most permissive cut).

    cum_test = np.cumsum(df["if_test"])

    if w_ipcw is not None:
        # Weighted FDP: (1 + sum Z) / max(1, #test) * (m / (n + 1))
        rr = (1.0 + np.cumsum(df["Z"])) / np.maximum(1, cum_test) * (m / (n + 1.0))
    else:
        # Unweighted FDP: numerator counts calibration errors (L == 0) up to tau
        cal_err = (df.get("L", 0) == 0) & (df["if_test"] == 0)
        rr = (1.0 + np.cumsum(cal_err)) / np.maximum(1, cum_test) * (m / (n + 1.0))

    ## Cut selection and index extraction
    ok = rr <= alpha
    if np.any(ok):
        # Most permissive threshold that still meets the FDR budget.
        tau_hat = float(df.loc[ok, "mu"].max())

        # Select every test index with score <= tau_hat.
        # Note: test_scores are the last m entries in the concatenation,
        # but we simply re-check against tau_hat in the separate test array.
        sel_idx = np.where(test_scores <= tau_hat)[0].astype(int)
    else:
        # No feasible τ: select none.
        tau_hat = float("-inf")
        sel_idx = np.array([], dtype=int)

    # Complement set for downstream procedures (e.g., LCBs).
    unsel_idx = np.setdiff1d(np.arange(m, dtype=int), sel_idx)

    return sel_idx, unsel_idx, tau_hat


# def get_jomi_survival_lcb(
#     unsel_idx: np.ndarray,
#     cal_labels: np.ndarray,
#     cal_pred: np.ndarray,
#     cal_threshold: np.ndarray,
#     hat_sigma: float,
#     test_pred: np.ndarray,
#     test_threshold: np.ndarray,
#     alpha: float,
# ) -> np.ndarray:
#     """
#     Generate JOMI lower confidence bounds for unselected survival samples.

#     For each unselected test sample, computes a lower confidence bound [eta_j, +inf)
#     for the survival time using JOMI post-selection inference.

#     Parameters
#     ----------
#     unsel_idx : np.ndarray
#         Indices of unselected test samples
#     cal_labels : np.ndarray
#         Survival times T_i for calibration data (n,)
#     cal_pred : np.ndarray
#         Predicted values mu(X_i) for calibration data (n,)
#     cal_threshold : np.ndarray
#         Survival thresholds c_i for calibration data (n,)
#     hat_sigma : float
#         Estimated scale parameter sigma
#     test_pred : np.ndarray
#         Predicted values mu(X_{n+j}) for test data (m,)
#     test_threshold : np.ndarray
#         Survival thresholds c_{n+j} for test data (m,)
#     alpha : float
#         FDR nominal level used in selection

#     Returns
#     -------
#     hat_LCB : np.ndarray
#         Lower confidence bounds for unselected samples (len(unsel_idx),)

#     Notes
#     -----
#     The LCB is computed by finding reference sets for each unselected sample and
#     using them to construct valid post-selection conformal intervals.
#     """
#     n = len(cal_pred)
#     m = len(test_pred)
#     cal_scores = np.log(cal_threshold) - cal_pred
#     test_scores = np.log(test_threshold) - test_pred

#     df = pd.DataFrame({
#         "mu": np.concatenate([cal_scores, test_scores]),
#         "L": np.concatenate([1 * (cal_labels > cal_threshold), np.zeros(m)]),
#         "if_test": np.concatenate([np.zeros(n), np.ones(m)]),
#         "id": range(n + m),
#     }).sort_values(by="mu", ascending=True)

#     hat_LCB = np.zeros(len(unsel_idx))

#     for jj in range(len(unsel_idx)):
#         j = unsel_idx[jj]

#         # Compute FDR estimates for swap scenarios
#         df["R00"] = (
#             (np.cumsum((df["L"] == 0) * (1 - df["if_test"])))
#             / np.maximum(1, 1 + np.cumsum(df["if_test"]) - 1 * (test_scores[j] <= df["mu"]))
#             * m
#             / (1 + n)
#         )
#         df["R01"] = (
#             (np.cumsum((df["L"] == 0) * (1 - df["if_test"])) + 1 * (test_scores[j] <= df["mu"]))
#             / np.maximum(1, 1 + np.cumsum(df["if_test"]) - 1 * (test_scores[j] <= df["mu"]))
#             * m
#             / (1 + n)
#         )
#         df["R11"] = (
#             (1 + np.cumsum((df["L"] == 0) * (1 - df["if_test"])) + 1 * (test_scores[j] <= df["mu"]))
#             / np.maximum(1, 1 + np.cumsum(df["if_test"]) - 1 * (test_scores[j] <= df["mu"]))
#             * m
#             / (1 + n)
#         )
#         df["R10"] = (
#             (1 + np.cumsum((df["L"] == 0) * (1 - df["if_test"])))
#             / np.maximum(1, 1 + np.cumsum(df["if_test"]) - 1 * (test_scores[j] <= df["mu"]))
#             * m
#             / (1 + n)
#         )

#         idx_sm_00 = np.where(df["R00"] <= alpha)[0]
#         idx_sm_01 = np.where(df["R01"] <= alpha)[0]
#         idx_sm_10 = np.where(df["R10"] <= alpha)[0]
#         idx_sm_11 = np.where(df["R11"] <= alpha)[0]

#         tau_00 = np.max(df["mu"].iloc[idx_sm_00]) if len(idx_sm_00) > 0 else -np.inf
#         tau_01 = np.max(df["mu"].iloc[idx_sm_01]) if len(idx_sm_01) > 0 else -np.inf
#         tau_10 = np.max(df["mu"].iloc[idx_sm_10]) if len(idx_sm_10) > 0 else -np.inf
#         tau_11 = np.max(df["mu"].iloc[idx_sm_11]) if len(idx_sm_11) > 0 else -np.inf

#         # Construct reference sets
#         Rj_1 = [i for i in range(n) if cal_labels[i] <= cal_threshold[i] and cal_scores[i] > tau_01] + [
#             i for i in range(n) if cal_labels[i] > cal_threshold[i] and cal_scores[i] > tau_11
#         ]
#         Rj_0 = [i for i in range(n) if cal_labels[i] <= cal_threshold[i] and cal_scores[i] > tau_00] + [
#             i for i in range(n) if cal_labels[i] > cal_threshold[i] and cal_scores[i] > tau_10
#         ]

#         # Compute lower confidence bounds using reference sets
#         eta_1 = np.quantile(
#             np.concatenate([
#                 -norm.cdf(np.log(cal_labels[Rj_1]) - cal_pred[Rj_1], loc=0, scale=hat_sigma),
#                 np.array([np.inf]),
#             ]),
#             q=1 - alpha,
#             method="higher",
#         )
#         eta_0 = np.quantile(
#             np.concatenate([
#                 -norm.cdf(np.log(cal_labels[Rj_0]) - cal_pred[Rj_0], loc=0, scale=hat_sigma),
#                 np.array([np.inf]),
#             ]),
#             q=1 - alpha,
#             method="higher",
#         )

#         eta_1 = np.exp(test_pred[j] + hat_sigma * norm.ppf(-eta_1))
#         eta_0 = np.exp(test_pred[j] + hat_sigma * norm.ppf(-eta_0))

#         # Select appropriate bound based on threshold
#         if eta_1 > test_threshold[j]:
#             eta = np.max((test_threshold[j], eta_0))
#         else:
#             eta = eta_1

#         hat_LCB[jj] = eta

#     return hat_LCB

# def get_jomi_survival_lcb(
#     unsel_idx:      np.ndarray,   # test indices to conformalize (subset of {0,…,m‑1})
#     cal_labels:     np.ndarray,   # ̃T_i (observed follow-up) on calibration set        (shape n)
#     cal_pred:       np.ndarray,   # μ̂(X_i) on calibration                               (shape n)
#     cal_threshold:  np.ndarray,   # horizon c_i on calibration                           (shape n)
#     cal_hat_sigma:  float,        # σ̂ on calibration
#     val_pred:       np.ndarray,   # μ̂(X_j) on test                                       (shape m)
#     val_threshold:  np.ndarray,   # horizon c_j on test                                   (shape m)
#     val_hat_sigma:  np.ndarray,   # σ̂ on test                                           (shape m)
#     alpha:          float,        # nominal level for one‑sided conformal LCB
#     w_ipcw:         np.ndarray | None = None,  # IPCW weights on calibration (Δ/Ĝ_c(Y|X)); None ⇒ unweighted
#     clip_ppf:       float = 1e-12  # numerical guard for Φ^{-1} inputs
# ):
#     """
#     Compute JOMI-style lower confidence bounds [LCB, +∞) for the *unselected* test units,
#     *consistent with the selection rule you used* (weighted or unweighted).

#     Returns
#     -------
#     hat_LCB : np.ndarray, shape (len(unsel_idx),)
#         Lower bounds aligned with index order in `unsel_idx`.

#     Selection‑conditional guarantee (FCR):
#         E[ mean( 1{ T_j < LCB_j } ) | j ∈ unsel_idx ] ≤ α
#     """

#     # -----------------------------------------------------------------
#     # 0) Basic sizes
#     # -----------------------------------------------------------------
#     n = len(cal_pred)     # calibration size
#     m = len(val_pred)     # test size

#     # -----------------------------------------------------------------
#     # 1) Reconstruct the *same* score used in selection:
#     #    s = (log c - μ̂)/σ̂   (lower ⇒ stronger evidence of long survival)
#     # -----------------------------------------------------------------
#     cal_scores = (np.log(cal_threshold) - cal_pred) / cal_hat_sigma    # shape (n,)
#     val_scores = (np.log(val_threshold) - val_pred) / val_hat_sigma    # shape (m,)

#     # -----------------------------------------------------------------
#     # 2) Build a combined table (calibration + test), sorted by score.
#     #    We'll reuse this to trace the four RR curves R00, R01, R10, R11.
#     #
#     #    Columns:
#     #      mu       : the score s
#     #      L        : (only if unweighted) indicator of "long" on calibration: 1{T̃_i > c_i}
#     #      Z        : (only if weighted)    IPCW short indicator: w_i * 1{T̃_i ≤ c_i}
#     #      if_test  : 1 for test rows, 0 for calibration rows
#     # -----------------------------------------------------------------
#     if w_ipcw is not None:
#         # Weighted variant: provide a running *weighted* count of shorts via Z.
#         df = pd.DataFrame({
#             "mu":      np.concatenate([cal_scores,              val_scores]),
#             "Z":       np.concatenate([w_ipcw * (cal_labels <= cal_threshold),   # weighted "short" indicator
#                                        np.zeros(m, dtype=float)]),               # test rows have no Z
#             "if_test": np.concatenate([np.zeros(n, dtype=int),
#                                        np.ones(m,  dtype=int)]),
#         }).sort_values(by="mu", ascending=True).reset_index(drop=True)
#     else:
#         # Unweighted variant: keep the binary label L=1{T̃>c} on calibration rows only.
#         df = pd.DataFrame({
#             "mu":      np.concatenate([cal_scores,              val_scores]),
#             "L":       np.concatenate([(cal_labels > cal_threshold).astype(int),
#                                        np.zeros(m, dtype=int)]),
#             "if_test": np.concatenate([np.zeros(n, dtype=int),
#                                        np.ones(m,  dtype=int)]),
#         }).sort_values(by="mu", ascending=True).reset_index(drop=True)

#     # Allocate output array; will be filled in the same order as `unsel_idx`.
#     hat_LCB = np.zeros(len(unsel_idx), dtype=float)

#     # -----------------------------------------------------------------
#     # 3) Precompute calibration residuals used in split‑conformal quantiles.
#     #    z_cal = (log T̃ - μ̂)/σ̂   →  Φ(z_cal) = F̂(T̃ | X)
#     #    We'll need -Φ(z_cal) values for quantiles later.
#     # -----------------------------------------------------------------
#     z_cal   = (np.log(cal_labels) - cal_pred) / cal_hat_sigma
#     F_cal   = norm.cdf(z_cal)              # in (0,1)
#     neg_cdf = -F_cal                       # in (-1, 0); convenient for "higher" quantiles below

#     # -----------------------------------------------------------------
#     # 4) Loop over each unselected test unit j.  JOMI operates per‑unit.
#     # -----------------------------------------------------------------
#     for out_pos, j in enumerate(unsel_idx):

#         # -------------------------------------------------------------
#         # 4-A) For unit j, we need the four *hypothetical* RR curves,
#         #      mirroring the selection rule and excluding j itself.
#         #
#         #      Notation: R_ab
#         #        a ∈ {0,1}  says whether the global "+1" regularizer is included (as in selection)
#         #        b ∈ {0,1}  assumes whether j *contributes* an additional "short" when it would be selected
#         #                   via the indicator 1{ val_scores[j] ≤ μ }, exactly as in your original code.
#         #
#         #      Denominator: same as selection, but exclude j from the test running count.
#         # -------------------------------------------------------------
#         ind_test_leq_mu = (val_scores[j] <= df["mu"].values)  # boolean vector along sorted μ

#         # Running denominator = max(1, (#test up to μ) - 1{j would be counted here}) × m/(n+1)
#         cum_test_all = np.cumsum(df["if_test"].values)                     # all test rows up to μ
#         denom = np.maximum(1,
#                            1 + cum_test_all - ind_test_leq_mu.astype(int)) * m / (n + 1)

#         # -------------------- Numerators (four worlds) ----------------
#         if w_ipcw is not None:
#             # Weighted calibration "shorts": cumulative sum of Z over calibration rows
#             cum_Z = np.cumsum(df["Z"].values)     # running weighted errors from calibration
#             # R00: base (no +1 regularizer, no j's own short counted)
#             R00 =           (        cum_Z                               ) / denom
#             # R01: base + j as potential short when selected (add 1 when μ crosses j)
#             R01 =           (        cum_Z + ind_test_leq_mu.astype(int) ) / denom
#             # R10: add the global "+1" as in selection RR
#             R10 =           ( 1    + cum_Z                               ) / denom
#             # R11: global "+1" + j as short
#             R11 =           ( 1    + cum_Z + ind_test_leq_mu.astype(int) ) / denom
#         else:
#             # Unweighted: use cumulative count of calibration *errors* (L==0) only on calibration rows
#             cal_error_mask = ((df["L"].values == 0) & (df["if_test"].values == 0))
#             cum_err = np.cumsum(cal_error_mask.astype(int))
#             # Same four worlds with integer counts
#             R00 =           (        cum_err                               ) / denom
#             R01 =           (        cum_err + ind_test_leq_mu.astype(int) ) / denom
#             R10 =           ( 1    + cum_err                               ) / denom
#             R11 =           ( 1    + cum_err + ind_test_leq_mu.astype(int) ) / denom

#         # -------------------------------------------------------------
#         # 4-B) For each curve, extract the *largest* μ threshold τ_ab
#         #      such that RR ≤ α (mirrors the selection step).
#         # -------------------------------------------------------------
#         # Use -inf as sentinel when constraint is never met (keeps "worse_than_τ" false).
#         tau_00 = df.loc[R00 <= alpha, "mu"].max() if np.any(R00 <= alpha) else -np.inf
#         tau_01 = df.loc[R01 <= alpha, "mu"].max() if np.any(R01 <= alpha) else -np.inf
#         tau_10 = df.loc[R10 <= alpha, "mu"].max() if np.any(R10 <= alpha) else -np.inf
#         tau_11 = df.loc[R11 <= alpha, "mu"].max() if np.any(R11 <= alpha) else -np.inf

#         # -------------------------------------------------------------
#         # 4-C) Build calibration index sets Rj_0 and Rj_1 used in the
#         #      split‑conformal quantiles (same construction as your code).
#         #
#         #      in_cal_short = 1{T̃_i ≤ c_i}  (i.e., L_i = 0)
#         #      in_cal_long  = 1{T̃_i  > c_i} (i.e., L_i = 1)
#         #
#         #      "worse_than_τab" means the calibration score is > τ_ab,
#         #      i.e., those calibration points lie outside the selected region
#         #      at that world (they form the residual set we take quantiles over).
#         # -------------------------------------------------------------
#         in_cal_short = (cal_labels <= cal_threshold)           # L_i = 0
#         in_cal_long  = ~in_cal_short                           # L_i = 1

#         worse_00 = (cal_scores > tau_00)
#         worse_01 = (cal_scores > tau_01)
#         worse_10 = (cal_scores > tau_10)
#         worse_11 = (cal_scores > tau_11)

#         # Rj_1: hypothetical label "long" for j  → use thresholds (01,11)
#         Rj_1 = np.where( (in_cal_short & worse_01) | (in_cal_long & worse_11) )[0]

#         # Rj_0: hypothetical label "short" for j → use thresholds (00,10)
#         Rj_0 = np.where( (in_cal_short & worse_00) | (in_cal_long & worse_10) )[0]

#         # -------------------------------------------------------------
#         # 4-D) Split‑conformal quantiles on the transformed residuals:
#         #      we use the vector  -Φ(z_cal)  (values in (-1,0)).
#         #
#         #      Add +∞ to keep the quantile defined even if Rj_b is empty.
#         #      (Then -η_b = -∞, and after clipping ppf argument we get L_b ≈ 0.)
#         # -------------------------------------------------------------
#         eta_1 = np.quantile(np.concatenate([neg_cdf[Rj_1], [np.inf]]),
#                             q=1 - alpha, method="higher")
#         eta_0 = np.quantile(np.concatenate([neg_cdf[Rj_0], [np.inf]]),
#                             q=1 - alpha, method="higher")

#         # Convert η_b → probability argument for Φ^{-1}:
#         #   p_b = -η_b  (since η_b ∈ (-1,0] ∪ {+∞})
#         #   guard to (clip_ppf, 1-clip_ppf)
#         p_1 = float(np.clip(-eta_1, clip_ppf, 1 - clip_ppf))
#         p_0 = float(np.clip(-eta_0, clip_ppf, 1 - clip_ppf))

#         # -------------------------------------------------------------
#         # 4-E) Map back to time via the log‑normal inverse CDF at test j:
#         #      L_b = exp( μ̂_j + σ̂_j * Φ^{-1}(p_b) )
#         # -------------------------------------------------------------
#         zq_1 = norm.ppf(p_1)
#         zq_0 = norm.ppf(p_0)

#         L_1 = np.exp(val_pred[j] + val_hat_sigma[j] * zq_1)
#         L_0 = np.exp(val_pred[j] + val_hat_sigma[j] * zq_0)

#         # -------------------------------------------------------------
#         # 4-F) Final LCB for j:
#         #      If under label=1 (long) the bound already exceeds c_j,
#         #      take the worst case between c_j and L_0 (label=0 world);
#         #      otherwise use L_1.
#         # -------------------------------------------------------------
#         if L_1 > val_threshold[j]:
#             eta_final = max(val_threshold[j], L_0)
#         else:
#             eta_final = L_1

#         hat_LCB[out_pos] = float(eta_final)

#     # -----------------------------------------------------------------
#     # 5) Return vector of LCBs in the same order as `unsel_idx`
#     # -----------------------------------------------------------------
#     return hat_LCB


def _score_from_threshold(
    model_family: str,
    threshold: np.ndarray,
    loc: np.ndarray | float,
    scale: np.ndarray | float,
) -> np.ndarray:
    """
    Convert a (threshold, params) tuple into the selection score used by the
    survival selector. Lower scores = stronger evidence of long survival.

    Currently implemented:
        - log_normal: s = (log(c) - loc) / scale
    """
    if model_family == "log_normal":
        return (np.log(threshold) - loc) / scale
    raise NotImplementedError(f"model_family={model_family!r} not implemented")


def _z_from_time(
    model_family: str,
    time: np.ndarray,
    loc: np.ndarray | float,
    scale: np.ndarray | float,
) -> np.ndarray:
    """
    Transform observed time into a standard score z whose CDF is Phi(z)
    under the model.
    """
    if model_family == "log_normal":
        return (np.log(time) - loc) / scale
    raise NotImplementedError(f"model_family={model_family!r} not implemented")


def _inv_time_from_ppf(
    model_family: str,
    p: float,
    loc: float,
    scale: float,
) -> float:
    """
    Inverse-CDF map back to time at probability p for a single unit.
    """
    if model_family == "log_normal":
        return float(np.exp(loc + scale * norm.ppf(p)))
    raise NotImplementedError(f"model_family={model_family!r} not implemented")


def get_jomi_survival_lcb(
    unsel_idx: np.ndarray,  # test indices to conformalize (subset of 0..m-1)
    # calibration split (size n)
    cal_labels: np.ndarray,  # observed follow-up times T_i (must be > 0)
    cal_threshold: np.ndarray,  # horizons c_i (> 0)
    cal_loc: np.ndarray,  # model location parameter (e.g., log-mean) for calibration
    cal_scale: np.ndarray | float,  # model scale parameter for calibration (scalar or (n,))
    # test split (size m)
    test_threshold: np.ndarray,  # horizons c_j (> 0)
    test_loc: np.ndarray,  # model location parameter for test
    test_scale: np.ndarray | float,  # model scale parameter for test (scalar or (m,))
    # selector settings
    alpha: float,  # nominal level for one-sided LCBs
    w_ipcw: np.ndarray | None,  # IPCW weights for calibration; None ⇒ unweighted
    clip_ppf: float = 1e-12,  # numerical guard for Phi^{-1} inputs
    # model family
    model_family: str = "log_normal",
) -> np.ndarray:
    """
    JOMI-style lower confidence bounds for *unselected* test units, aligned with the
    same selection rule (weighted or unweighted) used by the selector.

    Parameters
    ----------
    unsel_idx : np.ndarray
        Indices (subset of 0..m-1) of test units to conformalize.
    cal_labels, cal_threshold : np.ndarray
        Observed calibration times T_i (>0) and horizons c_i (>0), shape (n,).
    cal_loc, cal_scale : array-like
        Model parameters for calibration; for `log_normal`, loc=log-mean, scale=log-std.
        `cal_scale` can be scalar or shape (n,).
    test_threshold : np.ndarray
        Horizons c_j (>0) on the test split, shape (m,).
    test_loc, test_scale : array-like
        Model parameters for test; `test_scale` can be scalar or shape (m,).
    alpha : float
        Nominal miscoverage level in (0,1) for the one-sided LCBs.
    w_ipcw : np.ndarray | None, optional
        IPCW weights on calibration (shape (n,)); if provided, we use the weighted RR.
    clip_ppf : float, optional
        Lower/upper clipping for inverse-CDF arguments (stability).
    model_family : {"log_normal"}, optional
        Parametric family for the time-to-event model. Currently only "log_normal" is implemented.

    Returns
    -------
    hat_LCB : np.ndarray, shape (len(unsel_idx),)
        Lower confidence bounds in the same order as `unsel_idx`.

    Guarantee (selection-conditional; FCR-style)
    -------------------------------------------
        E[ mean( 1{ T_j < LCB_j } ) | j ∈ unsel_idx ] ≤ alpha
    """
    # Basic validation
    unsel_idx = np.asarray(unsel_idx, dtype=int)
    cal_labels = np.asarray(cal_labels)
    cal_threshold = np.asarray(cal_threshold)
    cal_loc = np.asarray(cal_loc)
    test_threshold = np.asarray(test_threshold)
    test_loc = np.asarray(test_loc)

    n = cal_labels.shape[0]
    if not (cal_threshold.shape == cal_loc.shape == (n,)):
        raise ValueError("bad calibration shapes")

    m = test_threshold.shape[0]
    if not (test_loc.shape == (m,)):
        raise ValueError("bad test shapes")

    if np.any(cal_labels <= 0) or np.any(cal_threshold <= 0) or np.any(test_threshold <= 0):
        raise ValueError("times/thresholds must be > 0")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if np.any((unsel_idx < 0) | (unsel_idx >= m)):
        raise ValueError("unsel_idx out of range")

    # allow scalar or per-sample scale
    cal_scale = np.asarray(cal_scale) if np.ndim(cal_scale) else float(cal_scale)
    test_scale = np.asarray(test_scale) if np.ndim(test_scale) else float(test_scale)
    if np.ndim(cal_scale) and cal_scale.shape != (n,):
        raise ValueError("cal_scale must be scalar or (n,)")
    if np.ndim(test_scale) and test_scale.shape != (m,):
        raise ValueError("test_scale must be scalar or (m,)")

    if w_ipcw is not None:
        w_ipcw = np.asarray(w_ipcw)
        if w_ipcw.shape != (n,):
            raise ValueError("w_ipcw must be (n,)")
        if np.any(w_ipcw < 0):
            raise ValueError("w_ipcw must be nonnegative")

    # Selection scores (same as the selector)
    cal_scores = _score_from_threshold(model_family, cal_threshold, cal_loc, cal_scale)
    test_scores = _score_from_threshold(model_family, test_threshold, test_loc, test_scale)

    # Combined ranking over calibration + test to build the RR curves
    if w_ipcw is not None:
        df = (
            pd.DataFrame({
                "mu": np.concatenate([cal_scores, test_scores]),
                "Z": np.concatenate([w_ipcw * (cal_labels <= cal_threshold), np.zeros(m)]),
                "if_test": np.concatenate([np.zeros(n, dtype=int), np.ones(m, dtype=int)]),
            })
            .sort_values("mu", ascending=True)
            .reset_index(drop=True)
        )
    else:
        df = (
            pd.DataFrame({
                "mu": np.concatenate([cal_scores, test_scores]),
                "L": np.concatenate([(cal_labels > cal_threshold).astype(int), np.zeros(m, dtype=int)]),
                "if_test": np.concatenate([np.zeros(n, dtype=int), np.ones(m, dtype=int)]),
            })
            .sort_values("mu", ascending=True)
            .reset_index(drop=True)
        )

    # Calibration residuals for split-conformal
    # z_cal ~ model-specific standardization; F_cal = Phi(z_cal)
    z_cal = _z_from_time(model_family, cal_labels, cal_loc, cal_scale)
    F_cal = norm.cdf(z_cal)
    neg_cdf = -F_cal  # in (-1, 0); convenient for upper-quantiles with "higher" method

    # outputs in same order as unsel_idx
    hat_LCB = np.zeros(len(unsel_idx), dtype=float)
    cum_test_all = np.cumsum(df["if_test"].values)  # length n+m

    # per-j (JOMI)
    for out_pos, j in enumerate(unsel_idx):
        # For each cut τ, four “worlds” R_ab:
        #   a ∈ {0,1}: include the global +1 regularizer (as in selection)
        #   b ∈ {0,1}: count unit j as a short when selected
        ind_test_leq_mu = (test_scores[j] <= df["mu"].values).astype(int)
        denom = np.maximum(1, 1 + cum_test_all - ind_test_leq_mu) * (m / (n + 1.0))

        if w_ipcw is not None:
            cum_Z = np.cumsum(df["Z"].values)
            R00 = (cum_Z) / denom
            R01 = (cum_Z + ind_test_leq_mu) / denom
            R10 = (1.0 + cum_Z) / denom
            R11 = (1.0 + cum_Z + ind_test_leq_mu) / denom
        else:
            cal_err = ((df["L"].values == 0) & (df["if_test"].values == 0)).astype(int)
            cum_err = np.cumsum(cal_err)
            R00 = (cum_err) / denom
            R01 = (cum_err + ind_test_leq_mu) / denom
            R10 = (1.0 + cum_err) / denom
            R11 = (1.0 + cum_err + ind_test_leq_mu) / denom

        # Most permissive feasible τ in each world
        tau_00 = df.loc[alpha >= R00, "mu"].max() if np.any(alpha >= R00) else -np.inf
        tau_01 = df.loc[alpha >= R01, "mu"].max() if np.any(alpha >= R01) else -np.inf
        tau_10 = df.loc[alpha >= R10, "mu"].max() if np.any(alpha >= R10) else -np.inf
        tau_11 = df.loc[alpha >= R11, "mu"].max() if np.any(alpha >= R11) else -np.inf

        # Calibration index sets for split-conformal residual quantiles
        in_cal_short = cal_labels <= cal_threshold  # L_i = 0
        in_cal_long = ~in_cal_short  # L_i = 1

        worse_00 = cal_scores > tau_00
        worse_01 = cal_scores > tau_01
        worse_10 = cal_scores > tau_10
        worse_11 = cal_scores > tau_11

        Rj_1 = np.where((in_cal_short & worse_01) | (in_cal_long & worse_11))[0]  # assume j is long
        Rj_0 = np.where((in_cal_short & worse_00) | (in_cal_long & worse_10))[0]  # assume j is short

        # Quantiles on -F_cal; +inf ensures defined quantile if set is empty
        eta_1 = np.quantile(np.concatenate([neg_cdf[Rj_1], [np.inf]]), q=1 - alpha, method="higher")
        eta_0 = np.quantile(np.concatenate([neg_cdf[Rj_0], [np.inf]]), q=1 - alpha, method="higher")

        # Back to Phi^{-1} inputs, clipped
        p_1 = float(np.clip(-eta_1, clip_ppf, 1 - clip_ppf))
        p_0 = float(np.clip(-eta_0, clip_ppf, 1 - clip_ppf))

        # Map back to time using the model's inverse-CDF at unit j
        # (handle scalar vs per-sample scale consistently)
        scale_j = test_scale[j] if np.ndim(test_scale) else float(test_scale)
        L_1 = _inv_time_from_ppf(model_family, p_1, loc=float(test_loc[j]), scale=float(scale_j))
        L_0 = _inv_time_from_ppf(model_family, p_0, loc=float(test_loc[j]), scale=float(scale_j))

        # Final LCB for j: if the "long" world already exceeds c_j, take max(c_j, L_0); else L_1.
        hat_LCB[out_pos] = max(test_threshold[j], L_0) if (test_threshold[j] < L_1) else L_1

    return hat_LCB
