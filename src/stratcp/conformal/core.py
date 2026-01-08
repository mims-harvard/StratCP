"""
Core conformal prediction functionality.

This module implements the main conformal prediction function that constructs
prediction sets using pre-computed nonconformity scores and reference sets.
"""

import numpy as np


def conformal(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float,
    nonempty: bool = True,
    test_max_id: np.ndarray | None = None,
    if_in_ref: list[np.ndarray] | None = None,
    class_conditional: bool = False,
    rand: bool = True,
    cal_weights: np.ndarray | None = None,
    test_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute conformal prediction sets with optional reference set conditioning.

    Constructs prediction sets using p-values computed from nonconformity scores,
    with support for post-selection inference via reference sets (JOMI).

    Parameters
    ----------
    cal_scores : np.ndarray
        Nonconformity scores V(X_i, Y_i) for calibration data (n,)
    test_scores : np.ndarray
        Nonconformity scores V(X_{n+j}, y) for test data (m, nclass)
    cal_labels : np.ndarray
        True labels Y_i for calibration data (n,)
    alpha : float
        Miscoverage level (e.g., 0.1 for 90% coverage)
    nonempty : bool, default=True
        If True, ensure prediction sets always include the top predicted class
    test_max_id : np.ndarray, optional
        Indices of maximum predicted class for each test sample (m,)
        Required if nonempty=True
    if_in_ref : list[np.ndarray], optional
        Reference set indicators. if_in_ref[y][j,i] indicates whether
        calibration sample i is in reference set for test sample j and label y.
        If None, uses all calibration samples (vanilla conformal prediction)
    class_conditional : bool, default=False
        If True, provides class-conditional coverage guarantees
    rand : bool, default=True
        If True, randomize p-values for exact finite-sample coverage
    cal_weights : np.ndarray, optional
        Importance weights w(X_i) for calibration samples (n,)
        Used for covariate shift adjustment
    test_weights : np.ndarray, optional
        Importance weights w(X_{n+j}) for test samples (m,)
        Used for covariate shift adjustment

    Returns
    -------
    prediction_sets : np.ndarray
        Binary matrix indicating prediction sets (m, nclass)
    cov : np.ndarray
        Binary vector indicating coverage for each test sample (m,)
    eff : np.ndarray
        Prediction set sizes for each test sample (m,)

    Notes
    -----
    - Without reference sets (if_in_ref=None): provides standard conformal prediction
    - With reference sets: provides valid post-selection inference (JOMI)
    - Selection-conditional coverage: P(Y_j in C_j | j in S) >= 1-alpha
    - Class-conditional coverage: P(Y_j in C_j | Y_j=y) >= 1-alpha for all y
    - Marginal coverage: P(Y_j in C_j) >= 1-alpha
    """
    n = len(cal_scores)
    m = test_scores.shape[0]
    nclass = test_scores.shape[1]

    cal_weights = np.ones(n) if cal_weights is None else cal_weights
    test_weights = np.ones(m) if test_weights is None else test_weights

    # Initialize reference sets to all ones if not provided (vanilla CP)
    if if_in_ref is None:
        if_in_ref = [np.ones((m, n)) for _ in range(nclass)]

    # Use p-values to define prediction sets
    pvals = np.ones((m, nclass))
    Us = np.random.uniform(low=0, high=1, size=m) if rand else np.ones(m)

    if class_conditional:
        # Class-conditional coverage
        for i in range(m):
            for j in range(nclass):
                pvals[i, j] = (
                    np.sum(cal_weights * (cal_scores < test_scores[i, j]) * if_in_ref[j][i, :] * (cal_labels == j))
                    + Us[i]
                    * (
                        np.sum(cal_weights * (cal_scores == test_scores[i, j]) * (cal_labels == j) * if_in_ref[j][i, :])
                        + test_weights[i]
                    )
                ) / (test_weights[i] + np.sum(cal_weights * (cal_labels == j) * if_in_ref[j][i, :]))
    else:
        # Marginal coverage (plain conformal prediction)
        for i in range(m):
            for j in range(nclass):
                pvals[i, j] = (
                    np.sum(cal_weights * (cal_scores < test_scores[i, j]) * if_in_ref[j][i, :])
                    + Us[i]
                    * (np.sum(cal_weights * (cal_scores == test_scores[i, j]) * if_in_ref[j][i, :]) + test_weights[i])
                ) / (test_weights[i] + np.sum(cal_weights * if_in_ref[j][i, :]))

    prediction_sets = pvals <= (1 - alpha)

    if nonempty:
        if test_max_id is None:
            raise ValueError("test_max_id must be provided when nonempty=True")
        prediction_sets[np.arange(m), test_max_id] = True

    return prediction_sets
