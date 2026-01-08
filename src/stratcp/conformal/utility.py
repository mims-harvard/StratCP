"""
Utility-aware conformal prediction using label similarity.

This module provides score functions that leverage similarity matrices between
labels to produce more interpretable and coherent prediction sets.
"""

from typing import Optional, Tuple

import numpy as np


def score_expand_max_sim(
    pred_probs: np.ndarray, sim_mat: np.ndarray, k_max: int = 3, null_lab: Optional[int] = None
) -> np.ndarray:
    """
    Compute scores by greedily expanding based on maximum similarity.

    Starting from the highest predicted class, greedily adds the most similar
    class among the top-K candidates at each step.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted class probabilities (m, n_classes)
    sim_mat : np.ndarray
        Similarity matrix between classes (n_classes, n_classes)
        Higher values = more similar
    k_max : int, default=3
        Number of top candidates to consider at each expansion step
    null_lab : int, optional
        Index of null/background class (if any) to handle specially

    Returns
    -------
    scores : np.ndarray
        Nonconformity scores for all classes (m, n_classes)

    Notes
    -----
    The algorithm:
    1. Start with the highest probability class
    2. At each step, search among top-K candidates for the one most similar
       to any already selected class
    3. Continue until all classes are ordered
    4. Compute cumulative probabilities in this order
    """
    nn, nnclass = pred_probs.shape
    scores = np.zeros((nn, nnclass))

    for i in range(nn):
        pred_prob = pred_probs[i, :]
        jmax = np.argmax(pred_prob)
        ordered_idx = np.argsort(pred_prob)[::-1]

        # Greedy expansion based on similarity
        ord_list = [jmax]
        compare_list = [jmax]

        if null_lab is not None:
            if null_lab in compare_list:
                compare_list.remove(null_lab)

        cand_list = np.delete(ordered_idx, 0)

        while len(cand_list) > 1:
            K = np.min((k_max, len(cand_list)))

            if null_lab is None or len(ord_list) > 1:
                # Standard case: find most similar to existing classes
                sim_exist = np.atleast_2d(sim_mat[compare_list, :][:, cand_list[0:K]])
                j_next = np.argmax(np.max(sim_exist, axis=0))
                ord_list.append(cand_list[j_next])
                if cand_list[j_next] != null_lab:
                    compare_list.append(cand_list[j_next])
                cand_list = np.delete(cand_list, j_next)
            else:
                # Special handling when null label is top predicted
                if jmax == null_lab and len(ord_list) == 1:
                    j_next = np.argmax(pred_prob[cand_list[0:K]])
                    ord_list.append(cand_list[j_next])
                    if cand_list[j_next] != null_lab:
                        compare_list.append(cand_list[j_next])
                    cand_list = np.delete(cand_list, j_next)
                else:
                    K = np.min((k_max, len(cand_list)))
                    sim_exist = np.atleast_2d(sim_mat[compare_list, :][:, cand_list[0:K]])
                    j_next = np.argmax(np.max(sim_exist, axis=0))
                    if cand_list[j_next] != null_lab:
                        compare_list.append(cand_list[j_next])
                    ord_list.append(cand_list[j_next])
                    cand_list = np.delete(cand_list, j_next)

        ord_list.append(cand_list[0])
        scores[i, ord_list] = np.cumsum(pred_prob[ord_list])

    return scores


def score_expand_weighted_sim(
    pred_probs: np.ndarray, sim_mat: np.ndarray, k_max: Optional[int] = None, null_lab: Optional[int] = None
) -> np.ndarray:
    """
    Compute scores by expanding based on weighted similarity.

    At each step, selects the candidate that maximizes the weighted average
    of similarity and prediction probability.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted class probabilities (m, n_classes)
    sim_mat : np.ndarray
        Similarity matrix between classes (n_classes, n_classes)
    k_max : int, optional
        Number of top candidates to consider. If None, considers all.
    null_lab : int, optional
        Index of null/background class to handle specially

    Returns
    -------
    scores : np.ndarray
        Nonconformity scores for all classes (m, n_classes)

    Notes
    -----
    This method balances similarity with prediction confidence by computing:
        score[candidate] = mean(similarity[candidate, existing] * prob[candidate])

    Generally produces more coherent prediction sets than max similarity alone.
    """
    nn, nnclass = pred_probs.shape
    scores = np.zeros((nn, nnclass))

    for i in range(nn):
        pred_prob = pred_probs[i, :]
        jmax = np.argmax(pred_prob)
        ordered_idx = np.argsort(pred_prob)[::-1]

        ord_list = [jmax]
        compare_list = [jmax]

        if null_lab is not None:
            if null_lab in compare_list:
                compare_list.remove(null_lab)

        cand_list = np.delete(ordered_idx, 0)

        while len(cand_list) > 1:
            if k_max is None:
                K = len(cand_list)
            else:
                K = np.min((k_max, len(cand_list)))

            if null_lab is None or len(ord_list) > 1:
                # Weight similarity by prediction probability
                sim_exist = np.atleast_2d(sim_mat[compare_list, :][:, cand_list[0:K]])
                prob_exist = pred_prob[cand_list[0:K]]
                wt_prob = sim_exist * prob_exist
                j_next = np.argmax(np.mean(wt_prob, axis=0))
                ord_list.append(cand_list[j_next])
                if null_lab is not None:
                    if cand_list[j_next] != null_lab:
                        compare_list.append(cand_list[j_next])
                cand_list = np.delete(cand_list, j_next)
            else:
                # Special handling for null label
                if jmax == null_lab and len(ord_list) == 1:
                    j_next = np.argmax(np.max(pred_prob[cand_list[0:K]], axis=0))
                    ord_list.append(cand_list[j_next])
                    if cand_list[j_next] != null_lab:
                        compare_list.append(cand_list[j_next])
                    cand_list = np.delete(cand_list, j_next)
                else:
                    sim_exist = np.atleast_2d(sim_mat[compare_list, :][:, cand_list[0:K]])
                    prob_exist = pred_prob[cand_list[0:K]]
                    wt_prob = sim_exist * prob_exist
                    j_next = np.argmax(np.mean(wt_prob, axis=0))
                    if cand_list[j_next] != null_lab:
                        compare_list.append(cand_list[j_next])
                    ord_list.append(cand_list[j_next])
                    cand_list = np.delete(cand_list, j_next)

        ord_list.append(cand_list[0])
        scores[i, ord_list] = np.cumsum(pred_prob[ord_list])

    return scores


def compute_score_utility(
    cal_probs: np.ndarray,
    test_probs: Optional[np.ndarray],
    cal_labels: np.ndarray,
    sim_mat: np.ndarray,
    method: str = "weighted",
    k_max: int = 3,
    nonempty: bool = True,
    null_lab: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute utility-aware nonconformity scores using label similarity.

    Parameters
    ----------
    cal_probs : np.ndarray
        Predicted probabilities for calibration data (n, n_classes)
    test_probs : np.ndarray or None
        Predicted probabilities for test data (m, n_classes)
    cal_labels : np.ndarray
        True labels for calibration data (n,)
    test_labels : np.ndarray or None
        True labels for test data (m,). Currently unused.
    sim_mat : np.ndarray
        Similarity matrix between classes (n_classes, n_classes)
        Values should be in [0, 1] with higher = more similar
    method : {'weighted', 'greedy'}, default='weighted'
        Expansion method:
        - 'weighted': Balance similarity and probability (recommended)
        - 'greedy': Pure max similarity
    k_max : int, default=3
        Number of candidates to consider at each step (for 'greedy')
    nonempty : bool, default=True
        Force non-empty prediction sets
    null_lab : int, optional
        Index of null/background class

    Returns
    -------
    cal_scores : np.ndarray
        Scores for calibration data (n,)
    test_scores : np.ndarray or None
        Scores for test data (m, n_classes) if `test_probs` is not None,
        otherwise None.

    Examples
    --------
    >>> # Create similarity matrix (e.g., from medical ontology)
    >>> sim_mat = np.array([
    ...     [1.0, 0.8, 0.3],
    ...     [0.8, 1.0, 0.4],
    ...     [0.3, 0.4, 1.0]
    ... ])
    >>>
    >>> cal_scores, test_scores = compute_score_utility(
    ...     cal_probs, test_probs, cal_labels, test_labels,
    ...     sim_mat, method='weighted'
    ... )
    """
    n = cal_labels.shape[0]
    m = 0 if test_probs is None else test_probs.shape[0]

    # Compute expansion scores
    if method == "greedy":
        cal_scores_full = score_expand_max_sim(cal_probs, sim_mat, k_max, null_lab)
        test_scores = None if test_probs is None else score_expand_max_sim(test_probs, sim_mat, k_max, null_lab)
    elif method == "weighted":
        cal_scores_full = score_expand_weighted_sim(cal_probs, sim_mat, None, null_lab)
        test_scores = None if test_probs is None else score_expand_weighted_sim(test_probs, sim_mat, None, null_lab)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'weighted' or 'greedy'.")

    # Extract calibration scores for true labels
    cal_scores = cal_scores_full[np.arange(n), cal_labels]

    # Enforce non-empty sets
    if nonempty:
        # Calibration: ensure top-1 label is always in the set
        cal_max_id = np.argmax(cal_probs, axis=1)
        cal_scores[cal_labels == cal_max_id] = 0.0

        # Test: ensure top-1 label is always in the set
        if test_scores is not None and m > 0:
            test_max_id = np.argmax(test_probs, axis=1)
            test_scores[np.arange(m), test_max_id] = 0.0

    return cal_scores, test_scores


def eval_similarity(
    pred_sets: np.ndarray, sim_mat: np.ndarray, null_lab: Optional[int] = None, off_diag: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Evaluate average pairwise similarity within prediction sets.

    This metric assesses how coherent/similar the classes in each prediction
    set are to each other.

    Parameters
    ----------
    pred_sets : np.ndarray
        Binary prediction set matrix (m, n_classes)
    sim_mat : np.ndarray
        Similarity matrix between classes (n_classes, n_classes)
    null_lab : int, optional
        Null label to exclude from similarity computation
    off_diag : bool, default=True
        If True, exclude diagonal (self-similarity) from average

    Returns
    -------
    avg_sim : np.ndarray
        Average similarity for each prediction set (m,)
    overall_sim : float
        Overall average similarity across all sets
    """
    avg_sim = np.zeros(pred_sets.shape[0])

    for i in range(pred_sets.shape[0]):
        idx_in = np.where(pred_sets[i, :])[0]

        if null_lab is not None:
            idx_in = idx_in[idx_in != null_lab]

        if len(idx_in) > 1:
            submat = sim_mat[idx_in, :][:, idx_in]
            if off_diag:
                # Average pairwise similarity (excluding self)
                avg_sim[i] = (np.sum(submat) - np.sum(np.diagonal(submat))) / (len(idx_in) * (len(idx_in) - 1))
            else:
                # Average including diagonal
                avg_sim[i] = np.mean(submat)
        else:
            avg_sim[i] = np.nan

    overall_sim = np.nanmean(avg_sim)
    return avg_sim, overall_sim
