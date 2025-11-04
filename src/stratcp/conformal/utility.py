"""
Utility-aware conformal prediction using label similarity.

This module provides score functions that leverage similarity matrices between
labels to produce more interpretable and coherent prediction sets.
"""

import numpy as np


def score_expand_max_sim(
    pred_probs: np.ndarray, sim_mat: np.ndarray, k_max: int = 3, null_lab: int | None = None
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
    pred_probs: np.ndarray, sim_mat: np.ndarray, k_max: int | None = None, null_lab: int | None = None
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
    test_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    sim_mat: np.ndarray,
    method: str = "weighted",
    k_max: int = 3,
    nonempty: bool = True,
    null_lab: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute utility-aware nonconformity scores using label similarity.

    Parameters
    ----------
    cal_probs : np.ndarray
        Predicted probabilities for calibration data (n, n_classes)
    test_probs : np.ndarray
        Predicted probabilities for test data (m, n_classes)
    cal_labels : np.ndarray
        True labels for calibration data (n,)
    test_labels : np.ndarray
        True labels for test data (m,)
    sim_mat : np.ndarray
        Similarity matrix between classes (n_classes, n_classes)
        Values should be in [0, 1] with higher = more similar
    method : {'weighted', 'greedy'}, default='weighted'
        Expansion method:
        - 'weighted': Balance similarity and probability (recommended)
        - 'greedy': Pure max similarity
    k_max : int, default=3
        Number of candidates to consider at each step
    nonempty : bool, default=True
        Force non-empty prediction sets
    null_lab : int, optional
        Index of null/background class

    Returns
    -------
    cal_scores : np.ndarray
        Scores for calibration data (n,)
    test_scores : np.ndarray
        Scores for test data (m, n_classes)

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
    n = len(cal_labels)
    m = len(test_labels)

    # Compute expansion scores
    if method == "greedy":
        cal_scores_full = score_expand_max_sim(cal_probs, sim_mat, k_max, null_lab)
        test_scores = score_expand_max_sim(test_probs, sim_mat, k_max, null_lab)
    elif method == "weighted":
        cal_scores_full = score_expand_weighted_sim(cal_probs, sim_mat, None, null_lab)
        test_scores = score_expand_weighted_sim(test_probs, sim_mat, None, null_lab)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'weighted' or 'greedy'.")

    # Extract calibration scores for true labels
    cal_scores = cal_scores_full[np.arange(n), cal_labels]

    # Enforce non-empty sets
    if nonempty:
        cal_max_id = np.argmax(cal_probs, axis=1)
        cal_scores[cal_labels == cal_max_id] = 0

        test_max_id = np.argmax(test_probs, axis=1)
        test_scores[np.arange(m), test_max_id] = 0

    return cal_scores, test_scores


def eval_similarity(
    pred_sets: np.ndarray, sim_mat: np.ndarray, null_lab: int | None = None, off_diag: bool = True
) -> tuple[np.ndarray, float]:
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


# def calculate_conformal_metrics(
#     method: str,
#     cov: np.ndarray,
#     size: np.ndarray,
#     set_matrix: np.ndarray,
#     test_labels_group: np.ndarray,
#     sel_cover_sum: float,
#     m: int,
#     unsel_idx: np.ndarray | None = None
# ) -> tuple[float, float, float, float | None, float | None]:
#     """
#     Compute conformal metrics for a SINGLE method/version (one of: 'tps', 'aps', or 'raps').

#     Args:
#         method : {'tps', 'aps', 'raps'}
#             Which conformal set variant the inputs correspond to. This is used only for clarity
#             and basic validation in logs/messages (the computation is generic).
#         cov : np.ndarray, shape (n_unselected,)
#             Binary coverage indicator for the *unselected* samples for the chosen method.
#             Each entry is 1 if the ground-truth label is covered by that sample's prediction set,
#             else 0.
#         size : np.ndarray, shape (n_unselected,)
#             Prediction set sizes for the *unselected* samples for the chosen method.
#         set_matrix : np.ndarray, shape (n_unselected, n_classes)
#             Row i is a 0/1 indicator vector for the prediction set of the i-th *unselected* sample.
#             Used to compute conditional coverage stratified by set size.
#         test_labels_group : np.ndarray, shape (m,)
#             Ground-truth labels for the entire evaluation group of size m (integer class indices).
#         sel_cover_sum : float
#             Sum of coverage indicators **for the selected samples only**.
#             This is combined with the coverage of the `cov` vector (unselected) to produce
#             marginal coverage across all m samples.
#         m : int
#             Total number of samples in the evaluation group (selected + unselected).
#         unsel_idx : Optional[np.ndarray], shape (n_unselected,), default=None
#             If provided, gives the indices of the unselected samples relative to the full
#             `test_labels_group`. When provided, set-size–conditioned coverage uses
#             `test_labels_group[unsel_idx]` so that rows in `set_matrix` align with the
#             corresponding labels subset. If omitted, it is assumed that `cov`, `size`,
#             and `set_matrix` already align with `test_labels_group` indexing.

#     Returns:
#         mgn_cov : float
#             Marginal coverage across all m samples for the chosen method.
#         mgn_size : float
#             Marginal prediction set size across all m samples for the chosen method.
#             (Adds the count of selected samples to the sum of sizes for unselected samples.)
#         cond_cov_unselected : float
#             Mean coverage across the unselected subset only (i.e., mean of `cov`).
#             Returns `np.nan` if there are no unselected samples.
#         cond_cov_set_size_one : Optional[float]
#             Coverage among unselected samples whose set size == 1. `np.nan` if no such samples.
#         cond_cov_set_size_two : Optional[float]
#             Coverage among unselected samples whose set size == 2. `np.nan` if no such samples.
#     """
#     # Basic validation & shaping
#     method = str(method).lower()
#     if method not in {"tps", "aps", "raps"}:
#         raise ValueError("`method` must be one of {'tps', 'aps', 'raps'}.")

#     cov = np.asarray(cov)
#     size = np.asarray(size)
#     set_matrix = np.asarray(set_matrix)
#     test_labels_group = np.asarray(test_labels_group)

#     if cov.ndim != 1 or size.ndim != 1:
#         raise ValueError("`cov` and `size` must be 1-D arrays (for unselected samples).")
#     if set_matrix.ndim != 2:
#         raise ValueError("`set_matrix` must be 2-D: (n_unselected, n_classes).")
#     if cov.shape[0] != size.shape[0] or cov.shape[0] != set_matrix.shape[0]:
#         raise ValueError("`cov`, `size`, and `set_matrix` must share the same first dimension.")
#     if test_labels_group.shape[0] != m:
#         raise ValueError("`test_labels_group` length must equal `m`.")

#     n_unselected = cov.shape[0]
#     if n_unselected > m:
#         raise ValueError("Number of unselected samples cannot exceed `m`.")

#     # # If provided, unsel_idx maps unselected rows to their indices in the full group
#     # if unsel_idx is not None:
#     #     unsel_idx = np.asarray(unsel_idx)
#     #     if unsel_idx.ndim != 1 or unsel_idx.shape[0] != n_unselected:
#     #         raise ValueError("`unsel_idx` must be 1-D with length equal to n_unselected.")
#     #     # Labels corresponding to the rows of set_matrix (unselected ordering)
#     #     labels_for_rows = test_labels_group[unsel_idx]
#     # else:
#     #     # Assume `cov`, `size`, and `set_matrix` already align with the labels given
#     #     # (e.g., test_labels_group is a subset or in the same order)
#     #     # If this is not the case in your pipeline, pass `unsel_idx`.
#     #     labels_for_rows = test_labels_group[:n_unselected]

#     # Marginal coverage
#     # Coverage over all m = (coverage on unselected) + (coverage sum on selected), divided by m.
#     # `cov` is 0/1 for each unselected sample; sum(cov) is coverage count on unselected.
#     mgn_cov = (float(np.sum(cov)) + float(sel_cover_sum)) / float(m) if m > 0 else np.nan

#     # Marginal size
#     # For marginal size across all m, add:
#     #   - sum of sizes over unselected samples, plus
#     #   - the count of selected samples (each contributes 1 to the size aggregate
#     #     in the original code's convention).
#     n_selected = m - n_unselected
#     mgn_size = (float(np.sum(size)) + float(n_selected)) / float(m) if m > 0 else np.nan

#     # # Conditional coverage: unselected only
#     # cond_cov_unselected = float(np.mean(cov)) if n_unselected > 0 else np.nan

#     # # Helper for coverage conditional on set size
#     # def _coverage_at_size(k: int) -> float:
#     #     """Coverage among unselected samples with set size == k (np.nan if none)."""
#     #     idx = np.where(size == k)[0]
#     #     if idx.size == 0:
#     #         return np.nan
#     #     # For each such row i, check whether the true label is in its set:
#     #     # set_matrix[i, labels_for_rows[i]] is 1 if covered, else 0.
#     #     # (Assumes labels are integer class indices aligned with columns.)
#     #     hits = set_matrix[idx, labels_for_rows[idx]]
#     #     return float(np.sum(hits) / idx.size)

#     # cond_cov_set_size_one = _coverage_at_size(1)
#     # cond_cov_set_size_two = _coverage_at_size(2)

#     return mgn_cov, mgn_size, cond_cov_unselected, cond_cov_set_size_one, cond_cov_set_size_two
