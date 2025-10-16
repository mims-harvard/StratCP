"""
Unit tests for nonconformity score functions.
"""

import numpy as np
import pytest

from stratcp.conformal.scores import (
    compute_score_aps,
    compute_score_raps,
    compute_score_tps,
    get_consec_ordering,
)


class TestComputeScoreTPS:
    """Tests for TPS score computation."""

    def test_basic_tps(self):
        """Test basic TPS score computation."""
        np.random.seed(42)
        n, m, nclass = 100, 50, 5

        cal_smx = np.random.dirichlet(np.ones(nclass), n)
        val_smx = np.random.dirichlet(np.ones(nclass), m)
        cal_labels = np.random.randint(0, nclass, n)
        val_labels = np.random.randint(0, nclass, m)

        cal_scores, val_all_scores = compute_score_tps(cal_smx, val_smx, cal_labels, val_labels)

        # Check shapes
        assert cal_scores.shape == (n,)
        assert val_all_scores.shape == (m, nclass)

        # Check score bounds (should be in [0, 1] for TPS)
        assert np.all(cal_scores >= 0) and np.all(cal_scores <= 1)
        assert np.all(val_all_scores >= 0) and np.all(val_all_scores <= 1)

    def test_nonempty_tps(self):
        """Test that nonempty=True sets argmax scores to 0."""
        np.random.seed(42)
        n, m, nclass = 50, 30, 5

        cal_smx = np.random.dirichlet(np.ones(nclass), n)
        val_smx = np.random.dirichlet(np.ones(nclass), m)
        cal_labels = np.argmax(cal_smx, axis=1)  # Use argmax as labels
        val_labels = np.argmax(val_smx, axis=1)

        cal_scores, val_all_scores = compute_score_tps(cal_smx, val_smx, cal_labels, val_labels, nonempty=True)

        # Argmax should have score 0
        val_max_id = np.argmax(val_smx, axis=1)
        assert np.all(val_all_scores[np.arange(m), val_max_id] == 0)
        assert np.all(cal_scores == 0)  # Since labels = argmax


class TestComputeScoreAPS:
    """Tests for APS score computation."""

    def test_basic_aps(self):
        """Test basic APS score computation."""
        np.random.seed(42)
        n, m, nclass = 100, 50, 5

        cal_smx = np.random.dirichlet(np.ones(nclass), n)
        val_smx = np.random.dirichlet(np.ones(nclass), m)
        cal_labels = np.random.randint(0, nclass, n)
        val_labels = np.random.randint(0, nclass, m)

        cal_scores, val_all_scores = compute_score_aps(cal_smx, val_smx, cal_labels, val_labels)

        # Check shapes
        assert cal_scores.shape == (n,)
        assert val_all_scores.shape == (m, nclass)

        # APS scores should be in [0, 1] since they're cumulative probabilities
        assert np.all(cal_scores >= 0) and np.all(cal_scores <= 1)
        assert np.all(val_all_scores >= 0) and np.all(val_all_scores <= 1)

    def test_aps_monotonicity(self):
        """Test that APS scores are monotonic in cumulative probability."""
        np.random.seed(42)
        n, nclass = 10, 5

        cal_smx = np.random.dirichlet(np.ones(nclass), n)
        val_smx = np.random.dirichlet(np.ones(nclass), 1)
        cal_labels = np.random.randint(0, nclass, n)
        val_labels = np.array([0])

        _, val_all_scores = compute_score_aps(cal_smx, val_smx, cal_labels, val_labels, nonempty=False)

        # Scores should increase with rank (except for equal probabilities)
        val_pi = val_smx.argsort(1)[:, ::-1]
        sorted_scores = val_all_scores[0, val_pi[0]]

        # Check that cumulative probabilities are non-decreasing
        assert np.all(np.diff(sorted_scores) >= -1e-10)  # Allow small numerical errors


class TestComputeScoreRAPS:
    """Tests for RAPS score computation."""

    def test_basic_raps(self):
        """Test basic RAPS score computation."""
        np.random.seed(42)
        n, m, nclass = 100, 50, 10

        cal_smx = np.random.dirichlet(np.ones(nclass), n)
        val_smx = np.random.dirichlet(np.ones(nclass), m)
        cal_labels = np.random.randint(0, nclass, n)
        val_labels = np.random.randint(0, nclass, m)

        cal_scores, val_all_scores = compute_score_raps(cal_smx, val_smx, cal_labels, val_labels, lam_reg=0.01)

        # Check shapes
        assert cal_scores.shape == (n,)
        assert val_all_scores.shape == (m, nclass)

        # RAPS scores should be slightly larger than APS due to regularization
        cal_scores_aps, val_all_scores_aps = compute_score_aps(cal_smx, val_smx, cal_labels, val_labels)
        assert np.all(cal_scores >= cal_scores_aps - 1e-10)  # Allow numerical precision


class TestGetConsecOrdering:
    """Tests for consecutive ordering function."""

    def test_basic_ordering(self):
        """Test basic consecutive ordering."""
        smx = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
        ordering = get_consec_ordering(smx)

        # Should start with argmax
        assert ordering[0] == 1
        assert len(ordering) == len(smx)
        assert len(set(ordering)) == len(smx)  # All unique

    def test_ordering_consecutiveness(self):
        """Test that ordering expands consecutively from argmax."""
        smx = np.array([0.05, 0.1, 0.6, 0.15, 0.1])
        ordering = get_consec_ordering(smx)

        # Should expand from index 2 (argmax)
        assert ordering[0] == 2

        # Each subsequent element should be adjacent to current range
        for i in range(1, len(ordering)):
            current_range = ordering[:i]
            new_elem = ordering[i]
            assert (new_elem == min(current_range) - 1) or (new_elem == max(current_range) + 1)

    def test_uniform_probabilities(self):
        """Test ordering with uniform probabilities."""
        smx = np.ones(5) / 5
        ordering = get_consec_ordering(smx)

        # Should still produce valid consecutive ordering
        assert len(ordering) == 5
        assert len(set(ordering)) == 5
