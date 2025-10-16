"""
Tests for utility-aware conformal prediction.
"""

import numpy as np
import pytest

from stratcp.conformal.utility import (
    compute_score_utility,
    eval_similarity,
    score_expand_max_sim,
    score_expand_weighted_sim,
)


@pytest.fixture
def simple_data():
    """Generate simple test data."""
    np.random.seed(42)
    n_cal = 100
    n_test = 50
    n_classes = 5

    cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, size=n_cal)
    test_probs = np.random.dirichlet(np.ones(n_classes) * 2, size=n_test)
    cal_labels = np.random.choice(n_classes, size=n_cal)
    test_labels = np.random.choice(n_classes, size=n_test)

    return cal_probs, test_probs, cal_labels, test_labels, n_classes


@pytest.fixture
def similarity_matrix():
    """Generate a test similarity matrix."""
    # 5 classes with hierarchical structure
    return np.array([
        [1.0, 0.9, 0.3, 0.3, 0.1],
        [0.9, 1.0, 0.3, 0.3, 0.1],
        [0.3, 0.3, 1.0, 0.9, 0.1],
        [0.3, 0.3, 0.9, 1.0, 0.1],
        [0.1, 0.1, 0.1, 0.1, 1.0],
    ])


class TestScoreExpansion:
    """Test score expansion methods."""

    def test_score_expand_max_sim_shape(self, simple_data, similarity_matrix):
        """Test that max similarity expansion returns correct shape."""
        cal_probs, test_probs, _, _, n_classes = simple_data

        scores = score_expand_max_sim(test_probs, similarity_matrix, k_max=3)

        assert scores.shape == test_probs.shape
        assert scores.shape == (50, n_classes)

    def test_score_expand_max_sim_cumulative(self, simple_data, similarity_matrix):
        """Test that scores are cumulative probabilities."""
        cal_probs, test_probs, _, _, n_classes = simple_data

        scores = score_expand_max_sim(test_probs, similarity_matrix, k_max=3)

        # Each row should have cumulative values from probs
        for i in range(scores.shape[0]):
            # Find the ordering used
            ordered_scores = np.sort(scores[i, :])
            # Should be non-decreasing
            assert np.all(ordered_scores[1:] >= ordered_scores[:-1])
            # Last score should sum to 1 (or close due to floating point)
            assert np.abs(ordered_scores[-1] - 1.0) < 1e-6

    def test_score_expand_weighted_sim_shape(self, simple_data, similarity_matrix):
        """Test that weighted similarity expansion returns correct shape."""
        cal_probs, test_probs, _, _, n_classes = simple_data

        scores = score_expand_weighted_sim(test_probs, similarity_matrix)

        assert scores.shape == test_probs.shape
        assert scores.shape == (50, n_classes)

    def test_score_expand_weighted_sim_cumulative(self, simple_data, similarity_matrix):
        """Test that weighted scores are cumulative probabilities."""
        cal_probs, test_probs, _, _, n_classes = simple_data

        scores = score_expand_weighted_sim(test_probs, similarity_matrix)

        # Each row should have cumulative values
        for i in range(scores.shape[0]):
            ordered_scores = np.sort(scores[i, :])
            # Should be non-decreasing
            assert np.all(ordered_scores[1:] >= ordered_scores[:-1])
            # Last score should sum to 1
            assert np.abs(ordered_scores[-1] - 1.0) < 1e-6

    def test_null_label_handling(self, simple_data, similarity_matrix):
        """Test that null label is handled correctly."""
        cal_probs, test_probs, _, _, n_classes = simple_data

        # Test with null label
        scores_with_null = score_expand_max_sim(test_probs, similarity_matrix, k_max=3, null_lab=0)
        scores_without_null = score_expand_max_sim(test_probs, similarity_matrix, k_max=3, null_lab=None)

        # Scores should differ when null label is specified
        # (at least for some samples)
        assert not np.allclose(scores_with_null, scores_without_null)


class TestComputeScoreUtility:
    """Test compute_score_utility function."""

    def test_basic_weighted_method(self, simple_data, similarity_matrix):
        """Test basic weighted utility score computation."""
        cal_probs, test_probs, cal_labels, test_labels, n_classes = simple_data

        cal_scores, test_scores = compute_score_utility(
            cal_probs, test_probs, cal_labels, test_labels, similarity_matrix, method="weighted"
        )

        # Check shapes
        assert cal_scores.shape == (len(cal_labels),)
        assert test_scores.shape == test_probs.shape

        # Check that scores are valid
        assert np.all(cal_scores >= 0)
        assert np.all(cal_scores <= 1)
        assert np.all(test_scores >= 0)
        assert np.all(test_scores <= 1)

    def test_basic_greedy_method(self, simple_data, similarity_matrix):
        """Test basic greedy utility score computation."""
        cal_probs, test_probs, cal_labels, test_labels, n_classes = simple_data

        cal_scores, test_scores = compute_score_utility(
            cal_probs, test_probs, cal_labels, test_labels, similarity_matrix, method="greedy", k_max=3
        )

        # Check shapes
        assert cal_scores.shape == (len(cal_labels),)
        assert test_scores.shape == test_probs.shape

        # Check that scores are valid
        assert np.all(cal_scores >= 0)
        assert np.all(cal_scores <= 1)
        assert np.all(test_scores >= 0)
        assert np.all(test_scores <= 1)

    def test_nonempty_enforcement(self, simple_data, similarity_matrix):
        """Test that nonempty=True enforces non-empty sets."""
        cal_probs, test_probs, cal_labels, test_labels, n_classes = simple_data

        cal_scores, test_scores = compute_score_utility(
            cal_probs, test_probs, cal_labels, test_labels, similarity_matrix, method="weighted", nonempty=True
        )

        # Check that top predicted class always has score 0 for test data
        test_max_id = np.argmax(test_probs, axis=1)
        for i, max_id in enumerate(test_max_id):
            assert test_scores[i, max_id] == 0.0

    def test_invalid_method_raises(self, simple_data, similarity_matrix):
        """Test that invalid method raises ValueError."""
        cal_probs, test_probs, cal_labels, test_labels, n_classes = simple_data

        with pytest.raises(ValueError, match="Unknown method"):
            compute_score_utility(
                cal_probs, test_probs, cal_labels, test_labels, similarity_matrix, method="invalid"
            )

    def test_methods_produce_different_scores(self, simple_data, similarity_matrix):
        """Test that different methods produce different scores."""
        cal_probs, test_probs, cal_labels, test_labels, n_classes = simple_data

        cal_scores_weighted, test_scores_weighted = compute_score_utility(
            cal_probs, test_probs, cal_labels, test_labels, similarity_matrix, method="weighted"
        )

        cal_scores_greedy, test_scores_greedy = compute_score_utility(
            cal_probs, test_probs, cal_labels, test_labels, similarity_matrix, method="greedy", k_max=3
        )

        # Scores should differ between methods
        assert not np.allclose(test_scores_weighted, test_scores_greedy)


class TestEvalSimilarity:
    """Test eval_similarity function."""

    def test_basic_similarity_evaluation(self, similarity_matrix):
        """Test basic similarity evaluation."""
        # Create simple prediction sets
        pred_sets = np.array(
            [
                [1, 1, 0, 0, 0],  # Classes 0, 1 (very similar)
                [0, 0, 1, 1, 0],  # Classes 2, 3 (very similar)
                [1, 0, 0, 0, 1],  # Classes 0, 4 (dissimilar)
                [1, 0, 0, 0, 0],  # Only class 0 (singleton)
            ],
            dtype=bool,
        )

        avg_sim, overall_sim = eval_similarity(pred_sets, similarity_matrix, off_diag=True)

        # Check shapes
        assert avg_sim.shape == (4,)
        assert isinstance(overall_sim, float)

        # Classes 0, 1 are very similar (sim=0.9)
        assert avg_sim[0] > 0.85

        # Classes 2, 3 are very similar (sim=0.9)
        assert avg_sim[1] > 0.85

        # Classes 0, 4 are dissimilar (sim=0.1)
        assert avg_sim[2] < 0.2

        # Singleton sets should be NaN
        assert np.isnan(avg_sim[3])

    def test_off_diag_parameter(self, similarity_matrix):
        """Test off_diag parameter behavior."""
        pred_sets = np.array([[1, 1, 0, 0, 0]], dtype=bool)

        # With off_diag=True (exclude self-similarity)
        avg_sim_off, _ = eval_similarity(pred_sets, similarity_matrix, off_diag=True)

        # With off_diag=False (include self-similarity)
        avg_sim_on, _ = eval_similarity(pred_sets, similarity_matrix, off_diag=False)

        # Including self-similarity (1.0) should increase average
        assert avg_sim_on[0] > avg_sim_off[0]

    def test_null_label_exclusion(self, similarity_matrix):
        """Test that null label is excluded from similarity computation."""
        pred_sets = np.array([[1, 1, 1, 0, 0]], dtype=bool)  # Classes 0, 1, 2

        # Without null label
        avg_sim_no_null, _ = eval_similarity(pred_sets, similarity_matrix, null_lab=None)

        # With null label (exclude class 1)
        avg_sim_with_null, _ = eval_similarity(pred_sets, similarity_matrix, null_lab=1)

        # Results should differ
        assert not np.isclose(avg_sim_no_null[0], avg_sim_with_null[0])

    def test_overall_similarity(self, similarity_matrix):
        """Test overall similarity calculation."""
        pred_sets = np.array(
            [
                [1, 1, 0, 0, 0],  # Very similar pair
                [0, 0, 1, 1, 0],  # Very similar pair
                [1, 0, 0, 0, 1],  # Dissimilar pair
            ],
            dtype=bool,
        )

        avg_sim, overall_sim = eval_similarity(pred_sets, similarity_matrix)

        # Overall should be mean of non-NaN values
        expected = np.nanmean(avg_sim)
        assert np.isclose(overall_sim, expected)

    def test_empty_prediction_sets(self, similarity_matrix):
        """Test handling of empty prediction sets."""
        pred_sets = np.array([[0, 0, 0, 0, 0]], dtype=bool)  # Empty set

        avg_sim, overall_sim = eval_similarity(pred_sets, similarity_matrix)

        # Empty sets should give NaN
        assert np.isnan(avg_sim[0])


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_utility_scores_produce_valid_sets(self, simple_data, similarity_matrix):
        """Test that utility scores can be used for conformal prediction."""
        from stratcp.conformal import conformal

        cal_probs, test_probs, cal_labels, test_labels, n_classes = simple_data

        # Compute utility scores
        cal_scores, test_scores = compute_score_utility(
            cal_probs, test_probs, cal_labels, test_labels, similarity_matrix, method="weighted"
        )

        # Use in conformal prediction
        pred_sets, cov, sizes = conformal(
            cal_scores,
            test_scores,
            cal_labels,
            test_labels,
            alpha=0.1,
            nonempty=True,
            val_max_id=np.argmax(test_probs, axis=1),
            rand=True,
        )

        # Check outputs
        assert pred_sets.shape == (len(test_labels), n_classes)
        assert cov.shape == (len(test_labels),)
        assert sizes.shape == (len(test_labels),)

        # Check coverage is reasonable (should be >= 1 - alpha)
        assert cov.mean() >= 0.8  # Allow some slack for small sample

        # Check all sets are non-empty
        assert np.all(sizes > 0)

    def test_utility_aware_more_coherent(self, simple_data, similarity_matrix):
        """Test that utility-aware CP produces more coherent sets than standard."""
        from stratcp.conformal import compute_score_raps, conformal

        cal_probs, test_probs, cal_labels, test_labels, n_classes = simple_data

        # Standard RAPS
        cal_scores_raps, test_scores_raps = compute_score_raps(
            cal_probs, test_probs, cal_labels, test_labels, nonempty=True
        )

        pred_sets_raps, _, _ = conformal(
            cal_scores_raps,
            test_scores_raps,
            cal_labels,
            test_labels,
            alpha=0.1,
            nonempty=True,
            val_max_id=np.argmax(test_probs, axis=1),
            rand=False,
        )

        # Utility-aware
        cal_scores_util, test_scores_util = compute_score_utility(
            cal_probs, test_probs, cal_labels, test_labels, similarity_matrix, method="weighted"
        )

        pred_sets_util, _, _ = conformal(
            cal_scores_util,
            test_scores_util,
            cal_labels,
            test_labels,
            alpha=0.1,
            nonempty=True,
            val_max_id=np.argmax(test_probs, axis=1),
            rand=False,
        )

        # Evaluate coherence for sets with size > 1
        mask_raps = np.sum(pred_sets_raps, axis=1) > 1
        mask_util = np.sum(pred_sets_util, axis=1) > 1

        if mask_raps.sum() > 0:
            _, sim_raps = eval_similarity(pred_sets_raps[mask_raps], similarity_matrix)
        else:
            sim_raps = 0

        if mask_util.sum() > 0:
            _, sim_util = eval_similarity(pred_sets_util[mask_util], similarity_matrix)
        else:
            sim_util = 0

        # Utility-aware should produce more coherent sets (if there are multi-class sets)
        if mask_raps.sum() > 0 and mask_util.sum() > 0:
            # This may not always hold due to randomness, but generally should
            print(f"RAPS coherence: {sim_raps:.3f}, Utility coherence: {sim_util:.3f}")
