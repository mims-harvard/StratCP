"""
Utility-Aware Conformal Prediction Example
===========================================

This example demonstrates how to use similarity matrices between labels
to produce more coherent and interpretable prediction sets.

Utility-aware CP is particularly useful when:
1. Labels have semantic relationships (e.g., medical diagnoses, biological categories)
2. You want prediction sets to contain similar/related classes
3. A similarity matrix or ontology is available
"""

import numpy as np
import matplotlib.pyplot as plt
from stratcp import StratifiedCP
from stratcp.conformal import eval_similarity

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. Generate Synthetic Data with Related Classes
# =============================================================================

print("=" * 70)
print("Utility-Aware Conformal Prediction Example")
print("=" * 70)
print()

# Simulate 5 classes with hierarchical relationships:
# Class 0: Category A.1
# Class 1: Category A.2 (similar to 0)
# Class 2: Category B.1
# Class 3: Category B.2 (similar to 2)
# Class 4: Category C (dissimilar to others)

n_cal = 1000
n_test = 500
n_classes = 5

# Generate probabilities with some structure
def generate_probs(n, n_classes):
    """Generate probabilities that respect class similarities."""
    probs = np.random.dirichlet(np.ones(n_classes) * 2, size=n)
    return probs

cal_probs = generate_probs(n_cal, n_classes)
test_probs = generate_probs(n_test, n_classes)

# Generate labels
cal_labels = np.random.choice(n_classes, size=n_cal)
test_labels = np.random.choice(n_classes, size=n_test)

print(f"Calibration set: {n_cal} samples")
print(f"Test set: {n_test} samples")
print(f"Number of classes: {n_classes}")
print()

# =============================================================================
# 2. Define Similarity Matrix
# =============================================================================

print("Creating similarity matrix based on class relationships:")
print("  - Classes 0 & 1 are highly similar (Category A)")
print("  - Classes 2 & 3 are highly similar (Category B)")
print("  - Class 4 is dissimilar to others (Category C)")
print()

# Create similarity matrix reflecting class structure
# Higher values = more similar (range [0, 1])
similarity_matrix = np.array([
    [1.0, 0.9, 0.3, 0.3, 0.1],  # Class 0 (A.1): very similar to 1
    [0.9, 1.0, 0.3, 0.3, 0.1],  # Class 1 (A.2): very similar to 0
    [0.3, 0.3, 1.0, 0.9, 0.1],  # Class 2 (B.1): very similar to 3
    [0.3, 0.3, 0.9, 1.0, 0.1],  # Class 3 (B.2): very similar to 2
    [0.1, 0.1, 0.1, 0.1, 1.0],  # Class 4 (C): dissimilar to all
])

print("Similarity Matrix:")
print(similarity_matrix)
print()

# =============================================================================
# 3. Compare Standard vs Utility-Aware CP
# =============================================================================

print("=" * 70)
print("Comparison: Standard RAPS vs Utility-Aware CP")
print("=" * 70)
print()

# Standard RAPS
print("Running Standard RAPS...")
scp_standard = StratifiedCP(
    score_fn='raps',
    alpha_sel=0.1,
    alpha_cp=0.1,
    nonempty=True,
    rand=True
)
results_standard = scp_standard.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

print(f"Selected samples: {len(results_standard['selected_idx'])}")
print(f"Unselected samples: {len(results_standard['unselected_idx'])}")
if len(results_standard['unselected_idx']) > 0:
    print(f"Avg set size (unselected): {results_standard['set_sizes']['unselected'].mean():.2f}")
    print(f"Coverage (unselected): {results_standard['coverage']['unselected'].mean():.2%}")
print()

# Utility-aware with weighted method
print("Running Utility-Aware CP (weighted method)...")
scp_utility_weighted = StratifiedCP(
    score_fn='utility',
    alpha_sel=0.1,
    alpha_cp=0.1,
    similarity_matrix=similarity_matrix,
    utility_method='weighted',
    nonempty=True,
    rand=True
)
results_utility_weighted = scp_utility_weighted.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

print(f"Selected samples: {len(results_utility_weighted['selected_idx'])}")
print(f"Unselected samples: {len(results_utility_weighted['unselected_idx'])}")
if len(results_utility_weighted['unselected_idx']) > 0:
    print(f"Avg set size (unselected): {results_utility_weighted['set_sizes']['unselected'].mean():.2f}")
    print(f"Coverage (unselected): {results_utility_weighted['coverage']['unselected'].mean():.2%}")
print()

# Utility-aware with greedy method
print("Running Utility-Aware CP (greedy method)...")
scp_utility_greedy = StratifiedCP(
    score_fn='utility',
    alpha_sel=0.1,
    alpha_cp=0.1,
    similarity_matrix=similarity_matrix,
    utility_method='greedy',
    nonempty=True,
    rand=True
)
results_utility_greedy = scp_utility_greedy.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

print(f"Selected samples: {len(results_utility_greedy['selected_idx'])}")
print(f"Unselected samples: {len(results_utility_greedy['unselected_idx'])}")
if len(results_utility_greedy['unselected_idx']) > 0:
    print(f"Avg set size (unselected): {results_utility_greedy['set_sizes']['unselected'].mean():.2f}")
    print(f"Coverage (unselected): {results_utility_greedy['coverage']['unselected'].mean():.2%}")
print()

# =============================================================================
# 4. Evaluate Prediction Set Coherence
# =============================================================================

print("=" * 70)
print("Prediction Set Coherence Analysis")
print("=" * 70)
print()

# Evaluate similarity for unselected samples
if len(results_standard['unselected_idx']) > 0:
    pred_sets_standard = results_standard['prediction_sets']['unselected']
    avg_sim_standard, overall_sim_standard = eval_similarity(
        pred_sets_standard, similarity_matrix, off_diag=True
    )
    print(f"Standard RAPS - Average pairwise similarity: {overall_sim_standard:.3f}")

if len(results_utility_weighted['unselected_idx']) > 0:
    pred_sets_utility_weighted = results_utility_weighted['prediction_sets']['unselected']
    avg_sim_weighted, overall_sim_weighted = eval_similarity(
        pred_sets_utility_weighted, similarity_matrix, off_diag=True
    )
    print(f"Utility-Aware (weighted) - Average pairwise similarity: {overall_sim_weighted:.3f}")

if len(results_utility_greedy['unselected_idx']) > 0:
    pred_sets_utility_greedy = results_utility_greedy['prediction_sets']['unselected']
    avg_sim_greedy, overall_sim_greedy = eval_similarity(
        pred_sets_utility_greedy, similarity_matrix, off_diag=True
    )
    print(f"Utility-Aware (greedy) - Average pairwise similarity: {overall_sim_greedy:.3f}")

print()
print("Higher similarity scores indicate more coherent prediction sets")
print("(i.e., containing related/similar classes)")
print()

# =============================================================================
# 5. Examine Sample Prediction Sets
# =============================================================================

print("=" * 70)
print("Example Prediction Sets for Unselected Samples")
print("=" * 70)
print()

class_names = ["A.1", "A.2", "B.1", "B.2", "C"]

if len(results_standard['unselected_idx']) > 0:
    # Show first 5 unselected samples
    for i in range(min(5, len(results_standard['unselected_idx']))):
        idx = results_standard['unselected_idx'][i]

        print(f"Sample {idx} (True label: {class_names[test_labels[idx]]})")
        print(f"  Probabilities: {test_probs[idx]}")

        # Standard RAPS
        pred_standard = pred_sets_standard[i]
        classes_standard = [class_names[j] for j in range(n_classes) if pred_standard[j]]
        print(f"  Standard RAPS: {classes_standard}")

        # Utility-aware weighted
        if len(results_utility_weighted['unselected_idx']) > i:
            pred_weighted = pred_sets_utility_weighted[i]
            classes_weighted = [class_names[j] for j in range(n_classes) if pred_weighted[j]]
            print(f"  Utility-Aware (weighted): {classes_weighted}")

        # Utility-aware greedy
        if len(results_utility_greedy['unselected_idx']) > i:
            pred_greedy = pred_sets_utility_greedy[i]
            classes_greedy = [class_names[j] for j in range(n_classes) if pred_greedy[j]]
            print(f"  Utility-Aware (greedy): {classes_greedy}")

        print()

# =============================================================================
# 6. Visualization
# =============================================================================

print("=" * 70)
print("Creating Visualization...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Similarity Matrix Heatmap
ax = axes[0, 0]
im = ax.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_title('Class Similarity Matrix')
plt.colorbar(im, ax=ax)

# Plot 2: Set Size Comparison
ax = axes[0, 1]
methods = ['Standard\nRAPS', 'Utility\n(weighted)', 'Utility\n(greedy)']
avg_sizes = []
if len(results_standard['unselected_idx']) > 0:
    avg_sizes.append(results_standard['set_sizes']['unselected'].mean())
else:
    avg_sizes.append(0)
if len(results_utility_weighted['unselected_idx']) > 0:
    avg_sizes.append(results_utility_weighted['set_sizes']['unselected'].mean())
else:
    avg_sizes.append(0)
if len(results_utility_greedy['unselected_idx']) > 0:
    avg_sizes.append(results_utility_greedy['set_sizes']['unselected'].mean())
else:
    avg_sizes.append(0)

ax.bar(methods, avg_sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('Average Set Size')
ax.set_title('Prediction Set Size Comparison\n(Unselected Samples)')
ax.grid(axis='y', alpha=0.3)

# Plot 3: Coverage Comparison
ax = axes[1, 0]
coverages = []
if len(results_standard['unselected_idx']) > 0:
    coverages.append(results_standard['coverage']['unselected'].mean())
else:
    coverages.append(0)
if len(results_utility_weighted['unselected_idx']) > 0:
    coverages.append(results_utility_weighted['coverage']['unselected'].mean())
else:
    coverages.append(0)
if len(results_utility_greedy['unselected_idx']) > 0:
    coverages.append(results_utility_greedy['coverage']['unselected'].mean())
else:
    coverages.append(0)

ax.bar(methods, coverages, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
ax.set_ylabel('Coverage')
ax.set_title('Coverage Comparison\n(Unselected Samples)')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Coherence Comparison
ax = axes[1, 1]
similarities = []
if len(results_standard['unselected_idx']) > 0:
    similarities.append(overall_sim_standard)
else:
    similarities.append(0)
if len(results_utility_weighted['unselected_idx']) > 0:
    similarities.append(overall_sim_weighted)
else:
    similarities.append(0)
if len(results_utility_greedy['unselected_idx']) > 0:
    similarities.append(overall_sim_greedy)
else:
    similarities.append(0)

ax.bar(methods, similarities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('Average Pairwise Similarity')
ax.set_title('Prediction Set Coherence\n(Higher = More Similar Classes)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('utility_aware_cp_comparison.png', dpi=300, bbox_inches='tight')
print("Saved visualization to 'utility_aware_cp_comparison.png'")
print()

# =============================================================================
# 7. Summary and Recommendations
# =============================================================================

print("=" * 70)
print("Summary and Recommendations")
print("=" * 70)
print()
print("When to use Utility-Aware CP:")
print("  1. Labels have semantic relationships (medical diagnoses, taxonomies)")
print("  2. Prediction sets should be interpretable/coherent")
print("  3. A similarity matrix is available")
print()
print("Method Selection:")
print("  - 'weighted': Balances similarity and probability (recommended)")
print("  - 'greedy': Pure max similarity expansion")
print()
print("Key Benefits:")
print("  - More coherent prediction sets (similar classes grouped)")
print("  - Improved interpretability for domain experts")
print("  - Maintains valid coverage guarantees")
print()
