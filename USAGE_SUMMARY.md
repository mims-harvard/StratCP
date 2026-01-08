# StratCP Package - Complete Usage Summary

## 🎯 Quick Reference

### Simplest Usage

```python
from stratcp import StratifiedCP

# One-liner: fit and predict
scp = StratifiedCP(alpha_sel=0.1, alpha_cp=0.1)
results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# Access results:
print(f"Selected: {len(results['selected_idx'])} samples")
print(f"Unselected: {len(results['unselected_idx'])} samples")
print(scp.summary())
``` 

The `results` object contains:
- `selected_idx`, indices of high-confidence samples by the FDR-controlled action arm.
- `unselected_idx`, indices of low-confidence samples in the deferral arm.
- `threshold`, the selection cutoff on the confidence score (max probability).
- `prediction_sets`, boolean array of shape `(n_unselected, n_class)` for unselected samples only.
- `set_sizes`, sizes of each prediction set in `prediction_sets`.



You can also perform per-class selection where we:
- FDR-controlled selection of high-confidence predictions for every class of argmax prediction, whose argmax equals unknown true label
- Conformal prediction sets for unselected, low-confident cases with 90% coverage

by simply adding the argument `eligibility = 'per_class'`.

```python
import stratcp as scp

# One-line usage: fit and predict
model = scp.StratifiedCP(alpha_sel=0.1, alpha_cp=0.1, eligibility='per_class')
results = model.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# Print detailed summary
print(model.summary())
```

The `results` object contains:
- `all_selected`, a list that contains confident indices selected for each class, followed by indices unselected by any class (low-confidence).
- `thresholds`, a list of per-class selection thresholds.
- `prediction_sets`, boolean array of shape `(n_unselected, n_class)` for unselected samples only (those in `all_selected[K]`).
- `set_sizes`, sizes of each prediction set in `prediction_sets`.

### Common Workflows

#### 1. Basic Classification with Prediction Sets

```python
from stratcp import StratifiedCP

# Fit on calibration data
scp = StratifiedCP(score_fn='aps', alpha_sel=0.1, alpha_cp=0.1)
scp.fit(cal_probs, cal_labels)

# Predict on new test data
results = scp.predict(test_probs, test_labels)

# Get prediction sets for each sample
for i, idx in enumerate(results['unselected_idx']):
    pred_set = results['prediction_sets'][i, :]
    classes_in_set = np.where(pred_set)[0]
    print(f"Sample {idx}: Prediction set = {classes_in_set}")
``` 

#### 2. Multi-class Classification with Per-Class Action Arms

```python
from stratcp import StratifiedCP

# Fit on calibration data
scp = StratifiedCP(score_fn='aps', alpha_sel=0.1, alpha_cp=0.1, eligibility='per_class') 
results = model.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# Predict on new test data
results = scp.predict(test_probs, test_labels)

# Get prediction sets for each sample
for i, idx in enumerate(results['all_selected'][-1]):
    pred_set = results['prediction_sets'][i,:]
    classes_in_set = np.where(pred_set)[0]
    print(f"Sample {idx}: Prediction set = {classes_in_set}")
``` 


#### 3. Utility-Aware CP with Similarity Matrix

```python
from stratcp import StratifiedCP
import numpy as np

# Define similarity matrix between classes
# Higher values = more similar (range [0, 1])
# Example: 5 disease classes with hierarchical relationships
similarity_matrix = np.array([
    [1.0, 0.9, 0.3, 0.3, 0.1],  # Class 0: very similar to 1
    [0.9, 1.0, 0.3, 0.3, 0.1],  # Class 1: very similar to 0
    [0.3, 0.3, 1.0, 0.9, 0.1],  # Class 2: very similar to 3
    [0.3, 0.3, 0.9, 1.0, 0.1],  # Class 3: very similar to 2
    [0.1, 0.1, 0.1, 0.1, 1.0],  # Class 4: dissimilar to all
])

# Use utility-aware CP for more coherent prediction sets
scp = StratifiedCP(
    score_fn='utility',
    similarity_matrix=similarity_matrix,
    utility_method='weighted',  # or 'greedy'
    alpha_sel=0.1,
    alpha_cp=0.1
)
results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# Evaluate prediction set coherence
from stratcp.conformal import eval_similarity
avg_sim, overall_sim = eval_similarity(
    results['prediction_sets']['unselected'],
    similarity_matrix
)
print(f"Average pairwise similarity: {overall_sim:.3f}")
```

## 📚 Parameters Guide

### StratifiedCP Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_fn` | str | `'raps'` | Nonconformity score: `'tps'`, `'aps'`, `'raps'`, or `'utility'` |
| `alpha_sel` | float | `0.1` | FDR level for selection (0.1 = 10% FDR) |
| `alpha_cp` | float | `0.1` | Miscoverage for CP (0.1 = 90% coverage) |
| `nonempty` | bool | `True` | Force non-empty prediction sets |
| `rand` | bool | `True` | Randomize p-values for exact coverage |
| `lam_reg` | float | `0.01` | RAPS regularization parameter |
| `similarity_matrix` | ndarray | `None` | Similarity matrix (n_classes, n_classes) for utility-aware CP |
| `utility_method` | str | `'weighted'` | Expansion method: `'weighted'` or `'greedy'` (for `score_fn='utility'`) |

### Input Data Format

**Calibration data:**
- `cal_probs`: `(n_cal, n_classes)` - Predicted class probabilities
- `cal_labels`: `(n_cal,)` - True class labels (0 to n_classes-1)

**Test data:**
- `test_probs`: `(n_test, n_classes)` - Predicted class probabilities
- `test_labels`: `(n_test,)` - Optional, true class labels for evaluation

## 🧬 Utility-Aware Conformal Prediction

### What is Utility-Aware CP?

Traditional conformal prediction produces prediction sets by ordering classes by predicted probability. Utility-aware CP leverages **similarity relationships between labels** to produce more coherent and interpretable prediction sets.

**Use Cases:**
- Medical diagnosis (diseases with hierarchical relationships)
- Biological classification (taxonomies, phylogenetic trees)
- Any domain with semantic label relationships

### Key Benefits

1. **More coherent prediction sets** - Similar/related classes are grouped together
2. **Better interpretability** - Easier for domain experts to understand
3. **Valid coverage** - Maintains all conformal prediction guarantees

### Quick Start

```python
from stratcp import StratifiedCP
import numpy as np

# 1. Define similarity matrix (n_classes, n_classes)
# Higher values = more similar (range [0, 1])
similarity_matrix = np.array([
    [1.0, 0.9, 0.2],
    [0.9, 1.0, 0.2],
    [0.2, 0.2, 1.0]
])

# 2. Use utility-aware CP
scp = StratifiedCP(
    score_fn='utility',
    similarity_matrix=similarity_matrix,
    utility_method='weighted',  # 'weighted' or 'greedy'
)
results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)
```

### Expansion Methods

#### Weighted (Recommended)

Balances similarity and prediction probability. At each expansion step, selects the candidate that maximizes:
```
score = mean(similarity[candidate, existing] * prob[candidate])
```

**When to use:** Most cases - produces coherent sets while respecting model confidence.

#### Greedy

Pure max similarity expansion. At each step, selects the candidate most similar to any existing class among top-K candidates.

**When to use:** When similarity structure is more important than predicted probabilities.

### Creating Similarity Matrices

#### From Domain Knowledge

```python
# Example: Medical diagnoses
# 0: Type 1 Diabetes, 1: Type 2 Diabetes, 2: Hypertension
similarity_matrix = np.array([
    [1.0, 0.8, 0.2],  # Type 1 similar to Type 2
    [0.8, 1.0, 0.2],  # Type 2 similar to Type 1
    [0.2, 0.2, 1.0]   # Hypertension different
])
```

#### From Embeddings

```python
from sklearn.metrics.pairwise import cosine_similarity

# Assume you have class embeddings (n_classes, embedding_dim)
class_embeddings = ...  # From BERT, word2vec, etc.

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(class_embeddings)

# Normalize to [0, 1] if needed
similarity_matrix = (similarity_matrix + 1) / 2
```

#### From Ontologies

```python
# Example: Disease ontology with ancestor relationships
def ontology_similarity(class_i, class_j, ontology):
    """Compute similarity based on common ancestors."""
    ancestors_i = set(ontology.get_ancestors(class_i))
    ancestors_j = set(ontology.get_ancestors(class_j))

    if class_i == class_j:
        return 1.0

    # Jaccard similarity of ancestors
    intersection = ancestors_i & ancestors_j
    union = ancestors_i | ancestors_j
    return len(intersection) / len(union) if union else 0.0
```

### Evaluating Prediction Set Coherence

```python
from stratcp.conformal import eval_similarity

# Evaluate average pairwise similarity in prediction sets
avg_sim, overall_sim = eval_similarity(
    results['prediction_sets']['unselected'],
    similarity_matrix,
    off_diag=True  # Exclude self-similarity
)

print(f"Average similarity: {overall_sim:.3f}")
print(f"Per-sample similarity: {avg_sim}")
```

**Interpretation:**
- Higher values = more coherent sets (similar classes grouped)
- Lower values = less coherent (dissimilar classes mixed)

### Complete Example

See `examples/utility_aware_cp.py` for a comprehensive example comparing standard vs utility-aware CP with visualizations.

## 🔬 Advanced: Lower-Level API

For more control, use the modular functions:

```python
from stratcp.selection import get_sel_single, get_reference_sel_single
from stratcp.conformal import compute_score_raps, conformal

# Manual workflow
# 1. Selection
sel_idx, unsel_idx, tau = get_sel_single(...)

# 2. Compute scores
cal_scores, test_scores = compute_score_raps(...)

# 3. CP for selected
pred_sets_sel, cov_sel, sizes_sel = conformal(...)

# 4. JOMI for unselected
ref_mats = get_reference_sel_single(...)
pred_sets_unsel, cov_unsel, sizes_unsel = conformal(..., if_in_ref=ref_mats)
```

## 📊 Interpreting Results

### Selection Interpretation

- **Selected samples**: High-confidence predictions where FDR ≤ α_sel
  - These typically have **smaller prediction sets** (often singleton)
  - Safe for automated decision-making

- **Unselected samples**: Lower-confidence predictions
  - These typically have **larger prediction sets**
  - Should be flagged for human review or additional testing

### Coverage Interpretation

The coverage guarantee is:
- P(true_label ∈ prediction_set) ≥ 1 - α_cp

For α_cp = 0.1, you get **at least 90% coverage** on average.

### Set Size Interpretation

- **Size 1 (singleton)**: One class predicted with high confidence
- **Size 2-3**: Small differential diagnosis
- **Size > 3**: High uncertainty, many possible classes

## 🏥 Example: Medical Diagnosis

```python
from stratcp import StratifiedCP

# Model predictions on patient data
# cal_probs: (1000, 5) for 5 disease classes
# test_probs: (200, 5) for new patients

scp = StratifiedCP(alpha_sel=0.1, alpha_cp=0.1)
results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# High-confidence patients (automated diagnosis)
for i, idx in enumerate(results['selected_idx']):
    pred_set = results['prediction_sets']['selected'][i]
    if np.sum(pred_set) == 1:  # Singleton
        diagnosis = np.where(pred_set)[0][0]
        print(f"Patient {idx}: Confident diagnosis = Class {diagnosis}")

# Low-confidence patients (manual review needed)
for i, idx in enumerate(results['unselected_idx']):
    pred_set = results['prediction_sets']['unselected'][i]
    possible_diagnoses = np.where(pred_set)[0]
    print(f"Patient {idx}: Differential diagnosis = {possible_diagnoses}")
    print(f"  -> Flag for specialist review")
```

## 📖 Complete Example Scripts

- `examples/simple_usage.py` - Basic usage examples
- `examples/utility_aware_cp.py` - Utility-aware CP with similarity matrices


## 📝 Citation

If you use StratCP in your research, please cite our paper (coming soon).
