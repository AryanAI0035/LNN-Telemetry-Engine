"""
utils/metrics.py — Hand-rolled classification metrics for the LNN Telemetry Engine.

Every function in this module is implemented from scratch using only NumPy.
The goal is to demonstrate a deep understanding of the math behind common
binary-classification metrics rather than simply calling sklearn.


* All inputs are coerced to 1-D NumPy int arrays so callers can pass plain
  Python lists or tensors without friction.
* Division-by-zero is handled gracefully — the affected metric returns 0.0
  instead of raising or returning NaN.
* Type hints and docstrings follow Google-style conventions.

Metrics implemented
-------------------
precision, recall, f1_score, accuracy, confusion_matrix,
optimize_threshold, classification_report

"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

# Type alias — anything that can be treated as a label vector.
ArrayLike = Union[np.ndarray, List[int], Sequence[int]]


# Helper: coerce arbitrary inputs to flat int arrays

def _to_array(x: ArrayLike) -> np.ndarray:
    """Convert *x* to a 1-D NumPy integer array.

    This lets every public function accept plain Python lists, nested
    sequences, or existing ndarrays without duplicating conversion logic.
    """
    return np.asarray(x, dtype=int).ravel()


# Core confusion-matrix counts

def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for binary classification.

    
    * True Positive  (TP): predicted 1, actual 1
    * False Positive (FP): predicted 1, actual 0
    * False Negative (FN): predicted 0, actual 1
    * True Negative  (TN): predicted 0, actual 0

    We compute these with simple boolean masks rather than looping — this is
    both faster and more readable than an explicit for-loop.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


# Precision

def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute precision = TP / (TP + FP).

    Precision answers: "Of everything we *predicted* as positive, how many
    actually were?"  A low precision means lots of false alarms.

    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    tp, fp, _fn, _tn = _counts(y_true, y_pred)

    # Guard against division by zero — if the model never predicted positive,
    # precision is undefined; we return 0.0 by convention.
    denominator = tp + fp
    if denominator == 0:
        return 0.0

    return tp / denominator


# Recall

def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute recall = TP / (TP + FN).

    Recall answers: "Of all *actual* positives, how many did we catch?"
    A low recall means we're missing real anomalies.

    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    tp, _fp, fn, _tn = _counts(y_true, y_pred)

    denominator = tp + fn
    if denominator == 0:
        return 0.0

    return tp / denominator


# F1 Score

def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the F1 score = 2 · (P · R) / (P + R).

    The F1 score is the *harmonic mean* of precision and recall.  Unlike
    the arithmetic mean, the harmonic mean penalizes extreme imbalance —
    you can't get a high F1 by gaming only one of the two.

    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    # If both precision and recall are zero, F1 is 0 (avoid 0/0).
    if p + r == 0:
        return 0.0

    # Harmonic mean formula: 2·P·R / (P + R)
    return 2.0 * p * r / (p + r)


# Accuracy

def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute accuracy = (TP + TN) / N.

    Simply counts how many predictions matched the ground truth.

    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)

    n = len(y_true)
    if n == 0:
        return 0.0

    # Element-wise comparison → count matches → divide by total.
    return float(np.sum(y_true == y_pred)) / n


# Confusion Matrix

def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Build a 2×2 confusion matrix for binary classification.

    Layout (matches sklearn convention)::

        [[TN, FP],
         [FN, TP]]

    Row index = actual class, column index = predicted class.

    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    tp, fp, fn, tn = _counts(y_true, y_pred)

    # Arrange in the standard [[TN, FP], [FN, TP]] layout.
    return np.array([[tn, fp],
                     [fn, tp]], dtype=int)


# Threshold Optimization

def optimize_threshold(
    y_true: ArrayLike,
    y_scores: Union[np.ndarray, List[float]],
    metric: str = "f1",
    thresholds: Optional[Union[np.ndarray, List[float]]] = None,
) -> tuple[float, float]:
    """Sweep thresholds to find the one that maximizes a chosen metric.

    Many models output continuous scores (e.g., sigmoid probabilities).  To
    convert these to binary predictions we pick a threshold *t* — scores ≥ t
    become 1, the rest become 0.  This function tries many thresholds and
    returns the best one.

    """
    # ── Map metric name → callable ────────────────────────────────────────
    metric_fn_map = {
        "f1": f1_score,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }

    if metric not in metric_fn_map:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose from {list(metric_fn_map.keys())}."
        )

    metric_fn = metric_fn_map[metric]

    # ── Coerce inputs ─────────────────────────────────────────────────────
    y_true = _to_array(y_true)
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    # ── Default threshold grid: 0.10, 0.15, …, 0.90 ──────────────────────
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    else:
        thresholds = np.asarray(thresholds, dtype=float).ravel()

    best_threshold = 0.5       # sensible default
    best_score = -1.0          # any real score will beat this

    for t in thresholds:
        # Binarize: scores >= threshold → 1, else → 0
        y_pred = (y_scores >= t).astype(int)
        score = metric_fn(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold, best_score


# Classification Report

def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: Optional[List[str]] = None,
    print_report: bool = True,
) -> str:
    """Generate a formatted classification report similar to sklearn's.

    Produces per-class precision / recall / F1 / support as well as overall
    accuracy.

    """
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)

    if class_names is None:
        class_names = ["Normal", "Anomaly"]

    tp, fp, fn, tn = _counts(y_true, y_pred)
    n = len(y_true)

    # ── Per-class metrics ─────────────────────────────────────────────────
    # Class 0 ("Normal"): treat class-0 as the "positive" class locally.
    #   precision_0 = TN / (TN + FN)   — of predicted-0, how many were 0?
    #   recall_0    = TN / (TN + FP)   — of actual-0, how many did we get?
    support_0 = tn + fp  # number of actual class-0 samples
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

    # Class 1 ("Anomaly"): the standard TP / FP / FN definitions.
    support_1 = tp + fn  # number of actual class-1 samples
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

    # ── Overall metrics ───────────────────────────────────────────────────
    acc = (tp + tn) / n if n > 0 else 0.0
    macro_prec = (prec_0 + prec_1) / 2
    macro_rec = (rec_0 + rec_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2
    weighted_prec = (prec_0 * support_0 + prec_1 * support_1) / n if n > 0 else 0.0
    weighted_rec = (rec_0 * support_0 + rec_1 * support_1) / n if n > 0 else 0.0
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / n if n > 0 else 0.0

    # ── Build the table ───────────────────────────────────────────────────
    header = f"{'':>15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}"
    sep = "─" * len(header)

    rows = [
        sep,
        header,
        sep,
        f"{class_names[0]:>15s} {prec_0:10.4f} {rec_0:10.4f} {f1_0:10.4f} {support_0:10d}",
        f"{class_names[1]:>15s} {prec_1:10.4f} {rec_1:10.4f} {f1_1:10.4f} {support_1:10d}",
        sep,
        f"{'accuracy':>15s} {'':>10s} {'':>10s} {acc:10.4f} {n:10d}",
        f"{'macro avg':>15s} {macro_prec:10.4f} {macro_rec:10.4f} {macro_f1:10.4f} {n:10d}",
        f"{'weighted avg':>15s} {weighted_prec:10.4f} {weighted_rec:10.4f} {weighted_f1:10.4f} {n:10d}",
        sep,
    ]

    report = "\n".join(rows)

    if print_report:
        print(report)

    return report
