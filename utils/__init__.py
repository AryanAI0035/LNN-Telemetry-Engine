"""
utils/ — Utility package for the LNN Telemetry Engine.

This package provides two core modules:
    • metrics.py     — Hand-rolled classification metrics (no sklearn dependency).
    • visualization.py — Matplotlib-based plotting with a premium dark theme.

All public helpers are re-exported here for convenient top-level imports:

    from utils import precision, recall, f1_score, accuracy
    from utils import confusion_matrix, optimize_threshold, classification_report
    from utils import (plot_sensor_anomalies, plot_hidden_states,
                       plot_training_curves, plot_confusion_matrix,
                       plot_detection_timeline)

"""

# ── Metrics ──────────────────────────────────────────────────────────────────
from utils.metrics import (
    precision,
    recall,
    f1_score,
    accuracy,
    confusion_matrix,
    optimize_threshold,
    classification_report,
)

# ── Visualization ────────────────────────────────────────────────────────────
from utils.visualization import (
    plot_sensor_anomalies,
    plot_hidden_states,
    plot_training_curves,
    plot_confusion_matrix,
    plot_detection_timeline,
)

__all__ = [
    # Metrics
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "confusion_matrix",
    "optimize_threshold",
    "classification_report",
    # Visualization
    "plot_sensor_anomalies",
    "plot_hidden_states",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_detection_timeline",
]
