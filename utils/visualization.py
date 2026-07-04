"""
utils/visualization.py —  for the LNN Telemetry Engine.


Every plot produced by this module should look *publication-ready* out of the
box: dark backgrounds, carefully chosen accent colours, generous whitespace,
and high-DPI output (300 dpi PNGs).

Colour palette
--------------
We use a curated set of accent colours that pop against a near-black canvas:

    TEAL      #00D9C0   — primary sensor traces / normal data
    ORANGE    #FF6F3C   — anomaly highlights / warnings
    PURPLE    #B37FEB   — secondary traces / model predictions
    MAGENTA   #FF4DA6   — false negatives / critical misses
    GOLD      #F5C542   — F1 / validation curves
    SLATE     #5E6B7A   — muted grid / annotations

Every function follows the same pattern:
1. Create figure + axes with the dark theme applied.
2. Draw the data.
3. Save to *save_path* (defaulting to ``outputs/figures/``).
4. Close the figure to free memory.

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Global configuration

# ── Colour palette ────────────────────────────────────────────────────────────
TEAL = "#00D9C0"
ORANGE = "#FF6F3C"
PURPLE = "#B37FEB"
MAGENTA = "#FF4DA6"
GOLD = "#F5C542"
SLATE = "#5E6B7A"
SOFT_RED = "#E8524A"
CYAN = "#56D6F5"

# A few extra colours for multi-line plots (hidden-state dimensions).
HIDDEN_STATE_PALETTE = [TEAL, ORANGE, PURPLE, GOLD, MAGENTA, CYAN]

# ── Default output directory ──────────────────────────────────────────────────
DEFAULT_FIGURE_DIR = os.path.join("outputs", "figures")

# ── DPI for saved PNGs ───────────────────────────────────────────────────────
SAVE_DPI = 300

# Type alias for convenience.
ArrayLike = Union[np.ndarray, List[float], Sequence[float]]


# Shared helpers

def _ensure_dir(path: str) -> None:
    """Create the parent directory for *path* if it doesn't already exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _resolve_save_path(save_path: Optional[str], default_name: str) -> str:
    """Return a fully-resolved save path, falling back to the default dir."""
    if save_path is None:
        save_path = os.path.join(DEFAULT_FIGURE_DIR, default_name)
    _ensure_dir(save_path)
    return save_path


def _apply_dark_theme() -> None:
    """Apply a consistent dark theme to all matplotlib figures.

    We override individual rcParams instead of using plt.style.use() so
    that our theme is deterministic regardless of the user's matplotlibrc.
    """
    dark_params = {
        # Canvas & axes
        "figure.facecolor": "#0D1117",
        "axes.facecolor": "#161B22",
        "axes.edgecolor": SLATE,
        "axes.labelcolor": "#C9D1D9",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        # Ticks
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Grid
        "axes.grid": True,
        "grid.color": "#21262D",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        # Legend
        "legend.facecolor": "#161B22",
        "legend.edgecolor": SLATE,
        "legend.fontsize": 9,
        "legend.labelcolor": "#C9D1D9",
        # Text
        "text.color": "#C9D1D9",
        # Savefig
        "savefig.facecolor": "#0D1117",
        "savefig.edgecolor": "#0D1117",
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
    }
    plt.rcParams.update(dark_params)


def _save_and_close(fig: plt.Figure, save_path: str) -> None:
    """Save a figure as a high-DPI PNG and close it to free memory."""
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Figure saved → {save_path}")


def _highlight_anomalies(
    ax: plt.Axes,
    labels: np.ndarray,
    color: str = ORANGE,
    alpha: float = 0.15,
) -> None:
    """Shade contiguous anomaly regions (label == 1) on *ax*.

    We find the start/end indices of every contiguous run of 1s and draw
    a vertical span (axvspan) for each.
    """
    # Pad with 0s so np.diff catches edges at the boundaries.
    padded = np.concatenate([[0], labels, [0]])
    diff = np.diff(padded)

    # Rising edge → anomaly starts; falling edge → anomaly ends.
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color=color, alpha=alpha, zorder=0)


# 1. Sensor Anomaly Plot

def plot_sensor_anomalies(
    sensor_data: np.ndarray,
    anomaly_labels: ArrayLike,
    sensor_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> str:
    """Multi-panel plot showing each sensor channel with anomaly highlights.

    """
    _apply_dark_theme()

    sensor_data = np.asarray(sensor_data)
    anomaly_labels = np.asarray(anomaly_labels, dtype=int).ravel()

    # Infer number of channels.
    if sensor_data.ndim == 1:
        sensor_data = sensor_data.reshape(-1, 1)
    n_channels = sensor_data.shape[1]

    if sensor_names is None:
        sensor_names = [f"Sensor {i}" for i in range(n_channels)]

    save_path = _resolve_save_path(save_path, "sensor_anomalies.png")

    # ── Create subplots — one row per channel ─────────────────────────────
    fig, axes = plt.subplots(
        n_channels, 1,
        figsize=(16, 2.8 * n_channels),
        sharex=True,
    )

    # Handle the case where there's only one sensor (axes is not a list).
    if n_channels == 1:
        axes = [axes]

    time_idx = np.arange(sensor_data.shape[0])

    for i, ax in enumerate(axes):
        # Plot the sensor trace in teal.
        ax.plot(time_idx, sensor_data[:, i], color=TEAL, linewidth=0.8, alpha=0.9)

        # Highlight anomaly regions in orange.
        _highlight_anomalies(ax, anomaly_labels, color=ORANGE, alpha=0.20)

        # Labels.
        ax.set_ylabel(sensor_names[i], fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=8)

    # Bottom axis gets the shared x-label.
    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle(
        "Sensor Channels with Anomaly Regions",
        fontsize=16, fontweight="bold", color="#E6EDF3", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    _save_and_close(fig, save_path)
    return save_path


# 2. Hidden State Evolution (the "money shot")

def plot_hidden_states(
    hidden_states: np.ndarray,
    anomaly_labels: ArrayLike,
    save_path: Optional[str] = None,
    n_dims: int = 6,
) -> str:
    """Plot the evolution of liquid hidden-state dimensions over time.

     — it shows how the liquid neural network's
    internal state responds to anomalous conditions in the telemetry stream.

    """
    _apply_dark_theme()

    hidden_states = np.asarray(hidden_states)
    anomaly_labels = np.asarray(anomaly_labels, dtype=int).ravel()
    save_path = _resolve_save_path(save_path, "hidden_states.png")

    # Only plot the first `n_dims` dimensions (or fewer if D is small).
    n_dims = min(n_dims, hidden_states.shape[1])
    time_idx = np.arange(hidden_states.shape[0])

    fig, ax = plt.subplots(figsize=(16, 6))

    # ── Draw each hidden dimension with a unique colour ───────────────────
    for d in range(n_dims):
        colour = HIDDEN_STATE_PALETTE[d % len(HIDDEN_STATE_PALETTE)]
        ax.plot(
            time_idx,
            hidden_states[:, d],
            color=colour,
            linewidth=1.0,
            alpha=0.85,
            label=f"h[{d}]",
        )

    # ── Anomaly highlights ────────────────────────────────────────────────
    _highlight_anomalies(ax, anomaly_labels, color=SOFT_RED, alpha=0.18)

    # ── Aesthetics ────────────────────────────────────────────────────────
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Hidden State Value", fontsize=12)
    ax.set_title(
        "Liquid Neural Network — Hidden State Evolution",
        fontsize=16, fontweight="bold", color="#E6EDF3",
    )
    ax.legend(
        loc="upper right",
        ncol=n_dims,
        framealpha=0.6,
        fontsize=9,
    )

    fig.tight_layout()
    _save_and_close(fig, save_path)
    return save_path


# 3. Training Curves

def plot_training_curves(
    train_losses: ArrayLike,
    val_losses: ArrayLike,
    train_f1s: ArrayLike,
    val_f1s: ArrayLike,
    save_path: Optional[str] = None,
) -> str:
    """Side-by-side loss and F1 curves for training & validation.

    """
    _apply_dark_theme()
    save_path = _resolve_save_path(save_path, "training_curves.png")

    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(16, 5.5))

    # ── Left panel: Loss curves ───────────────────────────────────────────
    ax_loss.plot(epochs, train_losses, color=TEAL, linewidth=1.8, label="Train Loss")
    ax_loss.plot(
        epochs, val_losses,
        color=ORANGE, linewidth=1.8, linestyle="--", label="Val Loss",
    )
    # Small circle markers on each epoch for readability.
    ax_loss.scatter(epochs, train_losses, color=TEAL, s=18, zorder=5)
    ax_loss.scatter(epochs, val_losses, color=ORANGE, s=18, zorder=5)

    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax_loss.legend(loc="upper right", fontsize=10)

    # ── Right panel: F1 curves ────────────────────────────────────────────
    ax_f1.plot(epochs, train_f1s, color=PURPLE, linewidth=1.8, label="Train F1")
    ax_f1.plot(
        epochs, val_f1s,
        color=GOLD, linewidth=1.8, linestyle="--", label="Val F1",
    )
    ax_f1.scatter(epochs, train_f1s, color=PURPLE, s=18, zorder=5)
    ax_f1.scatter(epochs, val_f1s, color=GOLD, s=18, zorder=5)

    ax_f1.set_xlabel("Epoch", fontsize=12)
    ax_f1.set_ylabel("F1 Score", fontsize=12)
    ax_f1.set_title("Training & Validation F1", fontsize=14, fontweight="bold")
    ax_f1.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "LNN Training Progress",
        fontsize=16, fontweight="bold", color="#E6EDF3", y=1.01,
    )
    fig.tight_layout()
    _save_and_close(fig, save_path)
    return save_path


# 4. Confusion Matrix Heatmap

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> str:
    """Annotated heatmap of a 2×2 confusion matrix.

    """
    _apply_dark_theme()
    save_path = _resolve_save_path(save_path, "confusion_matrix.png")

    if class_names is None:
        class_names = ["Normal", "Anomaly"]

    cm = np.asarray(cm)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # ── Custom colormap: dark teal → bright teal ──────────────────────────
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dark_teal", ["#0D1117", "#0A4D44", TEAL], N=256,
    )

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="equal")

    # ── Annotate each cell with its count ─────────────────────────────────
    threshold = cm.max() / 2.0  # text colour flips at halfway point
    for i in range(2):
        for j in range(2):
            text_colour = "#0D1117" if cm[i, j] > threshold else "#E6EDF3"
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                fontsize=22, fontweight="bold",
                color=text_colour,
            )

    # ── Axis labels ───────────────────────────────────────────────────────
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold", pad=12)

    # Subtle colorbar on the right.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    _save_and_close(fig, save_path)
    return save_path


# 5. Detection Timeline

def plot_detection_timeline(
    sensor_data: ArrayLike,
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
    save_path: Optional[str] = None,
    sensor_name: str = "Sensor 0",
) -> str:
    """Overlay true vs predicted anomalies on a single sensor trace.

    Colour code
    -----------
    * **True Positive (TP)**: Correctly detected anomaly — shaded *teal*.
    * **False Positive (FP)**: False alarm — shaded *orange*.
    * **False Negative (FN)**: Missed anomaly — shaded *magenta*.

    """
    _apply_dark_theme()
    save_path = _resolve_save_path(save_path, "detection_timeline.png")

    sensor_data = np.asarray(sensor_data, dtype=float).ravel()
    true_labels = np.asarray(true_labels, dtype=int).ravel()
    pred_labels = np.asarray(pred_labels, dtype=int).ravel()
    time_idx = np.arange(len(sensor_data))

    fig, ax = plt.subplots(figsize=(18, 5))

    # ── Background sensor trace ───────────────────────────────────────────
    ax.plot(
        time_idx, sensor_data,
        color=SLATE, linewidth=0.7, alpha=0.6, label=sensor_name,
    )

    # ── Build per-timestep outcome labels ─────────────────────────────────
    # TP = true==1 AND pred==1  →  teal
    # FP = true==0 AND pred==1  →  orange
    # FN = true==1 AND pred==0  →  magenta
    tp_mask = (true_labels == 1) & (pred_labels == 1)
    fp_mask = (true_labels == 0) & (pred_labels == 1)
    fn_mask = (true_labels == 1) & (pred_labels == 0)

    # Highlight contiguous regions for each outcome category.
    _highlight_anomalies(ax, tp_mask.astype(int), color=TEAL, alpha=0.30)
    _highlight_anomalies(ax, fp_mask.astype(int), color=ORANGE, alpha=0.30)
    _highlight_anomalies(ax, fn_mask.astype(int), color=MAGENTA, alpha=0.30)

    # ── Legend patches (since axvspan doesn't auto-legend nicely) ─────────
    from matplotlib.patches import Patch

    legend_elements = [
        plt.Line2D([0], [0], color=SLATE, linewidth=1, label=sensor_name),
        Patch(facecolor=TEAL, alpha=0.4, label="True Positive"),
        Patch(facecolor=ORANGE, alpha=0.4, label="False Positive"),
        Patch(facecolor=MAGENTA, alpha=0.4, label="False Negative"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=10,
        framealpha=0.7,
    )

    # ── Axis decoration ──────────────────────────────────────────────────
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        "Anomaly Detection Timeline — True vs Predicted",
        fontsize=15, fontweight="bold", color="#E6EDF3",
    )

    fig.tight_layout()
    _save_and_close(fig, save_path)
    return save_path
