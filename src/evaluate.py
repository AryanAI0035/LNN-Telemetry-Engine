import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.lnn_core import LNNAnomalyDetector
from data.data_loader import get_dataloaders
from utils.metrics import (
    precision,
    recall,
    f1_score,
    accuracy,
    confusion_matrix,
    classification_report,
    optimize_threshold,
)
from utils.visualization import (
    plot_sensor_anomalies,
    plot_hidden_states,
    plot_confusion_matrix,
    plot_detection_timeline,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# ==============================================================================
# FUNCTION THEORY: Inference Extraction
# ------------------------------------------------------------------------------
# Here we take the best checkpoint of our model and pass the entire test dataset 
# through it. We aren't training anymore! We are just recording its predictions 
# (the "scores") and also tracking its "hidden states". 
# The hidden states are the internal brainwaves of the LNN! We save them so 
# we can visualize exactly how the network reacts when an anomaly happens.
# ==============================================================================
@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    model.eval()
    all_scores = []
    all_labels = []
    all_hidden = []
    all_inputs = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)

        # Get both predictions AND hidden states
        outputs, hidden_states = model(batch_x, return_hidden=True)

        all_scores.append(outputs.cpu().numpy())
        all_labels.append(batch_y.numpy())
        all_hidden.append(hidden_states.cpu().numpy())
        all_inputs.append(batch_x.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten().astype(int)
    all_hidden = np.concatenate(all_hidden, axis=0)  
    all_inputs = np.concatenate(all_inputs, axis=0)   

    return all_scores, all_labels, all_hidden, all_inputs


# ==============================================================================
# FUNCTION THEORY: The Evaluation Pipeline
# ------------------------------------------------------------------------------
# We calculate multiple metrics here:
# - Precision: If the model says "Anomaly!", how often is it actually right?
# - Recall: Out of all the real anomalies, how many did the model catch?
# - F1-Score: The harmonic mean of Precision and Recall. This is the BEST 
#             metric for imbalanced data!
#
# We also "optimize the threshold". Our network outputs a probability between 
# 0 and 1. Usually, people use 0.5 as the cutoff. But sometimes, 0.45 or 0.6 
# works better depending on the noise in the data. We test multiple thresholds 
# to find the absolute best F1-Score.
# ==============================================================================
def evaluate(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  LNN Telemetry Engine — Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    data_cfg = config["data"]
    csv_path = os.path.join(data_cfg["raw_dir"], "telemetry_anomalous.csv")
    _, _, test_loader, data_meta = get_dataloaders(
        csv_path=csv_path,
        processed_dir=data_cfg["processed_dir"],
        seq_len=data_cfg["sequence_length"],
        stride=data_cfg["stride"],
        batch_size=config["training"]["batch_size"],
        train_ratio=data_cfg["train_split"],
        val_ratio=data_cfg["val_split"],
    )
    print(f"  Test samples: {len(test_loader.dataset)}")

    eval_cfg = config["evaluation"]
    checkpoint_path = eval_cfg["checkpoint_path"]

    if not os.path.exists(checkpoint_path):
        print(f"\n  ERROR: No checkpoint found at {checkpoint_path}")
        print(f"  Run 'python src/train.py' first to train the model.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = config["model"]
    model = LNNAnomalyDetector(
        input_size=model_cfg["input_size"],
        units=model_cfg["units"],
        output_size=model_cfg["output_size"],
        use_sparse_wiring=model_cfg["use_sparse_wiring"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} (val F1: {checkpoint['val_f1']:.4f})")
    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"{'='*60}\n")

    print("Running inference on test set...")
    scores, labels, hidden_states, inputs = run_inference(model, test_loader, device)

    # --------------------------------------------------------------------------
    # THEORY: Threshold Optimization
    # Testing 0.1, 0.2 ... 0.9 to see which cutoff produces the highest F1 Score.
    # --------------------------------------------------------------------------
    if eval_cfg.get("optimize_threshold", True):
        best_threshold, best_f1 = optimize_threshold(labels, scores, metric="f1")
        print(f"  Optimized threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    else:
        best_threshold = eval_cfg.get("anomaly_threshold", 0.5)
        print(f"  Using fixed threshold: {best_threshold:.2f}")

    predictions = (scores >= best_threshold).astype(int)

    print(f"\n  Results @ threshold = {best_threshold:.2f}:")
    print(f"  {'─'*40}")
    print(f"  Accuracy:  {accuracy(labels, predictions):.4f}")
    print(f"  Precision: {precision(labels, predictions):.4f}")
    print(f"  Recall:    {recall(labels, predictions):.4f}")
    print(f"  F1 Score:  {f1_score(labels, predictions):.4f}")
    print()
    classification_report(labels, predictions)

    viz_cfg = config.get("visualization", {})
    figures_dir = viz_cfg.get("output_dir", "outputs/figures")
    os.makedirs(figures_dir, exist_ok=True)

    sensor_names = [
        "Battery Voltage", "Solar Current", "CPU Temp",
        "Panel Temp", "Gyro X", "Gyro Y", "Gyro Z", "Magnetometer"
    ]

    # 1. Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, save_path=os.path.join(figures_dir, "confusion_matrix.png"))
    print(f"  Saved: {figures_dir}/confusion_matrix.png")

    # 2. Detection Timeline
    n_timeline = min(200, len(scores))
    sample_input = inputs[:n_timeline]
    sensor_summary = sample_input[:, -1, :] 

    plot_detection_timeline(
        sensor_data=sensor_summary[:, 0],  
        true_labels=labels[:n_timeline],
        pred_labels=predictions[:n_timeline],
        save_path=os.path.join(figures_dir, "detection_timeline.png"),
        sensor_name="Battery Voltage",
    )
    print(f"  Saved: {figures_dir}/detection_timeline.png")

    # 3. Sensor Anomaly Overlay
    plot_sensor_anomalies(
        sensor_data=sensor_summary,
        anomaly_labels=labels[:n_timeline],
        sensor_names=sensor_names,
        save_path=os.path.join(figures_dir, "sensor_anomalies.png"),
    )
    print(f"  Saved: {figures_dir}/sensor_anomalies.png")

    # 4. Hidden State Evolution Plot (The best part!)
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) > 0:
        sample_idx = anomaly_indices[0]
        start_idx = max(0, sample_idx - 10)
        end_idx = min(len(hidden_states), sample_idx + 40)

        hidden_slice = hidden_states[start_idx:end_idx] 
        label_slice = labels[start_idx:end_idx]
        hidden_timeline = hidden_slice[:, -1, :]  

        plot_hidden_states(
            hidden_states=hidden_timeline,
            anomaly_labels=label_slice,
            save_path=os.path.join(figures_dir, "hidden_state_evolution.png"),
        )
        print(f"  Saved: {figures_dir}/hidden_state_evolution.png")
    else:
        print("  Skipped hidden state plot (no anomalies found in test set)")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete! Figures saved to {figures_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained LNN")
    parser.add_argument("--config", type=str, default="configs/hyperparams.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate(config)


if __name__ == "__main__":
    main()
