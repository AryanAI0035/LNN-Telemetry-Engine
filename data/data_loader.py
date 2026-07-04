#!/usr/bin/env python3
# ==============================================================================
# THEORY: Time-Series Data Loading & Windowing
# ------------------------------------------------------------------------------
# Neural networks (especially recurrent ones) need data in a specific format.
# When working with time-series data like telemetry, we can't just feed in one 
# data point at a time and expect the network to know if it's anomalous. It 
# needs *context*.
#
# We solve this using a "Sliding Window". We take a chunk of time (e.g., 50 
# seconds), look at all the sensor data in that chunk, and ask the model: 
# "Is there an anomaly anywhere in this window?" Then we slide the window 
# forward by 1 second and ask again.
#
# We also have to be very careful about Data Leakage. We split our data 
# chronologically (first 70% for training, next 15% for validation, last 15% 
# for testing). We MUST NOT shuffle before splitting, otherwise the model 
# will learn from the "future" to predict the "past"!
# ==============================================================================

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

FEATURE_COLUMNS = [
    "battery_voltage", "solar_current", "cpu_temp", "panel_temp",
    "gyro_x", "gyro_y", "gyro_z", "magnetometer",
]
LABEL_COLUMN = "anomaly"
PROCESSED_DIR = "data/processed"


# ==============================================================================
# CLASS THEORY: PyTorch Dataset
# ------------------------------------------------------------------------------
# PyTorch requires us to wrap our data in a `Dataset` class so it knows how 
# to grab a single item (a window of features and its label) during training.
# ==============================================================================
class TelemetryDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        super().__init__()
        assert len(features) == len(labels)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ==============================================================================
# FUNCTION THEORY: Normalisation
# ------------------------------------------------------------------------------
# Neural networks hate large numbers. If CPU temperature is 60 and Solar Current 
# is 0.2, the network will think Temp is way more important just because it's 
# bigger. We use Min-Max Normalisation to squash everything between 0 and 1.
# CRITICAL: We only calculate min and max from the TRAINING set!
# ==============================================================================
def _compute_norm_params(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)
    return mins, maxs

def _apply_normalisation(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalised = (data - mins) / ranges
    normalised = np.nan_to_num(normalised, nan=0.0)
    return normalised

def save_norm_params(mins: np.ndarray, maxs: np.ndarray, output_dir: str = PROCESSED_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "norm_params.npz")
    np.savez(path, mins=mins, maxs=maxs)
    print(f"  [data_loader] Normalisation params saved → {path}")
    return path

def load_norm_params(input_dir: str = PROCESSED_DIR) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(input_dir, "norm_params.npz")
    loaded = np.load(path)
    return loaded["mins"], loaded["maxs"]


# ==============================================================================
# FUNCTION THEORY: Window Extraction
# ------------------------------------------------------------------------------
# This takes the huge continuous array of data and slices it into overlapping 
# chunks. If any point in the chunk is anomalous (label == 1), the whole chunk 
# is labeled as anomalous.
# ==============================================================================
def _extract_windows(data: np.ndarray, labels: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = len(data)
    windows = []
    window_labels = []

    for start in range(0, n_samples - seq_len + 1, stride):
        end = start + seq_len
        windows.append(data[start:end])
        window_labels.append(1 if labels[start:end].max() > 0 else 0)

    return np.array(windows), np.array(window_labels)


def _save_processed(arrays: Dict[str, np.ndarray], output_dir: str = PROCESSED_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, arr in arrays.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
    print(f"  [data_loader] Processed arrays saved → {output_dir}/")

def _load_processed(input_dir: str = PROCESSED_DIR) -> Optional[Dict[str, np.ndarray]]:
    expected = [
        "train_features", "train_labels", "val_features", "val_labels",
        "test_features", "test_labels",
    ]
    arrays = {}
    for name in expected:
        path = os.path.join(input_dir, f"{name}.npy")
        if not os.path.exists(path):
            return None
        arrays[name] = np.load(path)
    return arrays


# ==============================================================================
# FUNCTION THEORY: Pipeline Execution
# ------------------------------------------------------------------------------
# This orchestrates everything: it loads the raw CSV, splits it chronologically, 
# normalises it, extracts windows, caches it to disk to save time on future runs, 
# and finally returns PyTorch DataLoaders ready for the training script!
# ==============================================================================
def get_dataloaders(
    csv_path: str = "data/raw/telemetry_anomalous.csv",
    seq_len: int = 50,
    batch_size: int = 64,
    stride: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    processed_dir: str = PROCESSED_DIR,
    force_reprocess: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    
    cached = None if force_reprocess else _load_processed(processed_dir)

    if cached is not None:
        print("  [data_loader] Loading cached processed arrays …")
        train_features = cached["train_features"]
        train_labels   = cached["train_labels"]
        val_features   = cached["val_features"]
        val_labels     = cached["val_labels"]
        test_features  = cached["test_features"]
        test_labels    = cached["test_labels"]
        mins, maxs     = load_norm_params(processed_dir)
    else:
        print(f"  [data_loader] Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        features = df[FEATURE_COLUMNS].values.astype(np.float64)
        labels   = df[LABEL_COLUMN].values.astype(np.int32)

        n_total = len(features)
        print(f"  [data_loader] Total timesteps: {n_total:,}")

        train_end = int(n_total * train_ratio)
        val_end   = int(n_total * (train_ratio + val_ratio))

        feat_train = features[:train_end]
        feat_val   = features[train_end:val_end]
        feat_test  = features[val_end:]

        lab_train = labels[:train_end]
        lab_val   = labels[train_end:val_end]
        lab_test  = labels[val_end:]

        print(f"  [data_loader] Split sizes — "
              f"train: {len(feat_train):,}  "
              f"val: {len(feat_val):,}  "
              f"test: {len(feat_test):,}")

        mins, maxs = _compute_norm_params(feat_train)
        save_norm_params(mins, maxs, processed_dir)

        feat_train = _apply_normalisation(feat_train, mins, maxs)
        feat_val   = _apply_normalisation(feat_val, mins, maxs)
        feat_test  = _apply_normalisation(feat_test, mins, maxs)

        print(f"  [data_loader] Extracting windows (seq_len={seq_len}, stride={stride}) …")

        train_features, train_labels = _extract_windows(feat_train, lab_train, seq_len, stride)
        val_features, val_labels = _extract_windows(feat_val, lab_val, seq_len, stride)
        test_features, test_labels = _extract_windows(feat_test, lab_test, seq_len, stride)

        print(f"  [data_loader] Window counts — "
              f"train: {len(train_features):,}  "
              f"val: {len(val_features):,}  "
              f"test: {len(test_features):,}")

        _save_processed({
            "train_features": train_features,
            "train_labels":   train_labels,
            "val_features":   val_features,
            "val_labels":     val_labels,
            "test_features":  test_features,
            "test_labels":    test_labels,
        }, processed_dir)

    train_ds = TelemetryDataset(train_features, train_labels)
    val_ds   = TelemetryDataset(val_features, val_labels)
    test_ds  = TelemetryDataset(test_features, test_labels)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )

    metadata = {
        "seq_len":    seq_len,
        "n_features": train_features.shape[-1],
        "train_size": len(train_ds),
        "val_size":   len(val_ds),
        "test_size":  len(test_ds),
        "mins":       mins,
        "maxs":       maxs,
    }

    print(f"\n  [data_loader] DataLoaders ready  (batch_size={batch_size}, seq_len={seq_len})")
    print(f"  [data_loader]   Train batches : {len(train_loader)}")
    print(f"  [data_loader]   Val batches   : {len(val_loader)}")
    print(f"  [data_loader]   Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    print("=" * 60)
    print("  data_loader.py — Smoke Test")
    print("=" * 60)

    default_csv = "data/raw/telemetry_anomalous.csv"
    if not os.path.exists(default_csv):
        print(f"\n    CSV not found: {default_csv}")
        print("  Run `python src/simulate_telemetry.py` first to generate data.")
        exit(1)

    train_dl, val_dl, test_dl, meta = get_dataloaders(
        csv_path=default_csv,
        seq_len=50,
        batch_size=32,
        stride=1,
        force_reprocess=True,
    )

    x_batch, y_batch = next(iter(train_dl))
    print(f"\n  Sample batch shapes:")
    print(f"    x: {x_batch.shape}  (batch, seq_len, n_features)")
    print(f"    y: {y_batch.shape}  (batch, 1)")
    print(f"    x dtype: {x_batch.dtype}")
    print(f"    y dtype: {y_batch.dtype}")
    print(f"    x range: [{x_batch.min():.4f}, {x_batch.max():.4f}]")
    print(f"    Anomaly ratio in batch: {y_batch.mean():.2%}")
    print()
