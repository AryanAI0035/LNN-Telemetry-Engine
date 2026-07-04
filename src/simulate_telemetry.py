#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd

# ==============================================================================
# THEORY: Synthetic Data Generation
# ------------------------------------------------------------------------------
# Why generate our own data? Because real-world satellite telemetry with *perfectly 
# labeled anomalies* is incredibly rare and usually classified by the military or NASA.
# 
# To train our Liquid Neural Network, we need two things:
# 1. Clean Data (normal satellite operations)
# 2. Anomalous Data (where things go wrong, like a battery failure)
#
# Our simulation creates realistic noise and orbital cycles (like heating up in 
# the sun and cooling down in the Earth's shadow). Then, we randomly inject 
# 4 types of failures into the data and label those exact timestamps with a '1'.
# ==============================================================================

CHANNEL_SPECS = {
    "battery_voltage": (3.0, 4.2, 0.02),       
    "solar_current":   (0.0, 0.5, 0.01),        
    "cpu_temp":        (20.0, 45.0, 0.5),        
    "panel_temp":      (-40.0, 60.0, 1.0),       
    "gyro_x":          (-1.0, 1.0, 0.05),        
    "gyro_y":          (-1.0, 1.0, 0.05),        
    "gyro_z":          (-1.0, 1.0, 0.05),        
    "magnetometer":    (-50.0, 50.0, 0.5),       
}

CHANNEL_NAMES = list(CHANNEL_SPECS.keys())


def _generate_baseline(low: float, high: float, noise_std: float, n: int, rng: np.random.Generator) -> np.ndarray:
    mid = rng.uniform(low + 0.2 * (high - low), high - 0.2 * (high - low))
    period = rng.uniform(800, 1200)
    phase  = rng.uniform(0, 2 * np.pi)
    amplitude = 0.15 * (high - low)  
    t = np.arange(n, dtype=np.float64)
    signal = mid + amplitude * np.sin(2 * np.pi * t / period + phase)
    signal += rng.normal(0.0, noise_std, size=n)
    return signal


# ==============================================================================
# FUNCTION THEORY: Anomaly Injection
# ------------------------------------------------------------------------------
# Here we mathematically simulate four common space hardware failures:
# 1. Voltage Sag: The battery suddenly drops power (maybe an eclipse started).
# 2. Thermal Spike: The CPU or Solar Panel overheats rapidly.
# 3. Spin Drift: A reaction wheel breaks and the satellite starts slowly spinning.
# 4. Sensor Dropout: Cosmic radiation flips a bit, causing the sensor to output 0.0 or NaN.
# ==============================================================================
def _inject_voltage_sag(data: np.ndarray, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    col = CHANNEL_NAMES.index("battery_voltage")
    labels = np.zeros(len(data), dtype=np.int32)
    for idx in indices:
        duration = rng.integers(5, 21)
        end = min(idx + duration, len(data))
        drop = rng.uniform(0.5, 1.5)
        data[idx:end, col] -= drop
        labels[idx:end] = 1
    return labels

def _inject_thermal_spike(data: np.ndarray, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    target = rng.choice(["cpu_temp", "panel_temp"])
    col = CHANNEL_NAMES.index(target)
    labels = np.zeros(len(data), dtype=np.int32)
    for idx in indices:
        duration = rng.integers(10, 31)
        end = min(idx + duration, len(data))
        spike_mag = rng.uniform(15.0, 40.0)
        profile = np.concatenate([
            np.linspace(0, spike_mag, (end - idx) // 2 + 1),
            np.linspace(spike_mag, 0, (end - idx) - (end - idx) // 2)
        ])[: end - idx]
        data[idx:end, col] += profile
        labels[idx:end] = 1
    return labels

def _inject_spin_drift(data: np.ndarray, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    target = rng.choice(["gyro_x", "gyro_y", "gyro_z"])
    col = CHANNEL_NAMES.index(target)
    labels = np.zeros(len(data), dtype=np.int32)
    for idx in indices:
        duration = rng.integers(50, 201)
        end = min(idx + duration, len(data))
        drift_rate = rng.uniform(0.5, 3.0) / duration
        direction = rng.choice([-1, 1])
        drift = direction * drift_rate * np.arange(end - idx)
        data[idx:end, col] += drift
        labels[idx:end] = 1
    return labels

def _inject_sensor_dropout(data: np.ndarray, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    col = rng.integers(0, len(CHANNEL_NAMES))
    labels = np.zeros(len(data), dtype=np.int32)
    for idx in indices:
        duration = rng.integers(5, 16)
        end = min(idx + duration, len(data))
        fill = 0.0 if rng.random() < 0.5 else np.nan
        data[idx:end, col] = fill
        labels[idx:end] = 1
    return labels


ANOMALY_INJECTORS = [
    _inject_voltage_sag,
    _inject_thermal_spike,
    _inject_spin_drift,
    _inject_sensor_dropout,
]


def generate_telemetry(num_samples: int = 10000, anomaly_ratio: float = 0.05, seed: int = 42, output_dir: str = "data/raw") -> dict:
    rng = np.random.default_rng(seed)

    print("[1/4] Generating clean baselines for {} channels …".format(len(CHANNEL_NAMES)))
    baselines = {}
    for name in CHANNEL_NAMES:
        low, high, noise = CHANNEL_SPECS[name]
        baselines[name] = _generate_baseline(low, high, noise, num_samples, rng)
    data_clean = np.column_stack([baselines[ch] for ch in CHANNEL_NAMES])

    print("[2/4] Building normal telemetry DataFrame …")
    timestamps = pd.date_range(start="2026-01-01T00:00:00", periods=num_samples, freq="1s")
    df_normal = pd.DataFrame(data_clean, columns=CHANNEL_NAMES)
    df_normal.insert(0, "timestamp", timestamps)
    df_normal["anomaly"] = 0  

    print("[3/4] Injecting anomalies (target ratio ≈ {:.1%}) …".format(anomaly_ratio))
    data_anom = data_clean.copy()
    labels = np.zeros(num_samples, dtype=np.int32)

    avg_event_len = 30
    n_events = max(1, int(num_samples * anomaly_ratio / avg_event_len))

    event_starts = rng.choice(np.arange(100, num_samples - 200), size=n_events, replace=False)
    event_starts.sort()

    for start_idx in event_starts:
        injector = rng.choice(ANOMALY_INJECTORS)
        event_labels = injector(data_anom, np.array([start_idx]), rng)
        labels = np.maximum(labels, event_labels)  

    df_anomalous = pd.DataFrame(data_anom, columns=CHANNEL_NAMES)
    df_anomalous.insert(0, "timestamp", timestamps)
    df_anomalous["anomaly"] = labels

    print("[4/4] Saving CSVs …")
    os.makedirs(output_dir, exist_ok=True)
    normal_path = os.path.join(output_dir, "telemetry_normal.csv")
    anomalous_path = os.path.join(output_dir, "telemetry_anomalous.csv")

    df_normal.to_csv(normal_path, index=False)
    df_anomalous.to_csv(anomalous_path, index=False)

    actual_anomaly_ratio = labels.sum() / num_samples
    stats = {
        "num_samples": num_samples, "num_channels": len(CHANNEL_NAMES),
        "anomaly_events": n_events, "anomalous_timesteps": int(labels.sum()),
        "actual_anomaly_ratio": actual_anomaly_ratio,
        "normal_path": normal_path, "anomalous_path": anomalous_path,
    }

    print("\n" + "=" * 60)
    print("  Telemetry Generation Complete")
    print("=" * 60)
    print(f"  Total timesteps      : {num_samples:,}")
    print(f"  Sensor channels      : {len(CHANNEL_NAMES)}")
    print(f"  Anomaly events       : {n_events}")
    print(f"  Anomalous timesteps  : {int(labels.sum()):,}  ({actual_anomaly_ratio:.2%})")
    print(f"  Normal CSV           : {normal_path}")
    print(f"  Anomalous CSV        : {anomalous_path}")
    print("=" * 60)

    print("\n  Per-channel statistics (anomalous dataset):")
    print(f"  {'Channel':<20s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("  " + "-" * 54)
    for i, ch in enumerate(CHANNEL_NAMES):
        col = data_anom[:, i]
        print(f"  {ch:<20s} {np.nanmean(col):>10.4f} {np.nanstd(col):>10.4f} "
              f"{np.nanmin(col):>10.4f} {np.nanmax(col):>10.4f}")
    print()

    return {"normal_path": normal_path, "anomalous_path": anomalous_path, "stats": stats}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic CubeSat telemetry data")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--anomaly_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/raw")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    generate_telemetry(num_samples=args.num_samples, anomaly_ratio=args.anomaly_ratio, seed=args.seed, output_dir=args.output_dir)
