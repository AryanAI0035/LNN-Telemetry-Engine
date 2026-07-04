"""
models/__init__.py
==================
Package initializer for the LNN Telemetry Engine model architecture.

Exports the two core building blocks:
  - LNNAnomalyDetector : the end-to-end anomaly scoring network
  - build_wiring / get_wiring_summary : helpers for the sparse NCP wiring
"""

from .lnn_core import LNNAnomalyDetector
from .sparse_wiring import build_wiring, get_wiring_summary
