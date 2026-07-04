# ==============================================================================
# THEORY: Liquid Neural Networks (LNNs)
# ------------------------------------------------------------------------------
# Welcome to the core of the engine! 
# 
# Standard neural networks (like CNNs or LSTMs) operate on discrete time steps.
# Liquid Neural Networks (LNNs) are different. They are continuous-time recurrent 
# neural networks inspired by the tiny brain of the C. elegans worm.
#
# The hidden state of an LNN isn't just updated by a simple math function; 
# it's governed by an Ordinary Differential Equation (ODE). This allows the 
# network to process incoming data at any irregular time interval and adapt 
# dynamically to the environment (hence the name "Liquid").
#
# Specifically, we use the Closed-form Continuous-depth (CfC) architecture. 
# Solving ODEs numerically (like with standard Neural ODEs) is very slow. 
# CfCs use a mathematical closed-form solution to approximate the ODE, giving 
# us the power of a continuous-time network but running as fast as an LSTM!
# ==============================================================================

import torch
import torch.nn as nn

from ncps.torch import CfC
from ncps.wirings import AutoNCP

from .sparse_wiring import build_wiring


class LNNAnomalyDetector(nn.Module):
    # ==========================================================================
    # CLASS THEORY: The Anomaly Detector
    # --------------------------------------------------------------------------
    # This class wraps the CfC network. It takes in a sequence of telemetry 
    # data (like battery voltage and temperature over time) and outputs a single 
    # score between 0 and 1. 
    # 0 means "Normal", 1 means "Anomaly".
    # ==========================================================================
    
    def __init__(
        self,
        input_size: int,
        units: int = 32,
        output_size: int = 1,
        use_sparse_wiring: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.units = units
        self.output_size = output_size
        self.use_sparse_wiring = use_sparse_wiring

        # ----------------------------------------------------------------------
        # THEORY: Wiring Topology
        # We can either wire every neuron to every other neuron (Dense), or we
        # can wire them sparsely like a real brain (Sparse/AutoNCP). Sparse 
        # wiring saves a huge amount of memory and prevents overfitting.
        # ----------------------------------------------------------------------
        if use_sparse_wiring:
            self.wiring = build_wiring(units=units, output_size=output_size)
            self.backbone = CfC(input_size=input_size, units=self.wiring)
            self.output_head = nn.Sigmoid()
            self._backbone_out_dim = output_size
        else:
            self.wiring = None
            self.backbone = CfC(input_size=input_size, units=units)
            self.output_head = nn.Sequential(
                nn.Linear(units, output_size),
                nn.Sigmoid()
            )
            self._backbone_out_dim = units

    
    # ==========================================================================
    # FUNCTION THEORY: The Forward Pass
    # --------------------------------------------------------------------------
    # When data passes through the network, the CfC backbone calculates the
    # hidden state for every single time step. 
    # However, since we are doing sequence classification (looking at a whole 
    # window of data to make one decision), we only care about the very LAST 
    # time step's output to make our anomaly prediction!
    # ==========================================================================
    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor = None,
        return_hidden: bool = False,
    ):
        if hidden is not None:
            backbone_out, hn = self.backbone(x, hx=hidden)
        else:
            backbone_out, hn = self.backbone(x)

        # Grab the last timestep 
        last_timestep = backbone_out[:, -1, :] 

        # Apply the output head to get our prediction
        predictions = self.output_head(last_timestep)

        if return_hidden:
            return predictions, backbone_out
        else:
            return predictions

    
    # ==========================================================================
    # FUNCTION THEORY: Parameter Counting
    # --------------------------------------------------------------------------
    # For edge computing on satellites or rovers, we have strict memory limits. 
    # This helper counts exactly how many trainable weights the model has, so 
    # we can calculate its RAM footprint.
    # ==========================================================================
    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

    def model_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  LNN Telemetry Engine — Model Summary")
        print("=" * 60)

        mode = "Sparse (AutoNCP)" if self.use_sparse_wiring else "Dense (Fully Connected)"
        print(f"\n  Mode             : {mode}")
        print(f"  Input features   : {self.input_size}")
        print(f"  CfC neurons      : {self.units}")
        print(f"  Output size      : {self.output_size}")
        print(f"  Backbone out dim : {self._backbone_out_dim}")

        total_params = self.count_parameters()
        memory_fp32_kb = (total_params * 4) / 1024  
        memory_int8_kb = (total_params * 1) / 1024  

        print(f"\n  Trainable params : {total_params:,}")
        print(f"  Memory (FP32)    : {memory_fp32_kb:.1f} KB")
        print(f"  Memory (INT8)    : {memory_int8_kb:.1f} KB")

        print(f"\n  Layer Breakdown:")
        print(f"  {'-' * 50}")
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"    {name:20s} → {n_params:>8,} params  ({type(module).__name__})")

        lstm_params = 4 * (self.input_size * self.units + self.units * self.units + self.units)
        ratio = lstm_params / total_params if total_params > 0 else float("inf")
        print(f"\n  Comparable LSTM  : ~{lstm_params:,} params ({ratio:.1f}× larger)")
        print(f"    LNN is {ratio:.1f}× more parameter-efficient than LSTM")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n  LNN Core — Smoke Test\n")

    print("=" * 60)
    print("  TEST 1: Sparse Wiring (AutoNCP)")
    print("=" * 60)

    model_sparse = LNNAnomalyDetector(input_size=8, units=32, use_sparse_wiring=True)
    model_sparse.model_summary()

    dummy_input = torch.randn(4, 50, 8)
    preds = model_sparse(dummy_input)
    print(f"\n  Input shape      : {dummy_input.shape}")
    print(f"  Output shape     : {preds.shape}")
    print(f"  Output range     : [{preds.min().item():.4f}, {preds.max().item():.4f}]")

    preds_h, hidden_states = model_sparse(dummy_input, return_hidden=True)
    print(f"  Hidden states    : {hidden_states.shape}")

    print("\n" + "=" * 60)
    print("  TEST 2: Fully Connected (Dense Baseline)")
    print("=" * 60)

    model_dense = LNNAnomalyDetector(input_size=8, units=32, use_sparse_wiring=False)
    model_dense.model_summary()

    preds_dense = model_dense(dummy_input)
    print(f"\n  Input shape      : {dummy_input.shape}")
    print(f"  Output shape     : {preds_dense.shape}")
    print(f"  Output range     : [{preds_dense.min().item():.4f}, {preds_dense.max().item():.4f}]")

    print("\n✅  All smoke tests passed!")
