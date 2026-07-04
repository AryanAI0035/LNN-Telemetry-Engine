# ==============================================================================
# THEORY: Brain-Inspired Sparse Wiring (AutoNCP)
# ------------------------------------------------------------------------------
# In traditional deep learning, layers are "dense"—meaning every neuron connects 
# to every neuron in the next layer. This uses huge amounts of memory.
#
# Biologists mapping the brain of the C. elegans worm discovered that its 302 
# neurons are wired sparsely, organized into specific layers:
#   Sensory (Input) -> Inter (Hidden) -> Command (Decision) -> Motor (Output)
#
# Neural Circuit Policies (NCP) is a framework that mimics this biological 
# wiring. Instead of dense matrix multiplication, we use a sparse topology.
# This results in incredibly tiny models (great for microcontrollers!) that 
# don't overfit easily, because the structure naturally acts as a regularizer.
# ==============================================================================

import os
import sys

from ncps.wirings import AutoNCP 


# ==============================================================================
# FUNCTION THEORY: Building the Wiring
# ------------------------------------------------------------------------------
# The AutoNCP class automatically figures out the best way to distribute our 
# neurons across the Sensory, Inter, Command, and Motor categories, and connects 
# them sparsely.
# ==============================================================================
def build_wiring(units: int = 32, output_size: int = 1) -> AutoNCP:
    wiring = AutoNCP(units=units, output_size=output_size)
    return wiring


# ==============================================================================
# FUNCTION THEORY: Inspecting the Topology
# ------------------------------------------------------------------------------
# Before we train, it's cool to look at exactly how many synapses (connections) 
# were generated compared to a traditional dense network.
# ==============================================================================
def get_wiring_summary(wiring: AutoNCP) -> dict:
    total_neurons = wiring.units
    motor_neurons = wiring.output_dim

    try:
        import numpy as np
        adj = np.array(wiring.adjacency_matrix)
        total_synapses = int(np.count_nonzero(adj))
    except (AttributeError, Exception):
        total_synapses = -1  

    summary = {
        "total_neurons": total_neurons,
        "motor_neurons": motor_neurons,
        "total_synapses": total_synapses,
    }

    print("=" * 55)
    print("  LNN Telemetry Engine — Wiring Summary")
    print("=" * 55)
    print(f"  Total neurons (units)  : {total_neurons}")
    print(f"  Motor neurons (output) : {motor_neurons}")
    if total_synapses >= 0:
        print(f"  Total synapses         : {total_synapses}")
        max_synapses = total_neurons * total_neurons
        sparsity = 1.0 - (total_synapses / max_synapses) if max_synapses > 0 else 0.0
        print(f"  Sparsity               : {sparsity:.1%}")
    else:
        print("  Total synapses         : (unavailable before binding)")
    print("=" * 55)

    return summary


# ==============================================================================
# FUNCTION THEORY: Visualization
# ------------------------------------------------------------------------------
# This function draws the brain-like structure of our network so we can see 
# the sparse connections visually!
# ==============================================================================
def visualize_wiring(wiring: AutoNCP, save_path: str = "wiring_diagram.png") -> None:
    import matplotlib
    matplotlib.use("Agg")  
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if hasattr(wiring, "draw_graph"):
        try:
            plt.figure(figsize=(10, 6))
            wiring.draw_graph()
            plt.title("AutoNCP Sparse Wiring — C. elegans Inspired", fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[] Wiring diagram saved to: {save_path}")
            return
        except Exception as e:
            print(f"[!] draw_graph() failed ({e}), falling back to manual diagram.")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    n = wiring.units
    motor = wiring.output_dim
    inter = max(1, n // 3)
    command = max(1, n // 3)
    sensory = n - inter - command - motor

    layers = [
        (0.5, "Sensory\n(Input)", max(sensory, 1), "#58a6ff"),
        (1.7, "Inter\n(Hidden)", max(inter, 1), "#3fb950"),
        (2.9, "Command\n(Decision)", max(command, 1), "#d29922"),
        (4.1, "Motor\n(Output)", motor, "#f85149"),
    ]

    for x, label, count, color in layers:
        y_start = (7 - count) / 2
        for i in range(min(count, 7)):  
            y = y_start + i * 0.9
            circle = plt.Circle((x, y), 0.3, color=color, alpha=0.75, zorder=3)
            ax.add_patch(circle)
        if count > 7:
            ax.text(x, y_start - 0.7, f"(+{count - 7} more)", ha="center",
                    fontsize=8, color="white", alpha=0.7)
        ax.text(x, -0.5, label, ha="center", fontsize=10, color="white",
                fontweight="bold")
        ax.text(x, -1.0, f"n={count}", ha="center", fontsize=9, color="gray")

    for i in range(len(layers) - 1):
        x1 = layers[i][0] + 0.35
        x2 = layers[i + 1][0] - 0.35
        y_mid = 3.5
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=dict(arrowstyle="->", color="white",
                                     lw=2, alpha=0.5))

    ax.set_title("AutoNCP Sparse Wiring — C. elegans Inspired",
                 fontsize=14, color="white", pad=20)
    legend_handles = [
        mpatches.Patch(color=c, label=l.split("\n")[0])
        for _, l, _, c in layers
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              facecolor="#161b22", edgecolor="gray", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[] Wiring diagram saved to: {save_path}")


if __name__ == "__main__":
    print("\n  Building C. elegans-inspired sparse wiring...\n")

    wiring = build_wiring(units=32, output_size=1)
    summary = get_wiring_summary(wiring)

    output_path = os.path.join(os.path.dirname(__file__), "wiring_diagram.png")
    print(f"\n  Generating wiring visualization...")
    visualize_wiring(wiring, save_path=output_path)

    print(f"\n✅  Done!  Summary dict: {summary}")
