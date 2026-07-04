import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.lnn_core import LNNAnomalyDetector
from data.data_loader import get_dataloaders
from utils.metrics import precision, recall, f1_score, accuracy, classification_report
from utils.visualization import plot_training_curves

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# ==============================================================================
# FUNCTION THEORY: The Training Loop
# ------------------------------------------------------------------------------
# Here we actually train the neural network using backpropagation! 
# For every batch of data, the model makes a prediction. We calculate the Error 
# (Loss) using Binary Cross-Entropy. 
#
# Gradient Clipping is used here because Recurrent Neural Networks (like our LNN)
# are notorious for "Exploding Gradients" where the math blows up to infinity.
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> tuple:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)  
        batch_y = batch_y.to(device)  

        outputs = model(batch_x)  
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

        preds = (outputs.detach().cpu().numpy() >= 0.5).astype(int).flatten()
        labels = batch_y.detach().cpu().numpy().astype(int).flatten()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds)

    return avg_loss, epoch_f1


# ==============================================================================
# FUNCTION THEORY: Validation Loop
# ------------------------------------------------------------------------------
# We run the model on data it hasn't trained on (the validation set) to make 
# sure it isn't just memorizing the training data (overfitting). We use 
# torch.no_grad() here to save memory, since we don't need to calculate 
# gradients for learning!
# ==============================================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        running_loss += loss.item() * batch_x.size(0)

        preds = (outputs.cpu().numpy() >= 0.5).astype(int).flatten()
        labels = batch_y.cpu().numpy().astype(int).flatten()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds)

    return avg_loss, epoch_f1


# ==============================================================================
# FUNCTION THEORY: Main Orchestrator
# ------------------------------------------------------------------------------
# This function pulls everything together:
# 1. Sets up the DataLoaders
# 2. Builds the Neural Network
# 3. Solves the "Class Imbalance" problem by weighting the Loss function
# 4. Sets up the AdamW Optimizer and Cosine Annealing Learning Rate
# 5. Runs the Epochs and saves the Best Model
# ==============================================================================
def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  LNN Telemetry Engine — Training Pipeline")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    data_cfg = config["data"]
    csv_path = os.path.join(data_cfg["raw_dir"], "telemetry_anomalous.csv")
    train_loader, val_loader, test_loader, data_meta = get_dataloaders(
        csv_path=csv_path,
        processed_dir=data_cfg["processed_dir"],
        seq_len=data_cfg["sequence_length"],
        stride=data_cfg["stride"],
        batch_size=config["training"]["batch_size"],
        train_ratio=data_cfg["train_split"],
        val_ratio=data_cfg["val_split"],
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    model_cfg = config["model"]
    model = LNNAnomalyDetector(
        input_size=model_cfg["input_size"],
        units=model_cfg["units"],
        output_size=model_cfg["output_size"],
        use_sparse_wiring=model_cfg["use_sparse_wiring"],
    ).to(device)

    total_params = model.count_parameters()
    print(f"\n  Model: LNNAnomalyDetector (CfC + AutoNCP)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Neurons: {model_cfg['units']} | Output: {model_cfg['output_size']}")
    print(f"  Sparse wiring: {model_cfg['use_sparse_wiring']}")
    print(f"{'='*60}\n")

    # --------------------------------------------------------------------------
    # THEORY: Class Imbalance
    # Anomalies are rare! If 95% of our data is "Normal", the network can get 
    # 95% accuracy just by being lazy and guessing "Normal" every single time.
    # To fix this, we heavily penalize the network for missing an anomaly. 
    # We do this using `pos_weight` in the Binary Cross-Entropy function.
    # --------------------------------------------------------------------------
    train_labels = train_loader.dataset.labels
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight_val = n_neg / max(n_pos, 1)
    print(f"  Class balance: {n_neg:.0f} normal, {n_pos:.0f} anomalous (pos_weight={pos_weight_val:.2f})")

    def weighted_bce(output, target):
        weights = torch.where(target == 1, pos_weight_val, 1.0)
        bce = nn.functional.binary_cross_entropy(output, target, reduction='none')
        return (bce * weights).mean()

    criterion = weighted_bce

    # --------------------------------------------------------------------------
    # THEORY: AdamW Optimizer & Cosine Annealing
    # AdamW is an amazing optimizer that handles Weight Decay correctly.
    # Cosine Annealing slowly lowers the Learning Rate (like a cosine wave) 
    # so the model takes big steps early on, and tiny precise steps at the end.
    # --------------------------------------------------------------------------
    train_cfg = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"] - train_cfg.get("warmup_epochs", 0),
        eta_min=1e-6,
    )

    best_val_f1 = 0.0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 10)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    checkpoint_dir = os.path.dirname(config["evaluation"]["checkpoint_path"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=train_cfg.get("gradient_clip", 1.0),
        )

        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "config": config,
                },
                config["evaluation"]["checkpoint_path"],
            )
            marker = "  saved"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"  Epoch {epoch:3d}/{train_cfg['epochs']} │ "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} │ "
            f"F1: {train_f1:.4f}/{val_f1:.4f} │ "
            f"LR: {current_lr:.2e}{marker}"
        )

        # Early Stopping: If it stops learning, we stop training to save time!
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Best validation F1: {best_val_f1:.4f}")
    print(f"{'='*60}")

    print("\nEvaluating on test set with best checkpoint...\n")
    checkpoint = torch.load(
        config["evaluation"]["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_f1 = validate(model, test_loader, criterion, device)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = (outputs.cpu().numpy() >= 0.5).astype(int).flatten()
            labels = batch_y.numpy().astype(int).flatten()
            all_preds.extend(preds)
            all_labels.extend(labels)

    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test F1:   {test_f1:.4f}")
    print()
    classification_report(all_labels, all_preds)

    viz_cfg = config.get("visualization", {})
    figures_dir = viz_cfg.get("output_dir", "outputs/figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_training_curves(
        train_losses=history["train_loss"],
        val_losses=history["val_loss"],
        train_f1s=history["train_f1"],
        val_f1s=history["val_f1"],
        save_path=os.path.join(figures_dir, "training_curves.png"),
    )
    print(f"\n  Training curves saved to {figures_dir}/training_curves.png")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train the LNN Telemetry Engine")
    parser.add_argument("--config", type=str, default="configs/hyperparams.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    train(config)


if __name__ == "__main__":
    main()
