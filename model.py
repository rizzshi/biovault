"""BioVault Client-Side Threat Detection Model and Imbalance-Aware Local Training.

Defines the ThreatMLP neural network architecture tailored for IoMT flow classification
and a local training routine employing weighted CrossEntropyLoss derived from
client-specific inverse class frequencies to mitigate extreme data skew.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from contracts import AttackClass


class ThreatMLP(nn.Module):
    """Deep Multi-Layer Perceptron for 5-Class IoMT Threat Detection.

    Architecture:
        - Block 1: Linear(input_dim, hidden) -> BatchNorm1d(hidden) -> ReLU -> Dropout(0.3)
        - Block 2: Linear(hidden, hidden // 2) -> BatchNorm1d(hidden // 2) -> ReLU -> Dropout(0.2)
        - Output Layer: Linear(hidden // 2, num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int = 5, hidden: int = 256):
        """Initialize ThreatMLP layers.

        Args:
            input_dim: Number of input flow statistical features.
            num_classes: Number of threat taxonomy target classes (default: 5).
            hidden: Dimensionality of the first hidden projection layer (default: 256).
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden = hidden

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.out_proj = nn.Linear(hidden // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x: Input tensor of shape [B, input_dim].

        Returns:
            torch.Tensor: Unnormalized class logits of shape [B, num_classes].
        """
        # Safeguard: if a single-sample batch occurs during training, avoid BatchNorm1d crash
        if self.training and x.size(0) == 1:
            self.eval()
            h = self.block1(x)
            h = self.block2(h)
            out = self.out_proj(h)
            self.train()
            return out

        h = self.block1(x)
        h = self.block2(h)
        return self.out_proj(h)


def build_model(input_dim: int, num_classes: int = 5, hidden: int = 256) -> ThreatMLP:
    """Construct an initialized ThreatMLP model instance.

    Args:
        input_dim: Number of input flow statistical features.
        num_classes: Target classes (default: 5).
        hidden: Hidden layer capacity (default: 256).

    Returns:
        ThreatMLP: Instantiated PyTorch module ready for local training or FL dispatch.
    """
    return ThreatMLP(input_dim=input_dim, num_classes=num_classes, hidden=hidden)


def compute_class_weights(
    train_loader: DataLoader,
    num_classes: int = 5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute normalized inverse class frequency weights from local client batches.

    Mitigates extreme IoMT class imbalance (e.g. benign traffic heavily dominating
    rare ransomware or data tampering events) without leaking data across clients.

    Args:
        train_loader: DataLoader supplying local client training samples.
        num_classes: Number of target classes in the taxonomy (default: 5).
        device: Target PyTorch device for tensor allocation.

    Returns:
        torch.Tensor: Normalized class weight vector of shape [num_classes].
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, y in train_loader:
        for c in range(num_classes):
            class_counts[c] += (y == c).sum().item()

    total_samples = class_counts.sum().item()
    weights = torch.ones(num_classes, dtype=torch.float32)

    present_mask = class_counts > 0
    num_present = present_mask.sum().item()

    if num_present > 0 and total_samples > 0:
        # Standard balanced inverse frequency: N / (num_present * count)
        weights[present_mask] = total_samples / (num_present * class_counts[present_mask])
        # Normalize weights so the mean across active classes is 1.0
        weights[present_mask] = weights[present_mask] / weights[present_mask].mean()
        # Classes not present in this client's partition receive zero weight
        weights[~present_mask] = 0.0

    if device is not None:
        weights = weights.to(device)

    return weights


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, float, float]:
    """Train ThreatMLP locally on client data for the specified number of epochs.

    Utilizes weighted CrossEntropyLoss derived from client label frequencies
    to handle severe class imbalance without cross-institution leakage.

    Args:
        model: ThreatMLP PyTorch module.
        train_loader: PyTorch DataLoader providing client local batches.
        epochs: Number of local training epochs (default: 1).
        lr: Adam learning rate (default: 1e-3).
        class_weights: Optional precomputed class weights tensor of shape [num_classes].
            If None, computed dynamically from train_loader via inverse frequency.
        device: Optional PyTorch device. If None, auto-selects GPU if available.

    Returns:
        Tuple[nn.Module, float, float]: (trained_model, avg_train_loss, train_accuracy).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)

    if class_weights is None:
        class_weights = compute_class_weights(train_loader, num_classes=5, device=dev)
    else:
        class_weights = class_weights.to(dev)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(dev)
            y_batch = y_batch.to(dev)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            b_size = y_batch.size(0)
            total_loss += loss.item() * b_size
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += b_size

    avg_loss = total_loss / max(1, total_samples)
    train_accuracy = total_correct / max(1, total_samples)

    return model, float(avg_loss), float(train_accuracy)


if __name__ == "__main__":
    print("=" * 80)
    print("BioVault Model Verification: ThreatMLP & train_local")
    print("=" * 80)

    # 1. Forward pass on a dummy batch [8, num_features]
    input_dim = 36
    num_classes = 5
    batch_size = 8

    model = ThreatMLP(input_dim=input_dim, num_classes=num_classes, hidden=256)
    dummy_input = torch.randn(batch_size, input_dim)
    logits = model(dummy_input)

    print(f"\n[1] Forward pass verification:")
    print(f"    Input shape:  {list(dummy_input.shape)}")
    print(f"    Logits shape: {list(logits.shape)}")
    assert logits.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
    print("    SUCCESS: Forward pass correctly returned logits [8, 5].")

    # 2. train_local verification on a dummy loader
    from torch.utils.data import TensorDataset

    num_samples = 64
    dummy_X = torch.randn(num_samples, input_dim)
    # Severe class imbalance: mostly benign (0), few ransomware (2), few tampering (3)
    dummy_y = torch.tensor([0] * 50 + [1] * 2 + [2] * 4 + [3] * 6 + [4] * 2, dtype=torch.long)

    dummy_dataset = TensorDataset(dummy_X, dummy_y)
    dummy_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)

    print(f"\n[2] Testing train_local with class imbalance weighting:")
    weights = compute_class_weights(dummy_loader, num_classes=5)
    print(f"    Computed inverse-frequency class weights: {weights.tolist()}")

    trained_model, loss, acc = train_local(model, dummy_loader, epochs=1, lr=1e-3)
    print(f"    Trained 1 epoch -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    assert np.isfinite(loss), f"Loss must be finite, got {loss}"
    assert 0.0 <= acc <= 1.0, f"Accuracy must be in [0, 1], got {acc}"
    print("    SUCCESS: train_local ran 1 epoch and returned finite loss and valid accuracy.")

    print("\n" + "=" * 80)
    print("MODEL MODULE VERIFICATION PASSED!")
    print("=" * 80)
