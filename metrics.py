"""BioVault Clinical & Cyber Threat Metrics for IoMT Intrusion Detection.

Implements clinical evaluation routines prioritizing ransomware detection efficacy:
- Accuracy, Macro-F1, Per-Class Recall
- Ransomware Recall and False Negative Rate (FNR) — where missing a ransomware infection
  is clinically catastrophic compared to a false alert.
- Multi-Class Confusion Matrix for detailed threat taxonomy diagnostic inspection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from contracts import AttackClass


def confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int = 5,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """Compute the multi-class confusion matrix over a dataset loader.

    Args:
        model: ThreatMLP or compatible PyTorch module.
        loader: DataLoader providing evaluation batches.
        num_classes: Total number of classes in the threat taxonomy (default: 5).
        device: Target PyTorch evaluation device (CPU or CUDA).

    Returns:
        np.ndarray: Confusion matrix of shape [num_classes, num_classes], where
            cm[i, j] denotes ground-truth class i predicted as class j.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(dev)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = y_batch.numpy()

            for t, p in zip(targets, preds):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1

    return cm


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int = 5,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Evaluate model performance with clinical emphasis on Ransomware detection.

    Computes overall accuracy, macro-F1, per-class recall, and specifically
    isolates Ransomware Recall and False Negative Rate (FNR). In connected medical
    devices and hospital networks, false negatives for ransomware allow encryption
    payloads to propagate, threatening patient care and device telemetry continuity.

    Args:
        model: ThreatMLP PyTorch module.
        loader: DataLoader providing evaluation samples.
        num_classes: Number of threat classes (default: 5).
        device: Target PyTorch evaluation device.

    Returns:
        Dict[str, Any]: Evaluation metrics containing:
            - 'accuracy': Overall classification accuracy (float).
            - 'macro_f1': Macro-averaged F1 score across present classes (float).
            - 'per_class_recall': Dict mapping class index (0-4) to recall (float).
            - 'ransomware_recall': True Positive Rate for AttackClass.RANSOMWARE (float).
            - 'ransomware_fnr': False Negative Rate for AttackClass.RANSOMWARE (float).
            - 'per_class_f1': Dict mapping class index to F1 score (float).
            - 'per_class_precision': Dict mapping class index to precision (float).
            - 'loss': Average CrossEntropyLoss on the evaluation split (float).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    all_targets: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(dev)
            y_dev = y_batch.to(dev)

            logits = model(X_batch)
            loss = criterion(logits, y_dev)

            b_size = y_batch.size(0)
            total_loss += loss.item() * b_size
            total_samples += b_size

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(y_batch.numpy().tolist())

    avg_loss = total_loss / max(1, total_samples)

    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_targets, all_preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # Overall accuracy
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = float(correct / max(1, total))

    # Per-class metrics
    per_class_recall: Dict[int, float] = {}
    per_class_precision: Dict[int, float] = {}
    per_class_f1: Dict[int, float] = {}

    f1_list: List[float] = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fn = float(np.sum(cm[c, :]) - tp)
        fp = float(np.sum(cm[:, c]) - tp)

        # Recall = TP / (TP + FN)
        actual_positives = tp + fn
        recall = tp / actual_positives if actual_positives > 0 else 0.0
        per_class_recall[c] = float(recall)

        # Precision = TP / (TP + FP)
        predicted_positives = tp + fp
        precision = tp / predicted_positives if predicted_positives > 0 else 0.0
        per_class_precision[c] = float(precision)

        # F1 Score = 2 * P * R / (P + R)
        if (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class_f1[c] = float(f1)

        # Include in macro-F1 if class was present in evaluation targets
        if actual_positives > 0:
            f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    # Clinical ransomware metrics (AttackClass.RANSOMWARE = 2)
    ransomware_class = int(AttackClass.RANSOMWARE)
    ransomware_recall = float(per_class_recall.get(ransomware_class, 0.0))

    actual_ransomware = float(np.sum(cm[ransomware_class, :]))
    if actual_ransomware > 0:
        ransomware_fnr = float(1.0 - ransomware_recall)
    else:
        # No actual ransomware samples in this specific slice
        ransomware_fnr = 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "ransomware_recall": ransomware_recall,
        "ransomware_fnr": ransomware_fnr,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "loss": float(avg_loss),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("BioVault Metrics Verification: evaluate & confusion_matrix")
    print("=" * 80)

    from model import ThreatMLP
    from torch.utils.data import TensorDataset

    input_dim = 36
    num_classes = 5

    # Create dummy evaluation dataset containing all classes including Ransomware (2)
    torch.manual_seed(42)
    np.random.seed(42)

    val_samples = 100
    val_X = torch.randn(val_samples, input_dim)
    # Balanced evaluation slice for diagnostic verification
    val_y = torch.tensor([0] * 40 + [1] * 15 + [2] * 15 + [3] * 15 + [4] * 15, dtype=torch.long)

    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ThreatMLP(input_dim=input_dim, num_classes=num_classes)

    # 1. Confusion Matrix
    print("\n[1] Computing confusion matrix...")
    cm = confusion_matrix(model, val_loader, num_classes=num_classes)
    print("    Confusion Matrix (shape: [5, 5]):")
    print(cm)
    assert cm.shape == (5, 5), f"Expected shape (5, 5), got {cm.shape}"
    assert cm.sum() == val_samples, f"Expected {val_samples} samples, got {cm.sum()}"
    print("    SUCCESS: confusion_matrix returned valid 5x5 matrix.")

    # 2. Evaluate function
    print("\n[2] Computing clinical & threat metrics...")
    metrics = evaluate(model, val_loader, num_classes=num_classes)

    print(f"    Accuracy:          {metrics['accuracy']:.4f}")
    print(f"    Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"    Ransomware Recall: {metrics['ransomware_recall']:.4f}")
    print(f"    Ransomware FNR:    {metrics['ransomware_fnr']:.4f}")
    print(f"    Per-class Recall:  {metrics['per_class_recall']}")

    # Assert success criteria
    assert "macro_f1" in metrics, "metrics dict must contain 'macro_f1' key"
    assert "ransomware_recall" in metrics, "metrics dict must contain 'ransomware_recall' key"
    assert "ransomware_fnr" in metrics, "metrics dict must contain 'ransomware_fnr' key"
    assert 0.0 <= metrics["macro_f1"] <= 1.0, f"Invalid macro-F1: {metrics['macro_f1']}"
    assert 0.0 <= metrics["ransomware_recall"] <= 1.0, f"Invalid ransomware recall: {metrics['ransomware_recall']}"
    assert 0.0 <= metrics["ransomware_fnr"] <= 1.0, f"Invalid ransomware FNR: {metrics['ransomware_fnr']}"

    print("\n" + "=" * 80)
    print("METRICS MODULE VERIFICATION PASSED!")
    print("=" * 80)
