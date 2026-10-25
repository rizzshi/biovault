"""BioVault Federated Client for Flower Simulation.

Implements BioVaultClient conforming to Flower's NumPyClient interface.
Supports local ThreatMLP training and evaluation with inverse frequency
class weighting, with optional Opacus DP-SGD privacy instrumentation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from contracts import DPConfig
from metrics import evaluate
from model import ThreatMLP, compute_class_weights, train_local


class BioVaultClient(fl.client.NumPyClient):
    """Flower NumPyClient implementation for IoMT threat detection with optional DP-SGD."""

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int,
        class_weights: Optional[torch.Tensor] = None,
        use_dp: bool = False,
        dp_config: Optional[DPConfig] = None,
    ):
        """Initialize BioVaultClient with local dataset partitions.

        Args:
            cid: Client identifier string.
            train_loader: DataLoader providing client local training batches.
            val_loader: DataLoader providing client local validation batches.
            input_dim: Dimensionality of flow feature vectors.
            class_weights: Optional precomputed class weights. If None,
                computed from train_loader using inverse frequency.
            use_dp: Whether to enable Opacus Differential Privacy (DP-SGD).
            dp_config: Optional DPConfig hyperparameters (epsilon, delta, clipping norm).
        """
        self.cid = str(cid)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_dim = input_dim
        self.use_dp = use_dp
        self.dp_config = dp_config or DPConfig()

        # Initialize local ThreatMLP model
        self.model = ThreatMLP(input_dim=input_dim, num_classes=5)

        # If differential privacy is active, adapt layers (BatchNorm to GroupNorm)
        if self.use_dp:
            from opacus.validators import ModuleValidator
            if not ModuleValidator.is_valid(self.model):
                self.model = ModuleValidator.fix(self.model)

        # Precompute or assign class weights based on local fleet data
        if class_weights is None:
            self.class_weights = compute_class_weights(train_loader, num_classes=5)
        else:
            self.class_weights = class_weights

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays."""
        underlying = getattr(self.model, "_module", self.model)
        return [val.cpu().numpy() for _, val in underlying.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load global parameters into the local ThreatMLP model.

        Preserves tensor datatypes to ensure compatibility with normalization buffers.
        """
        underlying = getattr(self.model, "_module", self.model)
        state_dict = OrderedDict()
        for (k, current_val), p in zip(underlying.state_dict().items(), parameters):
            state_dict[k] = torch.as_tensor(p, dtype=current_val.dtype)
        underlying.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Fit model locally on this client's device fleet partition for 1 epoch.

        If use_dp is enabled, wraps local training with Opacus DP-SGD targeting
        the configured epsilon and logs epsilon_spent from the accountant.

        Args:
            parameters: Global model parameters received from server.
            config: Server training configuration dictionary.

        Returns:
            Tuple[List[np.ndarray], int, Dict[str, Any]]: Updated local parameters,
                number of training examples, and training metrics dict.
        """
        # 1. Update local model with incoming global parameters
        self.set_parameters(parameters)

        if self.use_dp:
            from dp import get_privacy_spent, make_client_private
            from opacus.validators import ModuleValidator

            dp = self.dp_config

            # 1. Build a fresh model and load incoming global parameters
            fresh_model = ThreatMLP(input_dim=self.input_dim, num_classes=5)
            if not ModuleValidator.is_valid(fresh_model):
                fresh_model = ModuleValidator.fix(fresh_model)

            state_dict = OrderedDict()
            for (k, current_val), p in zip(fresh_model.state_dict().items(), parameters):
                state_dict[k] = torch.as_tensor(p, dtype=current_val.dtype)
            fresh_model.load_state_dict(state_dict, strict=True)

            # 2. Call make_client_private
            lr = float(config.get("lr", 3e-4))
            optimizer = torch.optim.Adam(fresh_model.parameters(), lr=lr)
            p_model, p_optimizer, p_loader, privacy_engine = make_client_private(
                model=fresh_model,
                optimizer=optimizer,
                train_loader=self.train_loader,
                dp=dp,
            )

            # 3. Train with the private optimizer/loader
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            p_model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for epoch in range(dp.epochs):
                for X_batch, y_batch in p_loader:
                    p_optimizer.zero_grad()
                    logits = p_model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    p_optimizer.step()

                    b_size = y_batch.size(0)
                    total_loss += loss.item() * b_size
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += b_size

            train_loss = total_loss / max(1, total_samples)
            train_acc = total_correct / max(1, total_samples)

            # 4. Extract and return the private model's parameters (noisy gradients already applied)
            underlying = getattr(p_model, "_module", p_model)
            updated_params = [val.cpu().numpy() for _, val in underlying.state_dict().items()]
            self.set_parameters(updated_params)

            # 5. Add epsilon_spent to returned metrics dict
            eps_spent = get_privacy_spent(privacy_engine, delta=dp.target_delta)

            metrics = {
                "loss": float(train_loss),
                "accuracy": float(train_acc),
                "epsilon_spent": float(eps_spent),
            }
            num_examples = len(self.train_loader.dataset)
            return updated_params, num_examples, metrics

        else:
            # Standard non-private training
            self.model, train_loss, train_acc = train_local(
                model=self.model,
                train_loader=self.train_loader,
                epochs=1,
                lr=1e-3,
                class_weights=self.class_weights,
            )

            num_examples = len(self.train_loader.dataset)
            metrics = {
                "loss": float(train_loss),
                "accuracy": float(train_acc),
            }
            return self.get_parameters(config={}), num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate global model on this client's local validation partition.

        Args:
            parameters: Global model parameters received from server.
            config: Server evaluation configuration dictionary.

        Returns:
            Tuple[float, int, Dict[str, Any]]: Validation loss, number of validation
                examples, and clinical metrics dictionary (macro-F1, ransomware recall).
        """
        self.set_parameters(parameters)
        res = evaluate(self.model, self.val_loader, num_classes=5)

        num_examples = len(self.val_loader.dataset)
        metrics_dict = {
            "accuracy": float(res["accuracy"]),
            "macro_f1": float(res["macro_f1"]),
            "ransomware_recall": float(res["ransomware_recall"]),
            "ransomware_fnr": float(res["ransomware_fnr"]),
        }
        return float(res["loss"]), num_examples, metrics_dict
