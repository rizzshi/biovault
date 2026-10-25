"""BioVault Differential Privacy (DP-SGD) Engine using Opacus.

Wraps PyTorch models, optimizers, and DataLoaders with formal (ε, δ)-differential privacy
guarantees. Targets a configurable privacy budget epsilon, derives calibrated Gaussian
noise multipliers automatically via Opacus, and tracks cumulative privacy loss across
training rounds using Rényi Differential Privacy (RDP).
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from contracts import DPConfig


def make_client_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    dp: DPConfig,
    accountant: str = "rdp",
) -> Tuple[nn.Module, Any, DataLoader, PrivacyEngine]:
    """Instrument client training components with Opacus DP-SGD guarantees.

    Enforces per-sample gradient clipping and Gaussian noise perturbation targeting
    the specified (target_epsilon, target_delta) privacy budget without hand-tuning sigma.

    Critical Implementation Detail:
        Opacus wraps the standard PyTorch DataLoader with a DPDataLoader (Poisson/uniform
        subsampling). The returned private_loader MUST be used during the training loop.

    Args:
        model: PyTorch neural network module (e.g., ThreatMLP).
        optimizer: Standard PyTorch optimizer (e.g., Adam, SGD).
        train_loader: Client local training DataLoader.
        dp: DPConfig specifying target_epsilon, target_delta, max_grad_norm, and epochs.
        accountant: Differential privacy accountant ('rdp' or 'prv', default: 'rdp').

    Returns:
        Tuple containing:
            - private_model: Opacus GradSampleModule with per-sample gradient computation.
            - private_optimizer: DPOptimizer applying gradient clipping and Gaussian noise.
            - private_loader: DPDataLoader enforcing privacy-safe batch sampling.
            - privacy_engine: Instantiated PrivacyEngine managing the privacy accountant.
    """
    # Fix any layers incompatible with DP-SGD (e.g. BatchNorm to GroupNorm)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        # Re-bind optimizer parameters if they referenced pre-fix BatchNorm modules
        opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
        mod_params = {id(p) for p in model.parameters()}
        if opt_params != mod_params:
            opt_cls = optimizer.__class__
            opt_defaults = {k: v for k, v in optimizer.defaults.items()}
            optimizer = opt_cls(model.parameters(), **opt_defaults)

    privacy_engine = PrivacyEngine(accountant=accountant)

    private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=dp.epochs,
        target_epsilon=dp.target_epsilon,
        target_delta=dp.target_delta,
        max_grad_norm=dp.max_grad_norm,
    )

    return private_model, private_optimizer, private_loader, privacy_engine


def get_privacy_spent(privacy_engine: PrivacyEngine, delta: float = 1e-5) -> float:
    """Extract cumulative privacy budget expenditure epsilon from the accountant.

    Args:
        privacy_engine: Active Opacus PrivacyEngine instance.
        delta: Target failure probability delta (default: 1e-5).

    Returns:
        float: Computed epsilon budget consumed so far.
    """
    return float(privacy_engine.get_epsilon(delta=delta))


if __name__ == "__main__":
    print("=" * 80)
    print("BioVault Differential Privacy Module Verification")
    print("=" * 80)

    from model import ThreatMLP
    from torch.utils.data import TensorDataset

    # Verify make_client_private on dummy data
    input_dim = 36
    num_classes = 5
    samples = 128
    batch_size = 32

    dummy_model = ThreatMLP(input_dim=input_dim, num_classes=num_classes)
    dummy_opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    dummy_loader = DataLoader(
        TensorDataset(torch.randn(samples, input_dim), torch.randint(0, num_classes, (samples,))),
        batch_size=batch_size,
        shuffle=True,
    )

    dp_cfg = DPConfig(
        target_epsilon=3.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        epochs=1,
        batch_size=batch_size,
    )

    p_model, p_opt, p_loader, pe = make_client_private(
        model=dummy_model,
        optimizer=dummy_opt,
        train_loader=dummy_loader,
        dp=dp_cfg,
    )

    print(f"\n[1] DP-SGD Instrumentation:")
    print(f"    Private Model Type:     {type(p_model)}")
    print(f"    Private Optimizer Type: {type(p_opt)}")
    print(f"    Derived Noise Scale sigma: {p_opt.noise_multiplier:.4f}")

    # Run 1 epoch of DP training
    criterion = nn.CrossEntropyLoss()
    p_model.train()
    for X_b, y_b in p_loader:
        p_opt.zero_grad()
        loss = criterion(p_model(X_b), y_b)
        loss.backward()
        p_opt.step()

    eps_spent = get_privacy_spent(pe, delta=dp_cfg.target_delta)
    print(f"\n[2] Privacy Expenditure Verification:")
    print(f"    Target Epsilon:  {dp_cfg.target_epsilon}")
    print(f"    Target Delta:    {dp_cfg.target_delta}")
    print(f"    Epsilon Spent:   {eps_spent:.4f}")

    assert eps_spent <= dp_cfg.target_epsilon + 0.1, f"Epsilon spent ({eps_spent}) exceeds target ({dp_cfg.target_epsilon})"
    print("    SUCCESS: Privacy expenditure accounted and within target bound.")

    print("\n" + "=" * 80)
    print("DP MODULE VERIFICATION PASSED!")
    print("=" * 80)
