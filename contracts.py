"""BioVault Interface Contracts.

Locked interface specifications for privacy-preserving federated threat detection
on Internet of Medical Things (IoMT) networks with Differential Privacy (DP-SGD).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Tuple

import torch.nn as nn


@dataclass
class DPConfig:
    """Differential Privacy configuration specifying DP-SGD hyperparameters.

    Attributes:
        target_epsilon: Target privacy budget epsilon (ε). A lower ε provides
            stronger mathematical privacy guarantees against membership inference
            and reconstruction attacks, but requires adding larger calibrated Gaussian
            noise to clipped gradients, which reduces model classification accuracy
            and convergence speed (lower utility). A higher ε preserves more gradient
            signal fidelity (higher utility) at the cost of weaker privacy bounds.
        target_delta: Failure probability delta (δ). Binds the probability that the
            privacy loss exceeds ε. A smaller δ ensures stricter adherence to the
            privacy bound (higher privacy), but forces tighter noise calibration
            and consumes the privacy budget faster for a given number of training
            steps, trading off utility. Typically set to less than 1 / N (where N is
            the total training set size).
        max_grad_norm: Maximum L2 gradient clipping threshold. Clamps individual
            sample gradient vectors before batch aggregation to bound per-record
            sensitivity. A lower threshold strictly limits the influence of any single
            patient or device record (higher privacy), but introduces gradient clipping
            bias that can suppress rare IoMT threat indicators (lower utility). A higher
            threshold preserves unclipped directional updates (higher utility) but
            demands proportionally larger Gaussian noise injection.
        epochs: Local training epochs per federated round. Fewer local epochs (e.g., 1)
            limit local gradient steps and bound privacy budget consumption under
            Rényi Differential Privacy composition (higher privacy), but require more
            federated rounds to achieve model convergence. More local epochs improve
            local parameter optimization per round (higher utility) but accelerate
            privacy budget depletion.
        batch_size: Local batch size for subsampled gradient evaluation. Smaller
            batch sizes provide privacy amplification through Poisson subsampling
            (higher privacy per sample), but introduce higher gradient variance and
            destabilize training. Larger batches yield lower variance and more stable
            gradient estimates (higher utility) but reduce subsampling amplification,
            drawing down the privacy budget more rapidly per step.
    """

    target_epsilon: float = 3.0
    """Target privacy budget epsilon (ε). Trade-off: Smaller ε strictly bounds privacy leakage but injects higher noise that degrades detection utility."""

    target_delta: float = 1e-5
    """Target failure probability delta (δ). Trade-off: Lower δ tightens privacy guarantees but reduces allowable training steps or requires higher noise."""

    max_grad_norm: float = 1.0
    """Maximum L2 gradient clipping norm. Trade-off: Lower norm limits individual record sensitivity but introduces clipping bias that impairs model convergence."""

    epochs: int = 1
    """Local training epochs per round. Trade-off: Fewer epochs conserve cumulative privacy budget; more epochs optimize local weights but deplete budget faster."""

    batch_size: int = 256
    """Local training batch size. Trade-off: Smaller batches enhance subsampling privacy amplification; larger batches provide stabler gradient estimates."""


@dataclass
class FederatedConfig:
    """Federated Learning configuration for cross-silo healthcare orchestration.

    Attributes:
        num_rounds: Number of federated aggregation rounds. Fewer rounds limit total
            exposure of institutional client models and conserve the cumulative
            privacy budget (higher privacy), but risk model underfitting on diverse
            IoMT threat patterns (lower utility). More rounds allow the global model
            to converge across heterogeneous clinical environments (higher utility)
            at the expense of greater total privacy budget expenditure.
        num_clients: Total pool of registered healthcare institution/IoMT clients.
            A larger client pool enhances client-level subsampling privacy amplification
            and data diversity across medical centers (higher privacy and utility),
            though increasing communication orchestration overhead. A smaller client
            pool reduces coordination complexity but concentrates updates across fewer
            sites, increasing privacy exposure risk if updates are analyzed.
        fraction_fit: Fraction of available clients sampled per aggregation round.
            A lower fraction provides client-level subsampling privacy amplification
            by reducing how frequently any given hospital participates (higher privacy),
            but increases variance between global rounds (lower utility). A higher
            fraction samples more hospitals simultaneously for smoother aggregation
            (higher utility), but depletes client privacy budgets more quickly.
        min_fit_clients: Minimum number of clients required to participate in an
            aggregation round. Enforces k-anonymity-style aggregation protection by
            preventing single-client update isolation or server-side model inversion
            attacks (higher privacy). Setting this threshold too high risks stalled
            rounds if network latency spikes (lower utility); setting it too low
            risks exposing individual institutional updates.
    """

    num_rounds: int = 10
    """Number of federated aggregation rounds. Trade-off: Fewer rounds conserve privacy budget; more rounds improve global convergence across hospital silos."""

    num_clients: int = 5
    """Total pool of federated clients. Trade-off: Larger pools improve client subsampling privacy and threat diversity; smaller pools risk update memorization."""

    fraction_fit: float = 0.6
    """Fraction of clients sampled per round. Trade-off: Lower fractions amplify subsampling privacy; higher fractions yield stabler aggregation updates."""

    min_fit_clients: int = 3
    """Minimum clients required per round. Trade-off: Higher thresholds protect against single-client inversion attacks; lower thresholds prevent round dropping."""


class AttackClass(IntEnum):
    """5-class IoMT (Internet of Medical Things) threat taxonomy.

    Categorizes clinical device telemetry into baseline operations and
    critical medical cyber attack vectors.
    """

    BENIGN = 0
    """Baseline non-malicious medical device telemetry and hospital network traffic."""

    DDOS = 1
    """Distributed Denial of Service disrupting medical telemetry, monitoring, or infusion pumps."""

    RANSOMWARE = 2
    """Ransomware payload encrypting clinical storage, PACS archives, or device endpoints."""

    DATA_TAMPERING = 3
    """Integrity attack altering vital signs, EHR telemetry, or medical dosage commands."""

    OTHER = 4
    """Ancillary, zero-day, or unclassified IoMT network anomalies and intrusion attempts."""


def build_model(input_dim: int, num_classes: int = 5) -> nn.Module:
    """Construct the baseline neural network architecture for IoMT threat classification.

    Args:
        input_dim: Dimensionality of preprocessed IoMT telemetry feature vectors.
        num_classes: Number of target attack classes defined by AttackClass (default: 5).

    Returns:
        nn.Module: Initialized ThreatMLP neural network model.
    """
    from model import ThreatMLP

    return ThreatMLP(input_dim=input_dim, num_classes=num_classes)


def make_dp_private(
    model: nn.Module,
    optimizer: Any,
    train_loader: Any,
    dp: DPConfig,
) -> Tuple[nn.Module, Any, Any, Any]:
    """Wrap model, optimizer, and train_loader with differential privacy guarantees.

    Instruments PyTorch training components with Opacus PrivacyEngine to enforce
    gradient clipping and Gaussian noise perturbation per DPConfig.

    Args:
        model: PyTorch neural network module to privatize.
        optimizer: Standard PyTorch optimizer (e.g., Adam, SGD).
        train_loader: PyTorch DataLoader supplying local clinical batches.
        dp: DPConfig specifying epsilon, delta, max gradient norm, and training parameters.

    Returns:
        Tuple containing privatized (model, optimizer, train_loader, privacy_engine).
    """
    from dp import make_client_private

    return make_client_private(model=model, optimizer=optimizer, train_loader=train_loader, dp=dp)
