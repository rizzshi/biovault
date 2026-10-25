"""BioVault Federated Server & Central Evaluation.

Implements centralized server aggregation and evaluation routines using FedAvg.
Evaluates global model checkpoints across training rounds against an unpartitioned,
held-out global test set (15% of CICIoMT2024) and records metrics to:
  - results/baseline_fl.csv (when DP is disabled)
  - results/dp_fl.csv (when DP is enabled, including accountant epsilon_spent)
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import evaluate
from model import ThreatMLP


def get_evaluate_fn(
    test_loader: DataLoader,
    input_dim: int,
    num_classes: int = 5,
    use_dp: bool = False,
    results_path: Optional[str] = None,
    privacy_tracker: Optional[Dict[str, float]] = None,
) -> Callable[[int, fl.common.NDArrays, Dict[str, fl.common.Scalar]], Optional[Tuple[float, Dict[str, fl.common.Scalar]]]]:
    """Construct central evaluation function for FedAvg logging to results_path.

    Args:
        test_loader: DataLoader containing held-out global test set (15% of CICIoMT2024).
        input_dim: Flow statistical feature dimensionality.
        num_classes: Number of target attack classes (default: 5).
        use_dp: Whether Differential Privacy is active.
        results_path: Filepath where per-round metrics are logged. If None, defaults
            to results/dp_fl.csv (if use_dp) or results/baseline_fl.csv.
        privacy_tracker: Optional mutable dictionary tracking client privacy metrics.

    Returns:
        Callable: central_evaluate function conforming to Flower's evaluate_fn contract.
    """
    if results_path is None:
        results_path = os.path.join("results", "dp_fl.csv" if use_dp else "baseline_fl.csv")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Initialize results CSV with appropriate header
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        if use_dp:
            writer.writerow(["round", "loss", "accuracy", "macro_f1", "ransomware_recall", "ransomware_fnr", "epsilon_spent"])
        else:
            writer.writerow(["round", "loss", "accuracy", "macro_f1", "ransomware_recall", "ransomware_fnr"])

    eval_model = ThreatMLP(input_dim=input_dim, num_classes=num_classes)
    if use_dp:
        from opacus.validators import ModuleValidator
        if not ModuleValidator.is_valid(eval_model):
            eval_model = ModuleValidator.fix(eval_model)

    def central_evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Evaluate global parameters against the held-out global test set."""
        if len(parameters) == 0:
            return None

        # Load parameters into the global evaluation model
        state_dict = OrderedDict()
        for (k, current_val), p in zip(eval_model.state_dict().items(), parameters):
            state_dict[k] = torch.as_tensor(p, dtype=current_val.dtype)
        eval_model.load_state_dict(state_dict, strict=True)

        # Run clinical threat evaluation
        res = evaluate(eval_model, test_loader, num_classes=num_classes)

        loss = float(res["loss"])
        accuracy = float(res["accuracy"])
        macro_f1 = float(res["macro_f1"])
        ransomware_recall = float(res["ransomware_recall"])
        ransomware_fnr = float(res["ransomware_fnr"])

        # Retrieve privacy accountant expenditure if DP is active
        eps_spent = 0.0
        if use_dp:
            if server_round > 0 and privacy_tracker is not None:
                eps_spent = float(privacy_tracker.get("epsilon_spent", 0.0))
            elif server_round > 0 and "epsilon_spent" in config:
                eps_spent = float(config["epsilon_spent"])

        # Log per-round metrics to CSV
        with open(results_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if use_dp:
                writer.writerow([server_round, loss, accuracy, macro_f1, ransomware_recall, ransomware_fnr, eps_spent])
            else:
                writer.writerow([server_round, loss, accuracy, macro_f1, ransomware_recall, ransomware_fnr])

        # Save global checkpoint for demo inference
        checkpoint_path = os.path.join(os.path.dirname(results_path), "global_model.pt")
        torch.save(eval_model.state_dict(), checkpoint_path)

        dp_suffix = f" | Epsilon Spent: {eps_spent:.4f}" if use_dp else ""
        print(
            f"  [Server Round {server_round:2d}] "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"Ransomware Recall: {ransomware_recall:.4f} (FNR: {ransomware_fnr:.4f})"
            f"{dp_suffix}"
        )

        metrics_dict: Dict[str, fl.common.Scalar] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "ransomware_recall": ransomware_recall,
            "ransomware_fnr": ransomware_fnr,
        }
        if use_dp:
            metrics_dict["epsilon_spent"] = eps_spent

        return loss, metrics_dict

    return central_evaluate


def start_server(
    server_address: str = "0.0.0.0:8080",
    test_loader: Optional[DataLoader] = None,
    input_dim: int = 36,
    num_rounds: int = 10,
    use_dp: bool = False,
    results_path: Optional[str] = None,
) -> None:
    """Start standalone Flower server with FedAvg strategy.

    Configured with fraction_fit=0.6, min_fit_clients=3, min_available_clients=5.

    Args:
        server_address: Host and port to bind server.
        test_loader: Optional DataLoader for held-out global test set.
        input_dim: Dimensionality of features.
        num_rounds: Total federated aggregation rounds.
        use_dp: Whether Differential Privacy is active.
        results_path: Destination CSV path.
    """
    if results_path is None:
        results_path = os.path.join("results", "dp_fl.csv" if use_dp else "baseline_fl.csv")

    if test_loader is None and os.path.exists("data/CICIoMT2024"):
        try:
            from data import engineer_features, load_ciomt, train_test_split
            df_clean = load_ciomt("data/CICIoMT2024")
            _, test_df = train_test_split(df_clean, test_size=0.15, random_state=42, shuffle=True, stratify=df_clean["Label"])
            test_X, test_y, _ = engineer_features(test_df, fit_scaler=True)
            test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long())
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            input_dim = test_X.shape[1]
        except Exception:
            pass

    privacy_tracker: Dict[str, float] = {"epsilon_spent": 0.0}

    def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
        eps_list = [float(m["epsilon_spent"]) for _, m in fit_metrics if "epsilon_spent" in m]
        if eps_list:
            avg_eps = float(np.mean(eps_list))
            privacy_tracker["epsilon_spent"] = avg_eps
            return {"epsilon_spent": avg_eps}
        return {}

    eval_fn = None
    if test_loader is not None:
        eval_fn = get_evaluate_fn(
            test_loader=test_loader,
            input_dim=input_dim,
            use_dp=use_dp,
            results_path=results_path,
            privacy_tracker=privacy_tracker if use_dp else None,
        )

    initial_model = ThreatMLP(input_dim=input_dim, num_classes=5)
    if use_dp:
        from opacus.validators import ModuleValidator
        if not ModuleValidator.is_valid(initial_model):
            initial_model = ModuleValidator.fix(initial_model)

    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.6,
        min_fit_clients=3,
        min_available_clients=5,
        evaluate_fn=eval_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn if use_dp else None,
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioVault Federated Server")
    parser.add_argument("--dp", dest="use_dp", action="store_true", help="Enable Opacus Differential Privacy (DP-SGD)")
    parser.add_argument("--no-dp", dest="use_dp", action="store_false", help="Disable Differential Privacy (standard FedAvg)")
    parser.set_defaults(use_dp=False)
    parser.add_argument("--epsilon", type=float, default=3.0, help="Target privacy budget epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target privacy failure probability delta")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Server bind address")
    parser.add_argument("--results-path", type=str, default=None, help="Path to write results CSV")

    args = parser.parse_args()

    print(f"Starting BioVault Server (DP={args.use_dp}, Rounds={args.rounds}) on {args.address}...")
    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        use_dp=args.use_dp,
        results_path=args.results_path,
    )
