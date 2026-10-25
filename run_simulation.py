"""BioVault Federated Simulation Runner with Configurable Differential Privacy.

Orchestrates Flower federated learning simulations across non-IID hospital device fleets.
Supports both non-private baseline (FedAvg) and differentially private federated learning
(FedAvg + Opacus DP-SGD) targeting a configurable epsilon budget (e.g. ε = 3.0).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from client import BioVaultClient
from contracts import AttackClass, DPConfig, FederatedConfig
from data import engineer_features, get_client_loaders, load_ciomt, partition_non_iid, train_test_split
from model import ThreatMLP
from server import get_evaluate_fn


def run_federated_simulation(
    data_path: str = "data/CICIoMT2024",
    num_rounds: int = 10,
    num_clients: int = 5,
    alpha: float = 0.5,
    batch_size: int = 64,
    use_dp: bool = False,
    target_epsilon: float = 3.0,
    target_delta: float = 1e-5,
    results_path: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Flower simulation of BioVault with optional Opacus DP-SGD.

    Args:
        data_path: Path to CICIoMT2024 CSV dataset.
        num_rounds: Number of federated training rounds (default: 10).
        num_clients: Number of simulated hospital device fleets (default: 5).
        alpha: Dirichlet parameter for label skewness (default: 0.5).
        batch_size: Local DataLoader batch size (default: 64).
        use_dp: Whether to enable Opacus DP-SGD client training.
        target_epsilon: Target privacy budget epsilon (default: 3.0).
        target_delta: Target failure probability delta (default: 1e-5).
        results_path: Destination path for results CSV.
        seed: Random seed for deterministic reproducibility.

    Returns:
        pd.DataFrame: Logged per-round metrics DataFrame.
    """
    mode_str = f"DP-SGD (target epsilon={target_epsilon}, delta={target_delta})" if use_dp else "Non-Private Baseline"
    print("=" * 80)
    print(f"BioVault Federated Learning Simulation: {mode_str}")
    print("=" * 80)

    if results_path is None:
        results_path = os.path.join("results", "dp_fl.csv" if use_dp else "baseline_fl.csv")

    # 1. Ingest dataset
    print(f"\n[1] Loading CICIoMT2024 telemetry from '{data_path}'...")
    df_clean = load_ciomt(data_path)
    print(f"    Loaded {len(df_clean)} flows with {len(df_clean.columns)} columns.")

    # 2. Hold out random 15% as global test set (never seen by any client)
    print("\n[2] Splitting held-out global test set (15%) and client training pool (85%)...")
    train_pool_df, test_df = train_test_split(
        df_clean,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
        stratify=df_clean["Label"],
    )
    print(f"    Global Test Set:       {len(test_df)} samples")
    print(f"    Client Training Pool:  {len(train_pool_df)} samples")

    # 3. Partition client pool into 5 non-IID hospital device fleets
    print(f"\n[3] Partitioning client pool into {num_clients} non-IID fleets (Dirichlet alpha={alpha})...")
    partitions = partition_non_iid(
        train_pool_df,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        stratify_by_device=True,
    )

    # 4. Create client DataLoaders with local StandardScaler (zero global leakage)
    print("\n[4] Building local client DataLoaders (per-client local scaling)...")
    client_loaders: Dict[str, Tuple[DataLoader, DataLoader]] = {}
    input_dim = 0
    for i, p in enumerate(partitions):
        t_loader, v_loader = get_client_loaders(p, batch_size=batch_size, seed=seed)
        client_loaders[str(i)] = (t_loader, v_loader)
        if i == 0:
            input_dim = next(iter(t_loader))[0].shape[1]
        print(f"    Client {i} ({p['Device_Fleet'].iloc[0]}): Train {len(t_loader.dataset)} | Val {len(v_loader.dataset)}")

    # 5. Prepare global held-out test loader
    test_X, test_y, _ = engineer_features(test_df, fit_scaler=True)
    test_dataset = TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"    Global Test Loader: {len(test_loader.dataset)} samples across {len(test_loader)} batches")

    # 6. Initialize DP configuration if active
    dp_cfg = None
    if use_dp:
        dp_cfg = DPConfig(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=1.0,
            epochs=1,
            batch_size=batch_size,
        )

    # 7. Define Flower client creation factory
    def client_fn(cid: str) -> fl.client.Client:
        cid_str = str(getattr(cid, "node_id", cid)) if not isinstance(cid, str) else cid
        client_idx = str(int(cid_str) % num_clients) if cid_str.isdigit() else str(cid_str)

        train_ldr, val_ldr = client_loaders[client_idx]
        b_client = BioVaultClient(
            cid=client_idx,
            train_loader=train_ldr,
            val_loader=val_ldr,
            input_dim=input_dim,
            use_dp=use_dp,
            dp_config=dp_cfg,
        )
        return b_client.to_client()

    # 8. Initialize global model and FedAvg strategy
    torch.manual_seed(seed)
    np.random.seed(seed)
    initial_model = ThreatMLP(input_dim=input_dim, num_classes=5)
    if use_dp:
        from opacus.validators import ModuleValidator
        if not ModuleValidator.is_valid(initial_model):
            initial_model = ModuleValidator.fix(initial_model)

    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    )

    privacy_tracker: Dict[str, float] = {"epsilon_spent": 0.0}

    def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
        eps_list = [float(m["epsilon_spent"]) for _, m in fit_metrics if "epsilon_spent" in m]
        if eps_list:
            avg_eps = float(np.mean(eps_list))
            privacy_tracker["epsilon_spent"] = avg_eps
            return {"epsilon_spent": avg_eps}
        return {}

    eval_fn = get_evaluate_fn(
        test_loader=test_loader,
        input_dim=input_dim,
        num_classes=5,
        use_dp=use_dp,
        results_path=results_path,
        privacy_tracker=privacy_tracker if use_dp else None,
    )

    min_fit = max(1, min(3, num_clients))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.6,
        min_fit_clients=min_fit,
        min_available_clients=num_clients,
        evaluate_fn=eval_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn if use_dp else None,
        initial_parameters=initial_parameters,
    )

    # 9. Start Flower simulation
    print(f"\n[5] Launching Flower simulation ({num_rounds} rounds, DP={use_dp})...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # 10. Verify results CSV
    print(f"\n[6] Verifying logged results in '{results_path}'...")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    res_df = pd.read_csv(results_path)
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING SIMULATION METRICS:")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioVault Federated Learning Simulation")
    parser.add_argument("--dp", dest="use_dp", action="store_true", help="Enable Opacus DP-SGD client training")
    parser.add_argument("--no-dp", dest="use_dp", action="store_false", help="Disable DP (standard FedAvg baseline)")
    parser.set_defaults(use_dp=False)
    parser.add_argument("--epsilon", type=float, default=3.0, help="Target privacy budget epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target privacy failure probability delta")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of simulated hospital device fleets")
    parser.add_argument("--data-path", type=str, default=os.path.join("data", "CICIoMT2024"), help="Path to dataset directory or CSV")
    parser.add_argument("--results-path", type=str, default=None, help="Output results CSV path")

    args = parser.parse_args()

    df_metrics = run_federated_simulation(
        data_path=args.data_path,
        num_rounds=args.rounds,
        num_clients=args.clients,
        alpha=0.5,
        batch_size=64,
        use_dp=args.use_dp,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        results_path=args.results_path,
    )
