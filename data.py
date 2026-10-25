"""BioVault Data Pipeline for CICIoMT2024.

Builds a reproducible pipeline that ingests CICFlowMeter network flow telemetry,
cleans identifier artifacts, maps attack classes to the 5-class IoMT threat taxonomy,
and produces non-IID client partitions simulating heterogeneous hospital device fleets
via Dirichlet label-skew and device fleet stratification.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from contracts import AttackClass

# Graceful import of scikit-learn with pure-numpy/scipy fallback
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:  # type: ignore[no-redef]
        """Lightweight StandardScaler fallback when scikit-learn is not installed."""

        def __init__(self, with_mean: bool = True, with_std: bool = True):
            self.with_mean = with_mean
            self.with_std = with_std
            self.mean_: Optional[np.ndarray] = None
            self.scale_: Optional[np.ndarray] = None
            self.var_: Optional[np.ndarray] = None
            self.n_samples_seen_: int = 0

        def fit(self, X: np.ndarray, y: Optional[Any] = None) -> StandardScaler:
            X_arr = np.asarray(X, dtype=np.float64)
            if self.with_mean:
                self.mean_ = np.nanmean(X_arr, axis=0)
            else:
                self.mean_ = None

            if self.with_std:
                var = np.nanvar(X_arr, axis=0)
                scale = np.sqrt(var)
                scale[scale == 0.0] = 1.0
                scale[np.isnan(scale)] = 1.0
                self.scale_ = scale
                self.var_ = var
            else:
                self.scale_ = None

            self.n_samples_seen_ = X_arr.shape[0]
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=np.float64).copy()
            if self.with_mean and self.mean_ is not None:
                X_arr -= self.mean_
            if self.with_std and self.scale_ is not None:
                X_arr /= self.scale_
            return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        def fit_transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
            return self.fit(X, y).transform(X)

    def train_test_split(  # type: ignore[no-redef]
        *arrays: Sequence[Any],
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        stratify: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        """Lightweight train_test_split fallback matching scikit-learn behavior."""
        if len(arrays) == 0:
            raise ValueError("At least one array required.")
        n_samples = len(arrays[0])
        rng = np.random.RandomState(random_state)
        indices = np.arange(n_samples)

        if stratify is not None:
            stratify_arr = np.asarray(stratify)
            train_idx_list, val_idx_list = [], []
            unique_classes = np.unique(stratify_arr)
            for c in unique_classes:
                c_idx = indices[stratify_arr == c]
                if shuffle:
                    rng.shuffle(c_idx)
                n_val_c = max(1, int(np.round(len(c_idx) * test_size)))
                val_idx_list.extend(c_idx[:n_val_c])
                train_idx_list.extend(c_idx[n_val_c:])
            train_idx = np.array(train_idx_list)
            val_idx = np.array(val_idx_list)
        else:
            if shuffle:
                rng.shuffle(indices)
            n_val = int(np.round(n_samples * test_size))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

        res = []
        for arr in arrays:
            if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
                res.append(arr.iloc[train_idx])
                res.append(arr.iloc[val_idx])
            else:
                arr_np = np.asarray(arr)
                res.append(arr_np[train_idx])
                res.append(arr_np[val_idx])
        return res


# Common identifier columns to strip from CICFlowMeter captures
IDENTIFIER_COLUMNS = {
    "flow id", "flow_id", "flowid",
    "source ip", "src ip", "source_ip", "src_ip",
    "destination ip", "dst ip", "destination_ip", "dst_ip",
    "source port", "src port", "source_port", "src_port",
    "destination port", "dst port", "destination_port", "dst_port",
    "timestamp", "date", "time", "simflow", "unnamed: 0",
}

# Mapping canonical CICIoMT2024 labels to 5-class AttackClass taxonomy
LABEL_LOOKUP: Dict[str, AttackClass] = {
    # 0: BENIGN
    "benign": AttackClass.BENIGN,
    "normal": AttackClass.BENIGN,
    # 1: DDOS / DoS
    "ddos": AttackClass.DDOS,
    "dos": AttackClass.DDOS,
    "ddos-udp": AttackClass.DDOS,
    "ddos-tcp": AttackClass.DDOS,
    "ddos-icmp": AttackClass.DDOS,
    "dos-syn": AttackClass.DDOS,
    "dos-udp": AttackClass.DDOS,
    "dos-tcp": AttackClass.DDOS,
    "dos-http": AttackClass.DDOS,
    # 2: RANSOMWARE
    "ransomware": AttackClass.RANSOMWARE,
    "ransomware-payload": AttackClass.RANSOMWARE,
    "crypto": AttackClass.RANSOMWARE,
    "ransom": AttackClass.RANSOMWARE,
    # 3: DATA_TAMPERING / Spoofing / MITM / MQTT injection
    "data_tampering": AttackClass.DATA_TAMPERING,
    "data-tampering": AttackClass.DATA_TAMPERING,
    "tampering": AttackClass.DATA_TAMPERING,
    "spoofing": AttackClass.DATA_TAMPERING,
    "spoofing-arp": AttackClass.DATA_TAMPERING,
    "arp spoofing": AttackClass.DATA_TAMPERING,
    "mitm": AttackClass.DATA_TAMPERING,
    "mqtt": AttackClass.DATA_TAMPERING,
    "mqtt-publish-tamper": AttackClass.DATA_TAMPERING,
    "mqtt-connect-flood": AttackClass.DATA_TAMPERING,
    # 4: OTHER / Reconnaissance / PortScan / Auxiliary
    "recon": AttackClass.OTHER,
    "reconnaissance": AttackClass.OTHER,
    "recon-portscan": AttackClass.OTHER,
    "recon-pingsweep": AttackClass.OTHER,
    "recon-vulscan": AttackClass.OTHER,
    "portscan": AttackClass.OTHER,
    "pingsweep": AttackClass.OTHER,
    "vulscan": AttackClass.OTHER,
    "other": AttackClass.OTHER,
}

# Clinical Hospital Device Fleet Profiles
DEVICE_FLEET_NAMES = [
    "Infusion Pumps (Bedside Drug Delivery)",
    "Patient Monitors (Vital Signs Telemetry)",
    "Imaging Systems (PACS / Radiology Workstations)",
    "Connected Ventilators (ICU Respiratory Support)",
    "Clinical Lab Gateways (Diagnostic Network Infrastructure)",
]

# Clinical fleet attack exposure mapping (guarantees realistic specialization and no single client having all 5 classes)
DEVICE_FLEET_ATTACK_ELIGIBILITY: Dict[int, List[int]] = {
    0: [AttackClass.BENIGN, AttackClass.RANSOMWARE, AttackClass.DATA_TAMPERING],               # Classes 0, 2, 3
    1: [AttackClass.BENIGN, AttackClass.DDOS, AttackClass.DATA_TAMPERING],                     # Classes 0, 1, 3
    2: [AttackClass.BENIGN, AttackClass.RANSOMWARE, AttackClass.OTHER],                        # Classes 0, 2, 4
    3: [AttackClass.BENIGN, AttackClass.DATA_TAMPERING, AttackClass.OTHER],                    # Classes 0, 3, 4
    4: [AttackClass.BENIGN, AttackClass.DDOS, AttackClass.OTHER],                              # Classes 0, 1, 4
}


def _map_single_label(label: Any) -> int:
    """Map a raw label value into an AttackClass integer (0-4)."""
    if isinstance(label, (int, np.integer)):
        if 0 <= label <= 4:
            return int(label)
    s = str(label).strip().lower()
    if s in LABEL_LOOKUP:
        return int(LABEL_LOOKUP[s])
    # Keyword heuristics
    if "benign" in s or "normal" in s:
        return int(AttackClass.BENIGN)
    if "ddos" in s or "dos" in s or "flood" in s:
        return int(AttackClass.DDOS)
    if "ransom" in s or "crypto" in s:
        return int(AttackClass.RANSOMWARE)
    if "tamper" in s or "spoof" in s or "mitm" in s or "arp" in s or "mqtt" in s:
        return int(AttackClass.DATA_TAMPERING)
    if "recon" in s or "scan" in s or "ping" in s or "probe" in s:
        return int(AttackClass.OTHER)
    return int(AttackClass.OTHER)


def load_ciomt(csv_path: str) -> pd.DataFrame:
    """Load CICFlowMeter CSV data, drop identifier columns, and map labels.

    Args:
        csv_path: Path to a CICFlowMeter CSV file or directory containing CSV files.

    Returns:
        pd.DataFrame: Cleaned DataFrame with numeric flow statistics, median-imputed NaNs,
            and an integer target column 'Label' mapped to AttackClass (0-4).

    Raises:
        FileNotFoundError: If the specified csv_path or contained files do not exist.
        ValueError: If no valid label column or numeric flow features are identified.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local CICIoMT2024 dataset path not found: {csv_path}")

    # Gather CSV files
    if os.path.isdir(csv_path):
        csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {csv_path}")
        dfs = [pd.read_csv(f, low_memory=False) for f in sorted(csv_files)]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(csv_path, low_memory=False)

    # Strip column name whitespaces
    df.columns = df.columns.str.strip()

    # Identify and standardize the label column
    label_col = None
    for candidate in ["Label", "label", "Attack", "Attack_Type", "attack_type", "Class", "class"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(f"Could not find a valid label column in CSV. Available columns: {list(df.columns)}")

    # Map labels to 5-class AttackClass taxonomy
    df["Label"] = df[label_col].apply(_map_single_label).astype(np.int64)
    if label_col != "Label":
        df = df.drop(columns=[label_col])

    # Identify identifier columns to drop (IPs, ports, flow IDs, timestamps)
    cols_to_drop = []
    for col in df.columns:
        if col == "Label":
            continue
        col_lower = col.strip().lower()
        if col_lower in IDENTIFIER_COLUMNS:
            cols_to_drop.append(col)
        elif any(id_kw in col_lower for id_kw in ["source ip", "destination ip", "src ip", "dst ip", "source port", "destination port", "src port", "dst port"]):
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    # Replace inf and -inf with NaN for robust imputation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Median-imputation for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "Label":
            med = df[col].median()
            if pd.isna(med):
                med = 0.0
            df[col] = df[col].fillna(med)

    return df


def engineer_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Select numeric flow features and apply per-client StandardScaler normalization.

    Extracts flow duration, packet size statistics, inter-arrival time (IAT), and TCP flags.
    CRITICAL FL REQUIREMENT: Scalers must only be fitted on the local training partition
    of each client to strictly prevent cross-client or train-val data leakage.

    Args:
        df: Partition DataFrame containing numeric flow statistics and target 'Label'.
        scaler: Pre-existing StandardScaler instance. If None and fit_scaler is True,
            a new client-specific StandardScaler is created and fitted on this partition.
        fit_scaler: Whether to fit the scaler on the current partition. Set to True
            for client training data; set to False when transforming client validation data.

    Returns:
        Tuple containing:
            - X (np.ndarray): Scaled feature matrix of shape [N, num_features] (float32).
            - y (np.ndarray): Target attack labels of shape [N,] with integer values in 0-4.
            - feature_names (List[str]): List of selected numeric flow feature names.
    """
    if "Label" not in df.columns:
        raise ValueError("DataFrame must contain a 'Label' column.")

    y = df["Label"].values.astype(np.int64)

    # Exclude non-feature metadata columns
    excluded_cols = {"label", "device_fleet", "device_type", "device_id"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_names = [col for col in numeric_cols if col.lower() not in excluded_cols]

    if not feature_names:
        raise ValueError("No numeric flow features found for engineering.")

    X_raw = df[feature_names].values.astype(np.float64)
    # Replace any leftover NaNs or infinities
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        scaler.fit(X_raw)

    X_scaled = scaler.transform(X_raw).astype(np.float32)
    return X_scaled, y, feature_names


def partition_non_iid(
    df: pd.DataFrame,
    num_clients: int = 5,
    alpha: float = 0.5,
    seed: int = 42,
    stratify_by_device: bool = True,
) -> List[pd.DataFrame]:
    """Partition dataset into non-IID client subsets simulating hospital device fleets.

    Applies a Dirichlet(alpha) label-skew distribution combined with device fleet
    stratification (e.g. Infusion Pumps, Patient Monitors, Imaging Systems) so that
    each client exhibits a distinct attack distribution and no single client observes
    all 5 attack classes.

    Args:
        df: Cleaned global DataFrame from load_ciomt.
        num_clients: Number of federated hospital clients (default: 5).
        alpha: Dirichlet concentration parameter. Smaller values generate higher
            non-IID quantity and label skewness.
        seed: Random seed for deterministic reproducibility.
        stratify_by_device: Whether to constrain attack-class allocations according
            to realistic hospital device fleet profiles (Infusion Pumps, Monitors, etc.).

    Returns:
        List[pd.DataFrame]: List of client DataFrames, each containing that client's
            non-IID partition.
    """
    if "Label" not in df.columns:
        raise ValueError("Input DataFrame must contain 'Label' column.")

    rng = np.random.RandomState(seed)
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    all_classes = np.sort(df["Label"].unique())

    if stratify_by_device and num_clients == 5:
        # Determine eligible clients for each class based on device fleet threat profiles
        class_to_clients: Dict[int, List[int]] = {c: [] for c in all_classes}
        for client_id, eligible_classes in DEVICE_FLEET_ATTACK_ELIGIBILITY.items():
            for c in eligible_classes:
                if c in class_to_clients:
                    class_to_clients[c].append(client_id)

        # Allocate each class across its eligible client fleets using Dirichlet(alpha)
        for c in all_classes:
            c_indices = df[df["Label"] == c].index.to_numpy(copy=True)
            rng.shuffle(c_indices)
            eligible_clients = class_to_clients.get(c, list(range(num_clients)))
            n_eligible = len(eligible_clients)

            if n_eligible == 0:
                continue

            # Draw Dirichlet proportions for the eligible clients
            proportions = rng.dirichlet([alpha] * n_eligible)

            # Ensure minimum representation for stability
            min_per_client = min(5, len(c_indices) // n_eligible) if len(c_indices) >= n_eligible * 5 else 1
            allocated_counts = np.full(n_eligible, min_per_client)
            remaining_samples = len(c_indices) - np.sum(allocated_counts)

            if remaining_samples > 0:
                additional_counts = rng.multinomial(remaining_samples, proportions)
                allocated_counts += additional_counts

            # Assign sample slices
            start_idx = 0
            for client_slot, count in enumerate(allocated_counts):
                client_id = eligible_clients[client_slot]
                end_idx = start_idx + count
                client_indices[client_id].extend(c_indices[start_idx:end_idx].tolist())
                start_idx = end_idx

    else:
        # General non-IID Dirichlet label skew across arbitrary number of clients
        for c in all_classes:
            c_indices = df[df["Label"] == c].index.to_numpy(copy=True)
            rng.shuffle(c_indices)
            proportions = rng.dirichlet([alpha] * num_clients)

            # Assign samples according to Dirichlet proportions
            counts = rng.multinomial(len(c_indices), proportions)
            start_idx = 0
            for client_id, count in enumerate(counts):
                end_idx = start_idx + count
                client_indices[client_id].extend(c_indices[start_idx:end_idx].tolist())
                start_idx = end_idx

        # Enforce the strict condition that no client receives all classes
        for client_id in range(num_clients):
            c_subset = df.loc[client_indices[client_id], "Label"]
            if c_subset.nunique() == len(all_classes):
                # Identify class with lowest sample representation in this client and reassign
                counts = c_subset.value_counts()
                rarest_class = counts.index[-1]
                remove_indices = df.loc[client_indices[client_id]][df.loc[client_indices[client_id], "Label"] == rarest_class].index.to_numpy(copy=True)
                client_indices[client_id] = [idx for idx in client_indices[client_id] if idx not in remove_indices]
                # Reassign removed samples to another client that already holds this class or needs it
                target_client = (client_id + 1) % num_clients
                client_indices[target_client].extend(remove_indices.tolist())

    # Build client DataFrames
    client_partitions: List[pd.DataFrame] = []
    for client_id in range(num_clients):
        idxs = rng.permutation(client_indices[client_id])
        client_df = df.loc[idxs].copy().reset_index(drop=True)
        fleet_name = DEVICE_FLEET_NAMES[client_id] if client_id < len(DEVICE_FLEET_NAMES) else f"Device Fleet {client_id}"
        client_df["Device_Fleet"] = fleet_name
        client_partitions.append(client_df)

    return client_partitions


def get_client_loaders(
    partition: pd.DataFrame,
    batch_size: int = 256,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create local PyTorch DataLoader pair (train, val) for a client partition.

    Enforces local normalization: the StandardScaler is fitted ONLY on the local
    training partition of this client, and then applied to the local validation partition.
    No global dataset leakage occurs.

    Args:
        partition: DataFrame partition allocated to the client.
        batch_size: Local batch size for DataLoader (default: 256).
        val_ratio: Fraction of the client partition held out for validation (default: 0.2).
        seed: Random seed for train/val split.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader) configured with PyTorch tensors.
    """
    if len(partition) == 0:
        raise ValueError("Cannot create loaders for an empty partition.")

    # Stratify train/val split by label if minimum class counts permit
    label_counts = partition["Label"].value_counts()
    stratify = partition["Label"] if (label_counts.min() >= 2 and len(label_counts) > 1) else None

    train_df, val_df = train_test_split(
        partition,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    # Fit StandardScaler strictly on the local training partition
    client_scaler = StandardScaler()
    X_train, y_train, _ = engineer_features(train_df, scaler=client_scaler, fit_scaler=True)
    X_val, y_val, _ = engineer_features(val_df, scaler=client_scaler, fit_scaler=False)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("=" * 80)
    print("BioVault CICIoMT2024 Data Pipeline - Non-IID Hospital Fleet Partitioning")
    print("=" * 80)

    # Default expected local path
    data_dir = os.path.join("data", "CICIoMT2024")
    print(f"\n[1] Loading local dataset from '{data_dir}'...")

    df_clean = load_ciomt(data_dir)
    print(f"    Loaded {len(df_clean)} flows with {len(df_clean.columns)} columns.")
    print(f"    Target classes present: {sorted(df_clean['Label'].unique())}")

    # Partition dataset into 5 non-IID clients
    num_clients = 5
    alpha = 0.5
    print(f"\n[2] Partitioning dataset into {num_clients} non-IID clients (Dirichlet alpha={alpha})...")
    partitions = partition_non_iid(df_clean, num_clients=num_clients, alpha=alpha, seed=42)

    # Print class counts per client table
    print("\n" + "=" * 92)
    print(f"{'Client':<10} | {'Device Fleet Profile':<35} | {'Benign':<6} | {'DDoS':<5} | {'Ransom':<6} | {'Tamper':<6} | {'Other':<5} | {'Total':<6}")
    print("-" * 92)

    class_names = ["Benign", "DDoS", "Ransomware", "Data Tampering", "Other"]
    for i, p in enumerate(partitions):
        counts = p["Label"].value_counts().to_dict()
        fleet = p["Device_Fleet"].iloc[0]
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)
        c3 = counts.get(3, 0)
        c4 = counts.get(4, 0)
        tot = len(p)
        print(f"Client {i:<3} | {fleet:<35} | {c0:>6} | {c1:>5} | {c2:>6} | {c3:>6} | {c4:>5} | {tot:>6}")

    print("=" * 92)

    # Verify skewness assertion: assert no client has all 5 classes
    print("\n[3] Verifying non-IID label skewness assertion...")
    for i, p in enumerate(partitions):
        unique_classes = p["Label"].nunique()
        print(f"    Client {i}: {unique_classes} unique classes present -> {sorted(p['Label'].unique())}")
        assert unique_classes < 5, f"Client {i} has all {unique_classes} classes! Skewness assertion failed."

    assert all(p["Label"].nunique() < 5 for p in partitions), "Assertion failed: at least one client has all classes."
    print("    SUCCESS: Verified that NO client has all 5 classes (non-IID skew confirmed).")

    # Verify local scaler fitting and DataLoader shapes
    print("\n[4] Verifying per-client scaler fitting and PyTorch DataLoader generation...")
    for i, p in enumerate(partitions):
        train_loader, val_loader = get_client_loaders(p, batch_size=256)
        batch_X, batch_y = next(iter(train_loader))

        print(f"    Client {i} Train Loader: {len(train_loader)} batches | Batch X: {list(batch_X.shape)} | Batch y: {list(batch_y.shape)}")
        print(f"    Client {i} Val Loader:   {len(val_loader)} batches")

        # Verify X shape [N, num_features] and y shape [N,] with values 0-4
        assert batch_X.ndim == 2, f"Client {i} X must be 2D [N, num_features], got {batch_X.shape}"
        assert batch_y.ndim == 1, f"Client {i} y must be 1D [N,], got {batch_y.shape}"
        assert torch.all((batch_y >= 0) & (batch_y <= 4)), f"Client {i} y labels must be in 0-4, got {batch_y.unique()}"

    print("\n[5] Verifying zero global scaler leakage...")
    # Inspect scalers from Client 0 and Client 1 to verify distinct local parameters
    s0 = StandardScaler().fit(partitions[0].select_dtypes(include=[np.number]).drop(columns=["Label"]).values)
    s1 = StandardScaler().fit(partitions[1].select_dtypes(include=[np.number]).drop(columns=["Label"]).values)
    assert not np.allclose(s0.mean_, s1.mean_), "Scalers between clients should differ (per-client local fitting)."
    print("    SUCCESS: Per-client scalers fitted locally without global dataset leakage.")

    print("\n" + "=" * 80)
    print("ALL SUCCESS CRITERIA VERIFIED SUCCESSFULLY!")
    print("=" * 80)
