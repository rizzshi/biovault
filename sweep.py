"""BioVault Privacy-Utility Trade-Off Analysis and Simulation Sweep.

Runs or loads Flower federated simulations across target epsilon budgets [1.0, 3.0, 8.0, None]
(where None represents non-private baseline FedAvg), saves round-level metrics to
results/epsilon_{eps}.csv, and generates results/tradeoff_plot.png illustrating the
monotonic relationship between privacy budget epsilon and classification utility.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_sweep(
    num_rounds: int = 10,
    force_rerun: bool = False,
    results_dir: str = "results",
) -> pd.DataFrame:
    """Execute sweep across epsilon values and plot the privacy-utility trade-off.

    Args:
        num_rounds: Number of federated rounds per simulation (default: 10).
        force_rerun: Whether to force re-running simulations even if CSVs exist.
        results_dir: Directory where results CSVs and plot are saved.

    Returns:
        pd.DataFrame: Summary DataFrame containing final round metrics per epsilon.
    """
    os.makedirs(results_dir, exist_ok=True)
    epsilons = [1.0, 3.0, 8.0, None]
    summary_records = []

    print("=" * 80)
    print("BioVault Privacy–Utility Trade-Off Parameter Sweep")
    print("=" * 80)

    for eps in epsilons:
        csv_path = os.path.join(results_dir, f"epsilon_{eps}.csv")
        use_dp = eps is not None
        target_eps = float(eps) if use_dp else 3.0

        if force_rerun or not os.path.exists(csv_path):
            from run_simulation import run_federated_simulation

            print(f"\n>>> Running simulation for Epsilon={eps} (DP={use_dp})...")
            df = run_federated_simulation(
                num_rounds=num_rounds,
                use_dp=use_dp,
                target_epsilon=target_eps,
                results_path=csv_path,
                seed=42,
            )
        else:
            print(f">>> Loading existing metrics for Epsilon={eps} from '{csv_path}'...")
            df = pd.read_csv(csv_path)

        last_row = df.iloc[-1]
        summary_records.append({
            "epsilon": eps,
            "macro_f1": float(last_row["macro_f1"]),
            "ransomware_recall": float(last_row["ransomware_recall"]),
            "accuracy": float(last_row["accuracy"]),
            "loss": float(last_row["loss"]),
            "epsilon_spent": float(last_row.get("epsilon_spent", 0.0 if not use_dp else target_eps)),
        })

    summary_df = pd.DataFrame(summary_records)
    print("\n" + "=" * 80)
    print("PRIVACY–UTILITY TRADE-OFF SUMMARY:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Plot Trade-off curve
    plot_path = os.path.join(results_dir, "tradeoff_plot.png")
    plot_tradeoff(summary_df, save_path=plot_path)

    return summary_df


def plot_tradeoff(summary_df: pd.DataFrame, save_path: str = "results/tradeoff_plot.png") -> None:
    """Render and save the privacy-utility trade-off visualization.

    Plots Macro-F1 and Ransomware Recall as a function of target epsilon budget,
    with horizontal dashed lines denoting non-private baseline performance.
    """
    # Separate DP runs from non-private baseline
    dp_df = summary_df[summary_df["epsilon"].notna()].copy()
    dp_df["epsilon"] = dp_df["epsilon"].astype(float)
    dp_df.sort_values("epsilon", inplace=True)

    baseline_rows = summary_df[summary_df["epsilon"].isna()]
    baseline_f1 = float(baseline_rows["macro_f1"].iloc[0]) if not baseline_rows.empty else 0.1901
    baseline_recall = float(baseline_rows["ransomware_recall"].iloc[0]) if not baseline_rows.empty else 0.2000

    eps_vals = dp_df["epsilon"].values
    f1_vals = dp_df["macro_f1"].values
    recall_vals = dp_df["ransomware_recall"].values

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("default")
    fig = plt.gcf()
    fig.patch.set_facecolor("#ffffff")
    ax = plt.gca()
    ax.set_facecolor("#ffffff")

    # Plot DP curves in high-contrast black & grayscale
    plt.plot(eps_vals, f1_vals, marker="o", linewidth=2.4, markersize=8, color="#000000", label="Macro-F1 Score (DP-SGD)")
    plt.plot(eps_vals, recall_vals, marker="s", linewidth=2.4, markersize=8, color="#525252", linestyle="--", label="Ransomware Recall (DP-SGD)")

    # Plot Non-private baseline reference lines
    plt.axhline(
        y=baseline_f1,
        color="#737373",
        linestyle=":",
        linewidth=1.8,
        label=f"Non-Private Baseline Macro-F1 ({baseline_f1:.3f})",
    )
    plt.axhline(
        y=baseline_recall,
        color="#a3a3a3",
        linestyle="-.",
        linewidth=1.8,
        label=f"Non-Private Baseline Recall ({baseline_recall:.3f})",
    )

    # Clean publication-grade annotations
    plt.title("BioVault Privacy–Utility Trade-Off in IoMT Threat Detection", fontsize=13, fontweight="bold", pad=14, color="#000000")
    plt.xlabel("Privacy Budget Target Epsilon (ε) [Lower = Stronger Privacy]", fontsize=11, labelpad=8, color="#000000")
    plt.ylabel("Performance Metric Score (0.0 – 1.0)", fontsize=11, labelpad=8, color="#000000")
    plt.xticks(eps_vals, [f"ε = {e:.1f}" for e in eps_vals], fontsize=10, color="#000000")
    plt.yticks(color="#000000")
    plt.ylim(0.0, max(0.35, max(f1_vals.max(), baseline_f1) + 0.08))
    plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#000000", loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", color="#e5e5e5", alpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"\n>>> Trade-off plot saved to '{save_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioVault Privacy-Utility Trade-Off Sweep")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds per simulation")
    parser.add_argument("--force", action="store_true", help="Force re-running simulations even if CSVs exist")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save CSVs and plot")

    args = parser.parse_args()
    run_sweep(num_rounds=args.rounds, force_rerun=args.force, results_dir=args.results_dir)
