"""
Evaluate and plot portfolio performance for LSTM or Transformer predictions.

Usage:
    python src/evaluate_and_plot.py --model-type transformer
    python src/evaluate_and_plot.py --model-type lstm
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.optimizer import mean_variance_optimizer
from src.backtest import backtest
from src.metric import all_metrics, rmse, directional_accuracy, paired_ttest, jobson_korkie_memmel


def main(model_type: str):
    # Paths
    preds_path = project_root / 'results' / f'predictions_{model_type}.csv'
    targets_path = project_root / 'data' / 'processed' / 'targets.csv'
    fig_dir = project_root / 'results' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions & realized returns
    preds = pd.read_csv(preds_path, index_col=0, parse_dates=True)
    realized = pd.read_csv(targets_path, index_col=0, parse_dates=True)

    # Align dates & assets
    common_idx = preds.index.intersection(realized.index)
    preds = preds.loc[common_idx]
    realized = realized.loc[common_idx, preds.columns]

    # Portfolio optimization
    weights = mean_variance_optimizer(preds, realized)
    port_rets = backtest(weights, realized)

    # Performance metrics
    perf_metrics = all_metrics(port_rets, weights)
    print("Portfolio Performance Metrics:")
    for k, v in perf_metrics.items():
        print(f"{k}: {v:.4f}")

    # Forecast accuracy metrics
    forecast_rmse = rmse(realized, preds).mean()
    forecast_dir_acc = directional_accuracy(realized, preds).mean()
    print(f"Forecast RMSE: {forecast_rmse:.6f}")
    print(f"Directional Accuracy: {forecast_dir_acc:.2%}")

    # Benchmark (equal weights)
    eq_weights = pd.DataFrame(1/len(preds.columns), index=preds.index, columns=preds.columns)
    benchmark_rets = backtest(eq_weights, realized)

    # Statistical tests
    t_stat, p_val = paired_ttest(port_rets, benchmark_rets)
    print(f"Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")

    z_stat, p_val_z = jobson_korkie_memmel(port_rets, benchmark_rets)
    print(f"Jobson-Korkie-Memmel: z={z_stat:.4f}, p={p_val_z:.4f}")

    # Save metrics CSV
    metrics_csv_path = project_root / 'results' / f'metrics_{model_type}.csv'
    pd.DataFrame([perf_metrics]).to_csv(metrics_csv_path, index=False)

    # ================= PLOTS =================

    # Cumulative returns
    cum_rets = (1 + port_rets).cumprod()
    cum_bench = (1 + benchmark_rets).cumprod()
    plt.figure(figsize=(10,6))
    plt.plot(cum_rets, label=f"{model_type.upper()} Portfolio")
    plt.plot(cum_bench, label="Equal Weight Benchmark")
    plt.title(f"Cumulative Returns - {model_type.upper()}")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"cumulative_returns_{model_type}.png", dpi=300)
    plt.close()

    # Rolling Sharpe ratio
    rolling_sharpe = (port_rets.rolling(60).mean() / port_rets.rolling(60).std()) * np.sqrt(252)
    plt.figure(figsize=(10,6))
    plt.plot(rolling_sharpe, label="Rolling Sharpe (60d)")
    plt.title(f"Rolling Sharpe Ratio - {model_type.upper()}")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"rolling_sharpe_{model_type}.png", dpi=300)
    plt.close()

    # Portfolio weights heatmap
    plt.figure(figsize=(12,6))
    sns.heatmap(weights.T, cmap="viridis", cbar=True)
    plt.title(f"Portfolio Weights Over Time - {model_type.upper()}")
    plt.xlabel("Date")
    plt.ylabel("Asset")
    plt.tight_layout()
    plt.savefig(fig_dir / f"weights_heatmap_{model_type}.png", dpi=300)
    plt.close()

    print(f"Figures saved in {fig_dir}")
    print(f"Metrics saved in {metrics_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default = "transformer", choices=["lstm", "transformer"],
                        help="Model type to evaluate (lstm or transformer)")
    args = parser.parse_args()
    main(args.model_type)
