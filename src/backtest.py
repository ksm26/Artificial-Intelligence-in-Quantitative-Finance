"""
Backtesting engine: applies daily weights to realized returns and computes performance metrics.
"""
import pandas as pd
import numpy as np
from src.optimizer import mean_variance_optimizer

def backtest(weights, realized_returns, tc_bps=10):
    weights = weights.reindex(realized_returns.index).fillna(0)
    portfolio_returns = (weights.shift(1) * realized_returns).sum(axis=1)
    # transaction costs
    turnover = (weights.diff().abs()).sum(axis=1)
    tc = turnover * (tc_bps / 10000)
    portfolio_returns -= tc
    return portfolio_returns

def performance_metrics(returns, freq=252):
    ann_return = returns.mean() * freq
    ann_vol = returns.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    cum_return = (1 + returns).prod() - 1
    mdd = ((returns.cumsum() - returns.cumsum().cummax())).min()
    return dict(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        cum_return=cum_return,
        max_drawdown=mdd
    )

if __name__ == "__main__":
    # Example usage
    preds = pd.read_csv('results/predictions_lstm.csv', index_col=0, parse_dates=True)
    rets = pd.read_csv('data/processed/targets.csv', index_col=0, parse_dates=True)
    weights = mean_variance_optimizer(preds, rets)
    port_rets = backtest(weights, rets)
    metrics = performance_metrics(port_rets)
    print(metrics)
