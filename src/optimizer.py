"""
Portfolio optimizer using Mean-Variance Optimization (MVO).
- Takes return forecasts and historical covariances.
- Outputs daily portfolio weights.
"""
import numpy as np
import pandas as pd

def mean_variance_optimizer(pred_returns, hist_returns, risk_aversion=1.0):
    weights = {}
    for date in pred_returns.index:
        mu = pred_returns.loc[date].values  # forecasted mean returns
        cov = hist_returns.loc[:date].cov().values  # sample covariance up to current date
        try:
            inv_cov = np.linalg.pinv(cov)
            w = inv_cov @ mu
            w /= w.sum()
        except np.linalg.LinAlgError:
            w = np.ones_like(mu) / len(mu)
        weights[date] = w
    return pd.DataFrame(weights, index=pred_returns.columns).T
