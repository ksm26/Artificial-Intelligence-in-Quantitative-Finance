"""
Portfolio evaluation metrics and statistical tests.
"""
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------
# Basic performance metrics
# ---------------------------

def annualized_return(returns, freq=252):
    return np.mean(returns) * freq

def annualized_volatility(returns, freq=252):
    return np.std(returns, ddof=0) * np.sqrt(freq)

def sharpe_ratio(returns, risk_free=0.0, freq=252):
    excess = returns - risk_free / freq
    vol = annualized_volatility(excess, freq)
    return annualized_return(excess, freq) / vol if vol != 0 else np.nan

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def calmar_ratio(returns, freq=252):
    mdd = abs(max_drawdown(returns))
    return annualized_return(returns, freq) / mdd if mdd != 0 else np.nan

def sortino_ratio(returns, risk_free=0.0, freq=252):
    excess = returns - risk_free / freq
    downside = returns[returns < 0]
    downside_vol = np.std(downside, ddof=0) * np.sqrt(freq)
    return annualized_return(excess, freq) / downside_vol if downside_vol != 0 else np.nan

def turnover(weights):
    return weights.diff().abs().sum(axis=1).mean()

# ---------------------------
# Forecast accuracy metrics
# ---------------------------

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def directional_accuracy(y_true, y_pred):
    return ((np.sign(y_true) == np.sign(y_pred)).mean()).mean()

# ---------------------------
# Statistical tests
# ---------------------------

def paired_ttest(returns_a, returns_b):
    diff = returns_a - returns_b
    t_stat, p_val = stats.ttest_rel(returns_a, returns_b, nan_policy='omit')
    return t_stat, p_val

def jobson_korkie_memmel(returns_a, returns_b, freq=252):
    # Compute Sharpe ratios
    sr_a = sharpe_ratio(returns_a, freq=freq)
    sr_b = sharpe_ratio(returns_b, freq=freq)
    n = len(returns_a)
    var_a = np.var(returns_a, ddof=1)
    var_b = np.var(returns_b, ddof=1)
    cov_ab = np.cov(returns_a, returns_b, ddof=1)[0, 1]
    # Memmel correction
    diff_sr = sr_a - sr_b
    num = diff_sr
    denom = np.sqrt((var_a + var_b - 2 * cov_ab) / n)
    z = num / denom if denom != 0 else np.nan
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val

def bootstrap_sharpe_diff(returns_a, returns_b, n_boot=1000, freq=252, random_state=None):
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(returns_a), len(returns_a))
        sr_a = sharpe_ratio(returns_a.iloc[idx], freq=freq)
        sr_b = sharpe_ratio(returns_b.iloc[idx], freq=freq)
        diffs.append(sr_a - sr_b)
    diffs = np.array(diffs)
    p_val = (np.sum(diffs <= 0) / n_boot, np.sum(diffs >= 0) / n_boot)
    return np.mean(diffs), p_val

# ---------------------------
# Aggregate all metrics into a dict
# ---------------------------

def all_metrics(returns, weights=None, freq=252):
    metrics = {
        'ann_return': annualized_return(returns, freq),
        'ann_vol': annualized_volatility(returns, freq),
        'sharpe': sharpe_ratio(returns, freq=freq),
        'max_dd': max_drawdown(returns),
        'calmar': calmar_ratio(returns, freq),
        'sortino': sortino_ratio(returns, freq=freq)
    }
    if weights is not None:
        metrics['turnover'] = turnover(weights)
    return metrics

if __name__ == "__main__":
    # Quick test
    rets_a = pd.Series(np.random.normal(0.001, 0.02, 252))
    rets_b = pd.Series(np.random.normal(0.0008, 0.02, 252))
    print(all_metrics(rets_a))
    print("Paired t-test:", paired_ttest(rets_a, rets_b))
    print("Jobson-Korkie-Memmel:", jobson_korkie_memmel(rets_a, rets_b))
    print("Bootstrap Sharpe diff:", bootstrap_sharpe_diff(rets_a, rets_b))
