"""
Feature engineering utilities for AI Quant Finance replication repo.

Functions here operate on pandas DataFrames of prices (columns = assets, index = DateTimeIndex)
and return DataFrames of features with flattened column names like '<ASSET>__<FEATURE>'.
"""

import pandas as pd
import numpy as np


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns for price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        index = dates, columns = asset tickers or short names.

    Returns
    -------
    pd.DataFrame
        same shape as prices, first row is NaN
    """
    return np.log(prices / prices.shift(1))


def rolling_momentum(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Return price percentage change over `window` days (simple momentum).
    """
    return prices.pct_change(window)


def rolling_mean_return(returns: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute rolling mean of returns (e.g., MA of returns).
    """
    return returns.rolling(window=window).mean()


def rolling_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Compute rolling volatility (std) of returns.
    """
    return returns.rolling(window=window).std()


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index for each column in prices DataFrame.

    Uses Wilder's smoothing (simple rolling mean here which is fine for daily data).
    Returns values in the [0,100] range.
    """
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # Use simple rolling mean for gains/losses
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()

    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(prices: pd.DataFrame,
                   windows: dict = None) -> pd.DataFrame:
    """Construct a feature DataFrame from raw price DataFrame.

    Output: DataFrame indexed by dates, columns flattened as '<asset>__<feature>'.

    Default features (if windows is None):
      - ret   : log return (1d)
      - ma5   : 5-day rolling mean of returns
      - ma21  : 21-day rolling mean of returns
      - vol21 : 21-day rolling std of returns
      - rsi14 : 14-day RSI computed on price
      - mom21 : 21-day simple price return (momentum)

    """
    if windows is None:
        windows = dict(ma_short=5, ma_long=21, vol=21, rsi=14, mom=21)

    prices = prices.sort_index()
    returns = log_returns(prices)

    parts = []
    for asset in prices.columns:
        s = prices[asset]
        r = returns[asset]

        col_dict = {
            f"{asset}__ret": r,
            f"{asset}__ma{windows['ma_short']}": r.rolling(windows['ma_short']).mean(),
            f"{asset}__ma{windows['ma_long']}": r.rolling(windows['ma_long']).mean(),
            f"{asset}__vol{windows['vol']}": r.rolling(windows['vol']).std(),
            f"{asset}__rsi{windows['rsi']}": rsi(s, window=windows['rsi']),
            f"{asset}__mom{windows['mom']}": s.pct_change(windows['mom'])
        }
        df_asset = pd.DataFrame(col_dict)
        parts.append(df_asset)

    # concat along columns (assets are side-by-side, each has its feature columns)
    features = pd.concat(parts, axis=1)
    # sort columns alphabetically for deterministic ordering
    features = features.reindex(sorted(features.columns), axis=1)
    return features


# Small helper to drop rows with too many missing values (at start of series)

def drop_start_nan_rows(df: pd.DataFrame, min_non_na: int = 1) -> pd.DataFrame:
    """Drop the beginning rows until at least one column has non-NA value.

    This is useful to remove the first `lookback` rows created by rolling operations.
    """
    # drop rows that are all NA
    df = df.dropna(how='all')
    return df

