"""
Preprocessing pipeline: load processed prices (USD), compute features, scale on training
set only, save scaled features and targets for modeling.

Outputs (saved under data/processed/):
 - features_scaled.csv      : features (rows = dates, cols = flattened asset__feature)
 - targets.csv              : next-day log returns for each asset (columns = '<asset>__ret')
 - scaler_features.pkl      : fitted sklearn StandardScaler for features
 - index_splits.json        : dictionary of train/val/test date ranges for reproducibility

Run example:
    python src/preprocessing.py --data data/processed/prices_usd.csv

"""

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from features import build_features, log_returns, drop_start_nan_rows


DATA_DIR = Path("./data/processed")
ROOT = Path(".")


def fit_and_save_scaler(X_train: pd.DataFrame, out_path: Path) -> StandardScaler:
    scaler = StandardScaler()
    X_train = X_train.dropna()
    scaler.fit(X_train.values)
    joblib.dump(scaler, out_path)
    return scaler


def main(prices_csv: str,
         train_end: str = "2017-12-31",
         val_frac: float = 0.1,
         lookback: int = 60):
    prices_csv = Path(prices_csv)
    assert prices_csv.exists(), f"prices file not found: {prices_csv}"

    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    prices = prices.sort_index()

    # Build features
    features = build_features(prices)

    # drop initial rows where rolling ops create NaNs
    features = drop_start_nan_rows(features)

    # Targets: next-day log returns for each asset (use existing ret columns)
    # We assume that for each asset we have a column like '<asset>__ret'
    ret_cols = [c for c in features.columns if c.endswith('__ret')]
    if not ret_cols:
        raise RuntimeError("No return columns found in features (expected columns ending with '__ret')")

    # targets: shift -1 to get next-day return aligned with features at day t
    targets = features[ret_cols].shift(-1).dropna(how='all')

    # align features to targets (drop last row(s) in features without target)
    features = features.loc[targets.index]

    # split indices
    dates = features.index
    train_mask = dates <= pd.to_datetime(train_end)
    train_dates = dates[train_mask]
    val_dates = train_dates[int(len(train_dates)*(1-val_frac)):]  # last val_frac of train as validation
    train_dates = train_dates[:int(len(train_dates)*(1-val_frac))]
    test_mask = dates > pd.to_datetime(train_end)
    test_dates = dates[test_mask]

    splits = dict(
        train=[str(train_dates[0]), str(train_dates[-1])],
        val=[str(val_dates[0]), str(val_dates[-1])],
        test=[str(test_dates[0]), str(test_dates[-1])]
    )

    # Fit scaler on training features only
    X_train = features.loc[train_dates]
    scaler_path = DATA_DIR / "scaler_features.pkl"
    scaler = fit_and_save_scaler(X_train, scaler_path)

    # Transform full feature set
    # X_scaled = pd.DataFrame(scaler.transform(features.values), index=features.index, columns=features.columns)
    X_scaled = pd.DataFrame(scaled_array, index=X.index, columns=X.columns)
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).dropna()


    # Save outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_scaled.to_csv(DATA_DIR / "features_scaled.csv")
    targets.to_csv(DATA_DIR / "targets.csv")
    with open(DATA_DIR / "index_splits.json", "w") as fh:
        json.dump(splits, fh, indent=2)

    print("Saved features_scaled.csv, targets.csv, scaler_features.pkl, index_splits.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR / "prices_usd.csv"), help="input prices CSV (USD-converted)")
    p.add_argument("--train-end", default="2017-12-31")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--lookback", type=int, default=60)
    args = p.parse_args()
    main(args.data, args.train_end, args.val_frac, args.lookback)

