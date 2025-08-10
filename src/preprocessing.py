"""
Preprocessing pipeline: load processed prices (USD), compute features, scale on training
set only, save scaled features and targets for modeling.
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
    # Drop any NaN rows before fitting
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    scaler = StandardScaler()
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

    # Targets: next-day log returns for each asset
    ret_cols = [c for c in features.columns if c.endswith('__ret')]
    if not ret_cols:
        raise RuntimeError("No return columns found in features (expected columns ending with '__ret')")

    targets = features[ret_cols].shift(-1)
    # Align features to where targets are available
    mask_valid_targets = ~targets.isna().any(axis=1)
    features = features.loc[mask_valid_targets]
    targets = targets.loc[mask_valid_targets]

    # Remove any inf/-inf
    features = features.replace([np.inf, -np.inf], np.nan)
    targets = targets.replace([np.inf, -np.inf], np.nan)

    # Drop any remaining NaNs
    features = features.dropna(how="any")
    targets = targets.dropna(how="any")

    # Align features and targets to the same dates
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    # Split indices
    dates = features.index
    train_mask = dates <= pd.to_datetime(train_end)
    train_dates = dates[train_mask]
    val_dates = train_dates[int(len(train_dates) * (1 - val_frac)):]
    train_dates = train_dates[:int(len(train_dates) * (1 - val_frac))]
    test_mask = dates > pd.to_datetime(train_end)
    test_dates = dates[test_mask]

    splits = dict(
        train=[str(train_dates[0]), str(train_dates[-1])],
        val=[str(val_dates[0]), str(val_dates[-1])],
        test=[str(test_dates[0]), str(test_dates[-1])]
    )

    # Fit scaler on training features only
    scaler_path = DATA_DIR / "scaler_features.pkl"
    scaler = fit_and_save_scaler(features.loc[train_dates], scaler_path)

    # Transform full feature set
    scaled_array = scaler.transform(features.values)
    X_scaled = pd.DataFrame(scaled_array, index=features.index, columns=features.columns)

    # Final cleanup to remove any NaN/inf after scaling
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    targets = targets.loc[X_scaled.index]  # realign after drop

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
