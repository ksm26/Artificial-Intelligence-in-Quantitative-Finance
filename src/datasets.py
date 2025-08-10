"""
PyTorch Dataset wrapper that provides rolling-window sequences (X) and multi-output targets (y).

This file expects the preprocessed CSVs created by src/preprocessing.py:
 - data/processed/features_scaled.csv
 - data/processed/targets.csv

Each sample is a sequence of `lookback` rows from features_scaled (shape: lookback x n_features)
and a target vector (n_assets) containing the next-day returns.

Usage example:
    ds = SequenceDataset("data/processed/features_scaled.csv", "data/processed/targets.csv", lookback=60, split='train')
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json


class SequenceDataset(Dataset):
    def __init__(self, features_csv: str, targets_csv: str, lookback: int = 60, split: str = 'train', index_splits: str = None):
        self.features_csv = Path(features_csv)
        self.targets_csv = Path(targets_csv)
        assert self.features_csv.exists(), f"features file not found: {self.features_csv}"
        assert self.targets_csv.exists(), f"targets file not found: {self.targets_csv}"

        self.features = pd.read_csv(self.features_csv, index_col=0, parse_dates=True).sort_index()
        self.targets = pd.read_csv(self.targets_csv, index_col=0, parse_dates=True).sort_index()

        # read splits if provided or available next to features
        self.lookback = lookback
        if index_splits is None:
            splits_path = self.features_csv.parent / "index_splits.json"
        else:
            splits_path = Path(index_splits)
        if splits_path.exists():
            with open(splits_path, 'r') as fh:
                splits = json.load(fh)
            self.split = split
            start_str, end_str = splits[split]
            self.start_date = pd.to_datetime(start_str)
            self.end_date = pd.to_datetime(end_str)
        else:
            # fallback: use whole range
            self.start_date = self.features.index[0]
            self.end_date = self.features.index[-1]

        # Precompute sample start indices (end indices correspond to the row whose target we use)
        all_dates = self.features.index
        # valid indices are those where we have a full lookback window ending at idx and a target for that idx
        self.valid_dates = []
        for i in range(len(all_dates)):
            date = all_dates[i]
            if date < self.start_date or date > self.end_date:
                continue
            if i - self.lookback < 0:
                continue
            # ensure target for this date exists
            if date not in self.targets.index:
                continue
            self.valid_dates.append(date)

        if len(self.valid_dates) == 0:
            raise RuntimeError("No valid samples found for given split/lookback. Check splits and lookback length.")

        # determine feature / target shapes
        self.n_features = self.features.shape[1]
        self.n_targets = self.targets.shape[1]

    def __len__(self):
        return len(self.valid_dates)

    def __getitem__(self, idx):
        date = self.valid_dates[idx]
        # find integer location
        i = self.features.index.get_loc(date)
        x = self.features.iloc[i - self.lookback:i].values.astype(np.float32)  # (lookback, n_features)
        y = self.targets.loc[date].values.astype(np.float32)  # (n_targets,)
        # convert to torch tensors, return (seq_len, n_features) and (n_targets,)
        return torch.from_numpy(x), torch.from_numpy(y), str(date)


if __name__ == "__main__":
    # quick sanity check when executed directly
    ds = SequenceDataset("data/processed/features_scaled.csv", "data/processed/targets.csv", lookback=60, split='train')
    print("Dataset length:", len(ds))
    x, y, d = ds[0]
    print("x.shape:", x.shape, "y.shape:", y.shape, "date:", d)
