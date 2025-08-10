"""
Generates rolling forecasts using a trained model.

- Loads scaler, features, targets, and model checkpoint.
- Performs walk-forward prediction for each day in the test set.
- Saves predictions to results/predictions_<model>.csv
"""
import argparse
from pathlib import Path
import torch
import pandas as pd
import joblib
import numpy as np
from datasets import SequenceDataset
from train import LSTMModel, TransformerModel

def rolling_predict(model, dataset, device, target_cols):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    preds, dates = [], []
    with torch.no_grad():
        for xb, _, dt in loader:
            xb = xb.to(device)
            yhat = model(xb).cpu().numpy().squeeze()
            preds.append(yhat)
            dates.append(dt[0])
    return pd.DataFrame(preds, index=dates, columns=target_cols)

def get_latest_checkpoint(model_type: str, ckpt_root: Path) -> Path:
    """
    Returns the latest timestamped checkpoint path for the given model_type (lstm or transformer).
    """
    model_dir = ckpt_root / model_type
    if not model_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory found for model type '{model_type}' in {model_dir}")

    # Find latest timestamp folder
    timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamped runs found in {model_dir}")

    latest_dir = max(timestamp_dirs, key=lambda d: d.stat().st_mtime)

    # Find latest epoch checkpoint in that folder
    ckpts = sorted(latest_dir.glob("*.pth"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {latest_dir}")

    return ckpts[0]  # most recent checkpoint file


def main(model_type, checkpoint, features_csv, targets_csv, lookback, split, out_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SequenceDataset(features_csv, targets_csv, lookback=lookback, split=split)
    n_features, n_outputs = ds.n_features, ds.n_targets

    if model_type == 'lstm':
        model = LSTMModel(n_features, n_outputs)
    elif model_type == 'transformer':
        model = TransformerModel(n_features, n_outputs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Automatically resolve checkpoint if not explicitly given
    if checkpoint is None or checkpoint.strip() == "":
        checkpoint = get_latest_checkpoint(model_type, Path("results/checkpoints"))
        print(f"[INFO] Using latest checkpoint: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    target_cols = pd.read_csv(targets_csv, index_col=0).columns.tolist()
    preds_df = rolling_predict(model, ds, device, target_cols)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_csv)
    print(f"Saved predictions to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model-type', default='transformer', choices=['lstm', 'transformer'])
    p.add_argument('--checkpoint', default='', help="Path to checkpoint file. If empty, uses latest run automatically.")
    p.add_argument('--features-csv', default='data/processed/features_scaled.csv')
    p.add_argument('--targets-csv', default='data/processed/targets.csv')
    p.add_argument('--lookback', type=int, default=60)
    p.add_argument('--split', default='test')
    p.add_argument('--out-csv', default='', help="Output predictions CSV path. If empty, defaults to results/predictions_<model>.csv")
    args = p.parse_args()

    if not args.out_csv:
        args.out_csv = f"results/predictions_{args.model_type}.csv"

    main(args.model_type, args.checkpoint, args.features_csv, args.targets_csv,
         args.lookback, args.split, args.out_csv)