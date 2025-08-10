"""
src/train.py

Combined implementation for Step 3: model definitions (LSTM and Transformer)
and a training harness to train on the preprocessed dataset created in Step 2.

Usage examples (from repo root):

# Train LSTM
python src/train.py --model lstm --epochs 100 --batch-size 32 --lookback 60

# Train Transformer
python src/train.py --model transformer --epochs 100 --batch-size 32 --lookback 60

Outputs:
- best model checkpoint saved to results/checkpoints/<model>_best.pth
- training logs printed to stdout

"""

import os
import argparse
from pathlib import Path
import json
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import local dataset
from datasets import SequenceDataset

# --------------------------- Utilities ---------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------- Models -----------------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_features, n_outputs, hidden_size=50, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (hn, cn) = self.lstm(x)  # out: (batch, seq_len, hidden)
        # use last time-step hidden state
        last = out[:, -1, :]  # (batch, hidden)
        last = self.dropout(last)
        out = self.fc(last)  # (batch, n_outputs)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, n_features, n_outputs, d_model=64, n_heads=8, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # will use after permute to (batch, d_model, seq_len)
        self.out = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (batch, seq_len, d_model)
        # pool across time dimension
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        out = self.out(x)  # (batch, n_outputs)
        return out


# --------------------------- Training harness -------------------------
class Trainer:
    def __init__(self, model, device, optimizer, criterion, ckpt_dir: Path, patience: int = 10):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.ckpt_dir = ckpt_dir
        self.patience = patience
        self.best_val = float('inf')
        self.no_improve = 0
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, train_loader, val_loader=None, epochs: int = 100):
        self.model.to(self.device)
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            train_losses = []
            for xb, yb, _ in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)

            t1 = time.time()
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss if val_loss is None else f'{val_loss:.6f}'} | time={t1-t0:.1f}s")

            # save checkpoint every epoch with epoch number
            ckpt_path = self.ckpt_dir / f"epoch_{epoch:03d}_valloss_{val_loss:.6f}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)

            # early stopping check
            if val_loader is not None:
                if val_loss < self.best_val - 1e-8:
                    self.best_val = val_loss
                    self.no_improve = 0
                    print(f"  New best val_loss={val_loss:.6f}")
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.patience:
                        print(f"Early stopping triggered (no improvement for {self.patience} epochs). Best val={self.best_val:.6f}")
                        break

    def evaluate(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else None


# --------------------------- Main -------------------------------------

def build_model(model_name: str, n_features: int, n_outputs: int, args):
    if model_name.lower() == 'lstm':
        return LSTMModel(n_features=n_features, n_outputs=n_outputs,
                         hidden_size=args.hidden_size, n_layers=args.n_layers, dropout=args.dropout)
    elif model_name.lower() == 'transformer':
        return TransformerModel(n_features=n_features, n_outputs=n_outputs,
                                d_model=args.d_model, n_heads=args.n_heads,
                                num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                                dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='transformer', choices=['lstm', 'transformer'])
    p.add_argument('--features', type=str, default='data/processed/features_scaled.csv')
    p.add_argument('--targets', type=str, default='data/processed/targets.csv')
    p.add_argument('--split', type=str, default='train')
    p.add_argument('--val-split', type=str, default='val')
    p.add_argument('--lookback', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    # model hyperparams
    p.add_argument('--hidden-size', type=int, default=50)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dim-feedforward', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--ckpt-dir', type=str, default='results/checkpoints')

    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # load dataset
    train_ds = SequenceDataset(args.features, args.targets, lookback=args.lookback, split='train')
    val_ds = SequenceDataset(args.features, args.targets, lookback=args.lookback, split='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    n_features = train_ds.n_features
    n_outputs = train_ds.n_targets

    print(f"n_features={n_features}, n_outputs={n_outputs}, train_samples={len(train_ds)}, val_samples={len(val_ds)}")

    model = build_model(args.model, n_features, n_outputs, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # timestamped subfolder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = Path(args.ckpt_dir) / args.model / timestamp
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, device, optimizer, criterion, run_ckpt_dir, patience=args.patience)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

    print('Training finished. Checkpoints saved to:', run_ckpt_dir)


if __name__ == '__main__':
    main()
