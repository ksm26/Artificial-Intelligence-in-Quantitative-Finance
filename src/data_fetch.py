#!/usr/bin/env python3
"""src/data_fetch.py

Downloads market data (Yahoo Finance + optional FRED), converts non-USD indices to USD,
aligns to S&P500 calendar, saves raw and processed CSVs.

Usage:
    python src/data_fetch.py --start 2010-01-01 --end 2020-12-31
"""

import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
# optional: from pandas_datareader import data as pdr  # if you want FRED series

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Default tickers (change if you want)
TICKERS = {
    "^GSPC": "SP500",
    "^FTSE": "FTSE100",
    "^N225": "NIKKEI225",
    "EEM": "EEM",
    "GC=F": "GOLD_FUT",
    "IEF": "IEF"            # treasury ETF proxy
}
FX_TICKERS = ["GBPUSD=X", "USDJPY=X"]  # for converting FTSE (GBP) and N225 (JPY) to USD

def download_ticker(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        print(f"Warning: empty data for {ticker}")
        return None
    # pick adjusted close if available
    if "Adj Close" in df.columns:
        s = df["Adj Close"].rename(name=ticker)
    else:
        s = df["Close"].copy()
        s.name = ticker
    # save raw
    out_csv = os.path.join(RAW_DIR, f"{ticker.replace('^','').replace('=','_')}.csv")
    df.to_csv(out_csv)
    return s

def fetch_all(start, end):
    series = {}
    # market tickers
    for t in list(TICKERS.keys()) + FX_TICKERS:
        s = download_ticker(t, start, end)
        if s is not None:
            series[t] = s
    # optional: fetch FRED series (example)
    # from pandas_datareader import data as pdr
    # fred_series = pdr.DataReader('DGS10', 'fred', start, end)  # 10Y yield
    # fred_series.to_csv(os.path.join(RAW_DIR, 'DGS10_fred.csv'))
    return series

def build_price_matrix(series_dict):
    # concat on dates, outer join, then align to SP500 trading calendar and forward-fill small gaps
    df = pd.concat(series_dict.values(), axis=1)
    df.columns = list(series_dict.keys())
    # choose SP500 calendar if available, else use index union
    if "^GSPC" in df.columns:
        cal_index = df.index  # SP500 series included
    else:
        cal_index = df.index
    df = df.reindex(cal_index).sort_index()
    # forward fill small gaps (e.g., non-USD market holidays)
    df = df.ffill().bfill()
    return df

def convert_to_usd(price_df):
    df = price_df.copy()
    # FTSE (GBP) -> USD using GBPUSD=X (1 GBP -> X USD)
    if "^FTSE" in df.columns and "GBPUSD=X" in df.columns:
        df["FTSE_USD"] = df["^FTSE"] * df["GBPUSD=X"]
    # Nikkei (JPY) -> USD using USDJPY=X (USDJPY = JPY per USD). Convert JPY price -> USD by division
    if "^N225" in df.columns and "USDJPY=X" in df.columns:
        df["N225_USD"] = df["^N225"] / df["USDJPY=X"]
    # Keep other USD-native tickers (rename for clarity)
    keep = {}
    for t, short in TICKERS.items():
        if t in df.columns:
            # if we created a USD version already (FTSE_USD etc), prefer that
            if t == "^FTSE" and "FTSE_USD" in df.columns:
                keep[short] = df["FTSE_USD"]
            elif t == "^N225" and "N225_USD" in df.columns:
                keep[short] = df["N225_USD"]
            else:
                keep[short] = df[t]
    # Also include GOLD_FUT (GC=F) and FX or IEF if desired
    price_usd = pd.concat(keep, axis=1)
    # sort and return
    price_usd = price_usd.sort_index()
    return price_usd

def save_outputs(price_usd):
    price_usd.to_csv(os.path.join(PROCESSED_DIR, "prices_usd.csv"))
    # log returns
    logret = np.log(price_usd / price_usd.shift(1)).dropna(how="all")
    logret.to_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"))
    print("Saved processed files:", os.path.join(PROCESSED_DIR, "prices_usd.csv"))

def main(start, end):
    print(f"Fetching tickers between {start} and {end}...")
    series = fetch_all(start, end)
    if not series:
        raise RuntimeError("No series downloaded — check tickers / network.")
    price_df = build_price_matrix(series)
    price_usd = convert_to_usd(price_df)
    save_outputs(price_usd)
    print("Done — raw CSVs are in data/raw/, processed CSVs in data/processed/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2020-12-31")
    args = p.parse_args()
    main(args.start, args.end)
