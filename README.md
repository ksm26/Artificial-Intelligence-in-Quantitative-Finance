# 📈 AI in Quantitative Finance — Deep Learning for Portfolio Management

This project implements the end-to-end pipeline described in  
**"Artificial Intelligence in Quantitative Finance: Leveraging Deep Learning for Smarter Portfolio Management and Asset Allocation"**.

We combine **LSTM** and **Transformer** models for **financial return forecasting**, integrate the predictions into a **dynamic portfolio allocation strategy**, and evaluate using **comprehensive backtesting and statistical testing**.

---

## 📂 Project Structure

project-root/\
│
├── data/\
│ ├── raw/ # Raw downloaded price data\
│ ├── processed/ # Preprocessed features & targets\
│ │ ├── features_scaled.csv\
│ │ ├── targets.csv\
│ │ ├── scaler_features.pkl\
│ │ └── index_splits.json\
│
├── notebooks/\
│ ├── 01_data_exploration.ipynb\
│ ├── 02_model_training.ipynb\
│ ├── 03_backtest_and_eval.ipynb\
│
├── results/\
│ ├── checkpoints/ # Saved LSTM & Transformer models by timestamp\
│ │ ├── lstm/\
│ │ └── transformer/\
│ ├── figures/ # Plots & visualizations\
│ ├── metrics_lstm.csv\
│ ├── metrics_transformer.csv\
│ ├── predictions_lstm.csv\
│ └── predictions_transformer.csv\
│
├── src/\
│ ├── data_fetch.py # Step 1: Download financial data\
│ ├── preprocessing.py # Step 2: Build features, scale, save datasets\
│ ├── train.py # Step 3: Train LSTM or Transformer models\
│ ├── predict.py # Step 5: Generate rolling forecasts\
│ ├── optimizer.py # Portfolio optimization logic\
│ ├── backtest.py # Step 5: Backtesting engine\
│ ├── metrics.py # Step 6: Performance metrics & statistical tests\
│ ├── evaluate_and_plot.py # Step 6: Run evaluation, save metrics & plots\
│ └── datasets.py # SequenceDataset loader\
│
└── environment.yml # Conda environment with dependencies\

---

## ⚙️ Pipeline Overview

The workflow follows **six main steps**:

### **Step 1 — Data Collection**
- **Script:** `src/data_fetch.py`
- Fetches historical financial price data from the internet (e.g., Yahoo Finance).
- Saves raw prices to `data/raw/`.

**Run:**
```bash
python src/data_fetch.py --start 2010-01-01 --end 2023-12-31
```

### **Step 2 — Data Preprocessing**
- **Script:** `src/preprocessing.py`

- Builds predictive features (technical indicators, returns, rolling stats).

- Scales features (StandardScaler) using train split only.

- Saves: `features_scaled.csv`, `targets.csv` , `scaler_features.pkl` , `index_splits.json`

**Run:**
```bash
python src/preprocessing.py --data data/processed/prices_usd.csv
```

### **Step 3 — Model Training**
- **Script:** `src/train.py`

- Defines LSTM and Transformer architectures.

- Saves checkpoints to `results/checkpoints/<model>/<timestamp>/epoch_<n>.pth`

**Run:**
```bash
# LSTM
python src/train.py --model lstm --epochs 50

# Transformer
python src/train.py --model transformer --epochs 50
```

### **Step 4 — Forecast Generation**
- **Script:** `src/predict.py`

- Loads latest checkpoint for the chosen model.

- Generates rolling walk-forward predictions for the test set.

- Saves to `results/predictions_<model>.csv`

**Run:**
```bash
python src/predict.py --model-type lstm
python src/predict.py --model-type transformer
```

### **Step 5 — Portfolio Optimization & Backtest**
- **Script:** `src/optimizer.py` → Mean-Variance Optimizer.

- `src/backtest.py` → Computes portfolio returns from weights & realized returns.

### **Step 6 — Evaluation & Plots**
- **Script:** `src/evaluate_and_plot.py`

- Loads predictions & realized returns.

- Computes:

- Annualized return, volatility, Sharpe ratio, Sortino ratio, Calmar ratio.

- RMSE & directional accuracy for forecasts.

- Statistical tests: Paired t-test, Jobson-Korkie-Memmel.

- Saves:

- Metrics CSV → `results/metrics_<model>.csv`

- Plots → `results/figures/`

**Run:**
```bash
python src/evaluate_and_plot.py --model-type lstm
python src/evaluate_and_plot.py --model-type transformer
```



## 📊 Outcomes
- **LSTM & Transformer** deep learning models for return forecasting.

- Fully automated **data-to-decisions pipeline:**

Data fetching → Feature engineering → Model training → Prediction → Portfolio construction → Evaluation.

- **Dynamic portfolio allocation** using forecasts.

- **Comprehensive evaluation** with statistical significance testing.

- **Visual outputs** for cumulative returns, rolling Sharpe ratio, and portfolio weights.

## 🚀 What This Project Achieves
- Demonstrates **state-of-the-art AI techniques** (LSTM, Transformer) in finance.

- Provides a **research-ready backtesting framework**.

- Bridges **ML predictions** with portfolio management.

- Produces **replicable & extensible** results for further experimentation.

## 💻 Environment Setup

```bash
conda env create -f environment.yml
conda activate project
```

## 🏁 Quickstart

```bash
# 1. Fetch data
python src/data_fetch.py --start 2010-01-01 --end 2023-12-31

# 2. Preprocess
python src/preprocessing.py

# 3. Train model
python src/train.py --model lstm --epochs 50

# 4. Predict
python src/predict.py --model-type lstm

# 5. Evaluate
python src/evaluate_and_plot.py --model-type lstm
```
