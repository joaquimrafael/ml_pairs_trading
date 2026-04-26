# Machine Learning Enhanced Pairs Trading

> **Fork notice.** This repository is a fork of the official implementation of
> [Machine Learning Enhanced Pairs Trading](https://www.mdpi.com/2571-9394/6/2/24)
> (Hodarkar & Lemeneh, MDPI Forecasting 2024). It has been extended as part of an
> undergraduate thesis (TCC) focused on applying the pipeline to the Brazilian
> equities market, adding a statistical pair-selection step, new forecasting
> models, and additional trading strategies. See the [Contributors](#contributors)
> section for credit to the original authors.

## Table of Contents

- [Introduction](#introduction)
- [What was changed in this fork](#what-was-changed-in-this-fork)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Example Usage](#example-usage)
- [Output](#output)
- [Contributors](#contributors)

## Introduction

This project explores and applies deep learning, classical ML, and reinforcement
learning models to predict changes in the price ratio of closely related stock
pairs. Once the ratio changes are predicted, several pairs trading strategies are
simulated to assess financial performance.

Forecasting models available:

- Bidirectional Long Short-Term Memory (BiLSTM) with Attention — TensorFlow/Keras
- Long Short-Term Memory (LSTM) — Darts `RNNModel`
- Transformer
- N-BEATS (Neural Basis Expansion Analysis)
- N-HiTS (Neural Hierarchical Interpolation for Time Series)
- Temporal Convolutional Network (TCN)
- Temporal Fusion Transformer (TFT)
- Random Forest (as a Darts `RegressionModel`)
- MADDPG (Multi-Agent DDPG with three actors and one critic), following the
  TS-MADDPG framework from [Improved pairs trading strategy using two-level reinforcement learning framework](https://www.sciencedirect.com/science/article/abs/pii/S0952197623013325) (Xu & Luo)

Trading strategies applied to the predicted ratio:

- **Reversion (mean reversion)**
  - If ratio(t) > ratio(t−1): sell numerator, buy denominator.
  - If ratio(t) < ratio(t−1): buy numerator, sell denominator.
  - Otherwise: no trade.
- **Pure forecasting**
  - If predicted ratio(t+1) > ratio(t): buy numerator, sell denominator.
  - If predicted ratio(t+1) < ratio(t): sell numerator, buy denominator.
  - Otherwise: no trade.
- **Hybrid**
  - If predicted ratio(t+1) < ratio(t) and ratio(t) > ratio(t−1): sell numerator, buy denominator.
  - If predicted ratio(t+1) > ratio(t) and ratio(t) < ratio(t−1): buy numerator, sell denominator.
  - Otherwise: no trade.
- **Momentum** (added in this fork) — trades in the direction of the last move.
- **Ground truth** (added in this fork) — oracle baseline using the true next
  ratio, used as an upper bound for strategy evaluation.

## What was changed in this fork

Relative to the original repository:

- **Datasets.** Three Brazilian minute-level stock-pair datasets
  (Petrobras, Itaú Unibanco, Bradesco) replace the original example data.
- **Statistical pair-selection step** (`data_processing/pair_selector.py`),
  executed automatically before training. It runs Pearson correlation,
  Engle-Granger cointegration with OLS hedge-ratio estimation, an ADF test on
  the spread, and an AR(1)-based mean-reversion half-life estimate. Results are
  printed and saved to CSV, and feed back into the pipeline:
  - The OLS-estimated hedge ratio β replaces the β = 1 assumption in spread
    computation.
  - `INPUT_CHUNK_LENGTH` is set dynamically from the half-life (with ADF lags
    as a fallback when mean-reversion is not detected), overriding the CLI
    `--input_chunk_length`.
- **σ-based thresholds.** Trading thresholds are derived from the standard
  deviation of ratio changes over the training split (`[0, 0.5σ, 1σ, 2σ]`) and
  override the CLI `--thresholds`.
- **Three additional forecasting models:** LSTM, TFT, and Random Forest.
- **Two additional strategies:** momentum and ground-truth oracle.
- **Richer plotting / persistence.** Per-pair, per-model plots under
  `plot/<dataset_name>/<model_name>/` using colorblind-friendly palettes,
  plus CSVs of predictions, metrics, and pair-analysis results.

## Dataset

The datasets shipped under `dataset/` are minute-level OHLC-derived pairs from
B3 (Brazilian stock exchange):

| File | Pair (numerator / denominator) | Start date |
| --- | --- | --- |
| `dataset/petr3_4_min.csv` | PETR4 / PETR3 (Petrobras) | 2022-01-03 |
| `dataset/itau3_4_min.csv` | ITAU4 / ITAU3 (Itaú Unibanco) | 2022-01-03 |
| `dataset/bbdc3_4_min.csv` | BBDC3 / BBDC4 (Bradesco) | 2023-01-03 |

All files share the same CSV schema: a timestamp column followed by two price
columns (`Stock_A`, `Stock_B`). Example:

| Time | A | B |
| :--- | :---: | :---: |
| 2023 01 03 10 09 00.000 | 13.15 | 14.75 |
| 2023 01 03 10 10 00.000 | 13.10 | 14.68 |
| 2023 01 03 10 11 00.000 | 13.10 | 14.65 |

The pair ratio `p = A / B` is the target variable used in training and in the
trading strategies. To use a custom dataset, save it with the same column
layout and pass its path via `--data_path`.

## Pipeline

High-level flow executed by `run_trading_strategy.py` in SL mode:

1. **Load** the CSV and compute the `ratio` and `difference` columns
   (`DataProcessor`).
2. **Pair analysis** on the raw price series (`PairSelector`): correlation,
   cointegration, ADF on spread, half-life. The OLS-estimated β is stored back
   in `DataProcessor`, and the half-life sets `INPUT_CHUNK_LENGTH`.
3. **σ-based threshold derivation** from ratio changes in the training split.
4. **Split and scale** into train / validation / test sets
   (default 0.7 / 0.1 / 0.2 for the Darts path; BiLSTM uses its own windowing).
5. **Train** the selected model and **roll-forecast** over the test set.
6. **Simulate** all five strategies across every threshold and compute
   profits, Sharpe ratios, trade counts, and classification metrics
   (accuracy, precision, recall, F1, confusion matrices).
7. **Persist** plots and CSVs under `plot/<dataset_name>/<model_name>/`.

In RL mode, `DataProcessor.compute_states()` builds spread-ratio state vectors
and the MADDPG agent (3 actors, 1 critic) is trained on the training split and
evaluated on the test split.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify the environment
python test.py
```

Main dependencies: `darts==0.38.0`, `torch>=2.6.0`, `tensorflow>=2.20.0`,
`pandas==2.2.2`, `numpy>=2.0.0`, `statsmodels>=0.14.0`, `scipy>=1.12.0`,
`seaborn==0.13.2`, `matplotlib>=3.9.0`.

## Example Usage

Supervised learning (required: `--sl` and `--data_path`):

```bash
python run_trading_strategy.py --sl --sl_model tcn --data_path dataset/petr3_4_min.csv --n_epochs 3
python run_trading_strategy.py --sl --sl_model nbeats --data_path dataset/itau3_4_min.csv --n_epochs 50
python run_trading_strategy.py --sl --sl_model random_forest --data_path dataset/bbdc3_4_min.csv
```

Reinforcement learning (MADDPG):

```bash
python run_trading_strategy.py --rl --data_path dataset/petr3_4_min.csv
```

Full list of CLI flags:

```CLI
--rl                     Enable the RL-based model (MADDPG).
--sl                     Enable the SL-based model.
--sl_model               One of: bilstm, lstm, tcn, transformer, nbeats, nhits, tft, random_forest.
                         Default: tcn.
--input_chunk_length     Input sequence length. NOTE: overridden at runtime by the
                         half-life from pair analysis (or by ADF lags as fallback).
--output_chunk_length    Output sequence length. Default: 1.
--n_epochs               Number of training epochs. Default: 50.
--batch_size             Batch size. Default: 1024.
--train_ratio            Fraction of data used for training (validation is 10%).
                         Default: 0.5.
--data_path              Path to the dataset CSV (required).
--thresholds             Comma-separated list of four trade thresholds. NOTE:
                         overridden at runtime by σ-based thresholds computed
                         from the training split.
```

## Output

All artefacts are written under `plot/<dataset_name>/<model_name>/`, including:

- Prediction vs. ground-truth plots, residual distribution, correlation heatmap.
- Rolling correlation and spread with bands (from pair analysis).
- Cumulative P&L at the best threshold (by Sharpe and by total profit).
- Per-strategy confusion matrices, accuracy / precision / recall / F1 curves.
- CSVs of predictions, per-strategy metrics, and the pair-analysis report.

## Contributors

**Original authors (upstream repository):**

1. Sohail Hodarkar — <sph8686@nyu.edu>
2. Beakal Lemeneh — <beakalmulusew@gmail.com>

**This fork:**

- Joaquim Prieto & Lucas Trebachetti: undergraduate thesis (TCC) extending the pipeline to
  Brazilian equity pairs with statistical pair selection and additional
  models / strategies.
