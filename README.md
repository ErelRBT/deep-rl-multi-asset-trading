# Deep Reinforcement Learning for Multi-Asset Long/Short Trading

A quantitative research project that builds a **Proximal Policy Optimization (PPO)** agent for dynamic long/short allocation across a portfolio of U.S. equities, then evaluates it with both a strict out-of-sample holdout and rolling walk-forward validation.

The objective is not to present a cherry-picked profitable strategy. The project focuses on **research methodology, environment design, risk-aware backtesting, and honest benchmark comparison**.

## Project Overview

The project implements a custom `Gymnasium` trading environment and trains a PPO agent with `Stable-Baselines3`.

The agent allocates capital dynamically across:

- AAPL
- MSFT
- TSLA
- GOOGL
- AMZN

Daily market data are downloaded with `yfinance`.

For each asset, the state includes four lagged market features:

- 1-day return
- 5-day return
- 20-day annualized volatility
- 20-day volume z-score

The observation also includes the current risky-asset weights and residual cash weight.

The continuous action space controls:

1. signed long/short scores for each asset;
2. the portfolio's total gross-exposure budget.

## Trading Environment

The custom environment explicitly models several realistic portfolio constraints:

- **Initial capital:** $100,000
- **Maximum gross exposure:** 100%
- **Transaction cost:** 10 bps per turnover unit
- **Short-borrow cost:** 3% annualized
- **Continuous long/short allocation**
- **Residual cash allocation**
- **Episode length:** up to 252 trading steps
- **Termination condition:** portfolio value falls below 50% of initial capital

To reduce look-ahead bias, predictive features are shifted so that the decision at time *t* only uses information available from prior observations.

The PPO reward in the current version is the portfolio's one-step log return.

## Model

The agent uses `PPO("MlpPolicy")` from Stable-Baselines3 with the following core configuration:

```python
learning_rate = 3e-4
n_steps = 1024
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.20
ent_coef = 0.001
vf_coef = 0.50
seed = 42
```

Each training window runs for **200,000 timesteps**.

## Validation Design

Two evaluation protocols are included.

### 1. Holdout test

A PPO model trained on 2020-2022 data is evaluated on previously unseen data from the first half of 2023.

| Metric | PPO | Equal-Weight Buy & Hold |
|---|---:|---:|
| Total Return | -1.10% | 25.84% |
| Sharpe Ratio | -0.185 | 2.471 |
| Max Drawdown | -7.81% | -9.71% |

### 2. Rolling walk-forward validation

The main experiment uses **10 rolling windows**:

- 5 years of training data
- followed by 6 months of out-of-sample testing
- training window shifted forward by 6 months
- test periods covering 2019-2023

Capital is carried forward from one out-of-sample window to the next.

| Metric | PPO Walk-Forward | Equal-Weight Buy & Hold |
|---|---:|---:|
| Total Return | 24.31% | 404.95% |
| Sharpe Ratio | 0.374 | 1.065 |
| Max Drawdown | -37.23% | -57.37% |


## Interpretation

The current PPO specification **does not outperform the equal-weight buy-and-hold benchmark** on cumulative return or Sharpe ratio.

That result is important rather than something to hide. It suggests that the current state representation, reward function, asset universe, and training procedure are not sufficient to extract a robust allocation policy from this concentrated mega-cap equity universe.

At the same time, the walk-forward PPO path exhibits a smaller maximum drawdown than the benchmark in this experiment.

The project therefore demonstrates the full research loop:

**hypothesis → environment design → model training → out-of-sample testing → benchmark comparison → diagnosis → next iteration.**

## Why This Project Is Relevant

This repository demonstrates practical experience with:

- deep reinforcement learning;
- quantitative portfolio construction;
- custom Gymnasium environment design;
- continuous action spaces;
- transaction-cost and short-borrow modelling;
- financial feature engineering;
- out-of-sample backtesting;
- walk-forward validation;
- Sharpe ratio and drawdown analysis;
- benchmark-based model evaluation;
- reproducible Python research workflows.

## Planned Improvements

The next research iterations are to:

- replace the raw return reward with a risk-adjusted objective;
- normalize or standardize the input features;
- use a broader, cross-sector asset universe;
- compare PPO with alternative RL algorithms and simpler allocation rules;
- run multiple random seeds and report dispersion of results;
- introduce a dedicated validation layer for hyperparameter selection;
- test sensitivity to transaction costs and short-borrow assumptions;
- add volatility targeting and explicit turnover constraints;
- refactor the environment and evaluation pipeline into reusable Python modules.
- Train the model on improving the sharp ratio

## Tech Stack

`Python` · `NumPy` · `pandas` · `Matplotlib` · `yfinance` · `Gymnasium` · `Stable-Baselines3` · `PPO`
