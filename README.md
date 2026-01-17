# Forex Mean-Reversion Trading Strategy & Backtesting

## Overview
This project implements a rule-based Forex mean-reversion trading strategy on the EUR/USD currency pair using Python.
The strategy is designed to exploit short-term price reversions in range-bound market conditions.

## Strategy Logic
The trading strategy is based on:
- Relative Strength Index (RSI) to detect momentum exhaustion
- Bollinger Bands to identify price deviations from the statistical mean

### Entry Rules
- Long position when RSI < 30 and price is below the lower Bollinger Band
- Short position when RSI > 70 and price is above the upper Bollinger Band

### Exit Rules
- Exit long positions when RSI returns to 50
- Exit short positions when RSI returns to 50

All positions are shifted by one period to avoid lookahead bias.

## Data
- Instrument: EUR/USD
- Frequency: Daily
- Source: Yahoo Finance
- Time horizon: ~10 years

## Backtesting Methodology
- Daily returns computed from adjusted close prices
- Strategy returns calculated as position × market return
- Performance evaluated using:
  - Annualized return
  - Annualized volatility
  - Sharpe ratio
  - Maximum drawdown
  - Trade-level win rate

## Results Summary
- Strategy exhibits higher win rate in range-bound market regimes
- Drawdowns are controlled compared to buy-and-hold
- Performance degrades during strong trending markets

## Limitations
- Transaction costs and FX spreads are not included
- Strategy performance is regime-dependent
- Daily data does not capture intraday dynamics
- Mean-reversion assumptions may fail during macroeconomic events

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- yFinance

## Key Learning
This project demonstrates systematic trading logic, backtesting discipline, and risk-aware strategy evaluation suitable for quant and trading analytics roles.
