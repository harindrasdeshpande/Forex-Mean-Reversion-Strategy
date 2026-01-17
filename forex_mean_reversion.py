# STEP 1: DATA ACQUISITION & PREPARATION

import pandas as pd
import numpy as np
import yfinance as yf

# Download EUR/USD daily data
pair = "EURUSD=X"
df = yf.download(pair, period="10y", interval="1d", auto_adjust=True)

# Keep relevant columns
# Flatten columns and extract proper price series
df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close"]]
df.dropna(inplace=True)

# Ensure Close is a Series
close = df["Close"].astype(float)

df.dropna(inplace=True)

# Calculate returns
df["Returns"] = df["Close"].pct_change()

print("\n--- DATA SAMPLE ---")
print(df.head())

print("\n--- DATA INFO ---")
print(df.describe())



# STEP 2: INDICATOR CONSTRUCTION

# RSI Calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = compute_rsi(close)

# Bollinger Bands
window = 20
df["MA"] = df["Close"].rolling(window).mean()
df["STD"] = df["Close"].rolling(window).std()

df["Upper_Band"] = df["MA"] + (2 * df["STD"])
df["Lower_Band"] = df["MA"] - (2 * df["STD"])

print("\n--- INDICATOR SAMPLE ---")
print(df[["Close", "RSI", "Upper_Band", "Lower_Band"]].tail())


# STEP 3: SIGNAL GENERATION

df["Signal"] = 0

# Long signal
df.loc[(df["RSI"] < 30) & (df["Close"] < df["Lower_Band"]), "Signal"] = 1

# Short signal
df.loc[(df["RSI"] > 70) & (df["Close"] > df["Upper_Band"]), "Signal"] = -1

# Position handling (shifted to avoid lookahead bias)
df["Position"] = df["Signal"].shift(1)
df["Position"] = df["Position"].fillna(0)

# Exit conditions
for i in range(1, len(df)):
    # Exit long
    if df.iloc[i - 1]["Position"] == 1 and df.iloc[i]["RSI"] >= 50:
        df.iloc[i, df.columns.get_loc("Position")] = 0

    # Exit short
    if df.iloc[i - 1]["Position"] == -1 and df.iloc[i]["RSI"] <= 50:
        df.iloc[i, df.columns.get_loc("Position")] = 0

print("\n--- SIGNAL COUNTS ---")
print(df["Signal"].value_counts())

print("\n--- POSITION COUNTS ---")
print(df["Position"].value_counts())

# STEP 4: BACKTESTING & PnL

# Strategy returns
df["Strategy_Return"] = df["Position"] * df["Returns"]

# Cumulative returns
df["Cumulative_Market_Return"] = (1 + df["Returns"]).cumprod()
df["Cumulative_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()

print("\n--- BACKTEST SAMPLE ---")
print(df[["Returns", "Position", "Strategy_Return"]].tail())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Cumulative_Market_Return"], label="Buy & Hold")
plt.plot(df.index, df["Cumulative_Strategy_Return"], label="Strategy")
plt.title("EUR/USD Mean Reversion Strategy Backtest")
plt.legend()
plt.show()

# Performance metrics
trading_days = 252

strategy_return_annual = df["Strategy_Return"].mean() * trading_days
strategy_vol_annual = df["Strategy_Return"].std() * np.sqrt(trading_days)

sharpe_ratio = strategy_return_annual / strategy_vol_annual

print("\n--- STRATEGY PERFORMANCE ---")
print("Annualized Return:", round(strategy_return_annual, 4))
print("Annualized Volatility:", round(strategy_vol_annual, 4))
print("Sharpe Ratio:", round(sharpe_ratio, 2))

# Max Drawdown
rolling_max = df["Cumulative_Strategy_Return"].cummax()
drawdown = df["Cumulative_Strategy_Return"] / rolling_max - 1
max_drawdown = drawdown.min()

print("Max Drawdown:", round(max_drawdown, 3))

# STEP 5: TRADE STATISTICS

# Identify trade changes
df["Trade_Change"] = df["Position"].diff()

# Entry and exit points
entries = df[df["Trade_Change"].abs() == 1]
trade_returns = []

current_return = 0

for i in range(len(df)):
    if df.iloc[i]["Position"] != 0:
        current_return += df.iloc[i]["Strategy_Return"]
    elif current_return != 0:
        trade_returns.append(current_return)
        current_return = 0

trade_returns = pd.Series(trade_returns)

win_rate = (trade_returns > 0).mean()
avg_win = trade_returns[trade_returns > 0].mean()
avg_loss = trade_returns[trade_returns < 0].mean()

print("\n--- TRADE STATISTICS ---")
print("Number of Trades:", len(trade_returns))
print("Win Rate:", round(win_rate, 2))
print("Average Win:", round(avg_win, 5))
print("Average Loss:", round(avg_loss, 5))


# STRATEGY LIMITATIONS
# 1. Performance degrades during strong trending markets
# 2. Assumes zero transaction costs (FX spreads will reduce returns)
# 3. Uses daily data; intraday noise not captured
# 4. Mean-reversion assumptions may break during macro events
