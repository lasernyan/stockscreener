#!/usr/bin/env python3
"""
XAUUSD (Gold) Breakout Strategy Backtest
=========================================
Strategy: Buy when price breaks above the previous N-period high,
          sell when price breaks below the previous N-period low.

Uses yfinance to fetch Gold futures (GC=F) as a proxy for XAUUSD.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


# ── Configuration ──────────────────────────────────────────────────────────
TICKER = "GC=F"                # Gold futures (proxy for XAUUSD)
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
LOOKBACK = 20                  # N-period high/low lookback (trading days)
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02          # 2% risk per trade
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5        # Stop-loss = ATR * multiplier
ATR_TP_MULTIPLIER = 3.0        # Take-profit = ATR * multiplier


def generate_synthetic_gold_data(start: str, end: str) -> pd.DataFrame:
    """Generate realistic synthetic XAUUSD OHLCV data using geometric Brownian motion."""
    print("  Generating synthetic XAUUSD price data (network unavailable) ...")
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Parameters calibrated to gold: ~15% annual vol, slight upward drift
    initial_price = 1550.0       # Approximate gold price at start of 2020
    annual_drift = 0.08
    annual_vol = 0.15
    dt = 1 / 252

    # GBM simulation
    daily_returns = np.exp(
        (annual_drift - 0.5 * annual_vol**2) * dt
        + annual_vol * np.sqrt(dt) * np.random.randn(n)
    )
    close = initial_price * np.cumprod(daily_returns)

    # Generate OHLV from close
    daily_range = close * np.random.uniform(0.005, 0.02, size=n)
    high = close + daily_range * np.random.uniform(0.3, 0.7, size=n)
    low = close - daily_range * np.random.uniform(0.3, 0.7, size=n)
    open_ = low + (high - low) * np.random.uniform(0.2, 0.8, size=n)
    volume = np.random.randint(50_000, 300_000, size=n)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)
    return df


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance, fall back to synthetic data."""
    print(f"Fetching {ticker} data from {start} to {end} ...")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            df.dropna(inplace=True)
            print(f"  Fetched {len(df)} bars from Yahoo Finance.\n")
            return df
    except Exception as e:
        print(f"  Download failed: {e}")

    # Fallback to synthetic data
    df = generate_synthetic_gold_data(start, end)
    print(f"  Generated {len(df)} synthetic bars.\n")
    return df


def compute_signals(df: pd.DataFrame, lookback: int, atr_period: int) -> pd.DataFrame:
    """Compute breakout signals, ATR, and stop/target levels."""
    df = df.copy()

    # Rolling high / low channels
    df["Upper"] = df["High"].rolling(lookback).max()
    df["Lower"] = df["Low"].rolling(lookback).min()

    # ATR for position sizing & stops
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(atr_period).mean()

    # Shift channels by 1 to avoid look-ahead bias
    df["Upper_prev"] = df["Upper"].shift(1)
    df["Lower_prev"] = df["Lower"].shift(1)

    # Signals
    df["Long_Signal"] = df["Close"] > df["Upper_prev"]
    df["Short_Signal"] = df["Close"] < df["Lower_prev"]

    df.dropna(inplace=True)
    return df


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate the breakout strategy and return a trade log."""
    capital = INITIAL_CAPITAL
    position = 0       # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    equity_curve = []

    for i, (idx, row) in enumerate(df.iterrows()):
        pnl = 0.0

        if position != 0:
            # Check stop-loss / take-profit
            if position == 1:
                if row["Low"] <= stop_loss:
                    pnl = stop_loss - entry_price
                    trades[-1].update(exit_date=idx, exit_price=stop_loss,
                                      pnl=pnl, exit_reason="SL")
                    capital += pnl * trades[-1]["qty"]
                    position = 0
                elif row["High"] >= take_profit:
                    pnl = take_profit - entry_price
                    trades[-1].update(exit_date=idx, exit_price=take_profit,
                                      pnl=pnl, exit_reason="TP")
                    capital += pnl * trades[-1]["qty"]
                    position = 0
            elif position == -1:
                if row["High"] >= stop_loss:
                    pnl = entry_price - stop_loss
                    trades[-1].update(exit_date=idx, exit_price=stop_loss,
                                      pnl=pnl, exit_reason="SL")
                    capital += pnl * trades[-1]["qty"]
                    position = 0
                elif row["Low"] <= take_profit:
                    pnl = entry_price - take_profit
                    trades[-1].update(exit_date=idx, exit_price=take_profit,
                                      pnl=pnl, exit_reason="TP")
                    capital += pnl * trades[-1]["qty"]
                    position = 0

        # Open new position if flat
        if position == 0:
            atr = row["ATR"]
            if row["Long_Signal"] and atr > 0:
                position = 1
                entry_price = row["Close"]
                stop_loss = entry_price - ATR_SL_MULTIPLIER * atr
                take_profit = entry_price + ATR_TP_MULTIPLIER * atr
                risk_per_unit = ATR_SL_MULTIPLIER * atr
                qty = max(1, int((capital * RISK_PER_TRADE) / risk_per_unit))
                trades.append(dict(
                    entry_date=idx, direction="LONG",
                    entry_price=entry_price, sl=stop_loss, tp=take_profit,
                    qty=qty, exit_date=None, exit_price=None,
                    pnl=None, exit_reason=None,
                ))
            elif row["Short_Signal"] and atr > 0:
                position = -1
                entry_price = row["Close"]
                stop_loss = entry_price + ATR_SL_MULTIPLIER * atr
                take_profit = entry_price - ATR_TP_MULTIPLIER * atr
                risk_per_unit = ATR_SL_MULTIPLIER * atr
                qty = max(1, int((capital * RISK_PER_TRADE) / risk_per_unit))
                trades.append(dict(
                    entry_date=idx, direction="SHORT",
                    entry_price=entry_price, sl=stop_loss, tp=take_profit,
                    qty=qty, exit_date=None, exit_price=None,
                    pnl=None, exit_reason=None,
                ))

        equity_curve.append({"date": idx, "equity": capital})

    # Close any open position at last close
    if position != 0 and trades:
        last = df.iloc[-1]
        if position == 1:
            pnl = last["Close"] - entry_price
        else:
            pnl = entry_price - last["Close"]
        trades[-1].update(exit_date=df.index[-1], exit_price=last["Close"],
                          pnl=pnl, exit_reason="EOD")
        capital += pnl * trades[-1]["qty"]

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return trade_df, equity_df


def print_stats(trade_df: pd.DataFrame, equity_df: pd.DataFrame) -> None:
    """Print backtest performance summary."""
    if trade_df.empty:
        print("No trades were generated.")
        return

    closed = trade_df.dropna(subset=["pnl"]).copy()
    closed["dollar_pnl"] = closed["pnl"] * closed["qty"]
    total_pnl = closed["dollar_pnl"].sum()
    wins = closed[closed["dollar_pnl"] > 0]
    losses = closed[closed["dollar_pnl"] <= 0]
    win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
    avg_win = wins["dollar_pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["dollar_pnl"].mean() if len(losses) > 0 else 0
    profit_factor = (wins["dollar_pnl"].sum() / abs(losses["dollar_pnl"].sum())
                     if len(losses) > 0 and losses["dollar_pnl"].sum() != 0 else float("inf"))

    # Max drawdown
    peak = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - peak) / peak
    max_dd = dd.min() * 100

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    print("=" * 60)
    print("         XAUUSD BREAKOUT BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period           : {START_DATE} to {END_DATE}")
    print(f"  Lookback         : {LOOKBACK} periods")
    print(f"  Initial Capital  : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Equity     : ${final_equity:,.2f}")
    print(f"  Total Return     : {total_return:+.2f}%")
    print(f"  Max Drawdown     : {max_dd:.2f}%")
    print("-" * 60)
    print(f"  Total Trades     : {len(closed)}")
    print(f"  Winning Trades   : {len(wins)}  ({win_rate:.1f}%)")
    print(f"  Losing Trades    : {len(losses)}")
    print(f"  Avg Win ($)      : ${avg_win:,.2f}")
    print(f"  Avg Loss ($)     : ${avg_loss:,.2f}")
    print(f"  Profit Factor    : {profit_factor:.2f}")
    print(f"  Total P&L ($)    : ${total_pnl:,.2f}")
    print("=" * 60)


def plot_results(df: pd.DataFrame, equity_df: pd.DataFrame,
                 trade_df: pd.DataFrame) -> None:
    """Generate and save result charts."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1, 2]})

    # ── Price + channels ──
    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Close", linewidth=0.8, color="black")
    ax.plot(df.index, df["Upper_prev"], label=f"{LOOKBACK}-bar High",
            linewidth=0.6, color="green", linestyle="--")
    ax.plot(df.index, df["Lower_prev"], label=f"{LOOKBACK}-bar Low",
            linewidth=0.6, color="red", linestyle="--")

    # Mark entries
    if not trade_df.empty:
        longs = trade_df[trade_df["direction"] == "LONG"]
        shorts = trade_df[trade_df["direction"] == "SHORT"]
        ax.scatter(longs["entry_date"], longs["entry_price"],
                   marker="^", color="green", s=60, zorder=5, label="Long Entry")
        ax.scatter(shorts["entry_date"], shorts["entry_price"],
                   marker="v", color="red", s=60, zorder=5, label="Short Entry")

    ax.set_title("XAUUSD Breakout Strategy", fontsize=14)
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── ATR ──
    ax = axes[1]
    ax.plot(df.index, df["ATR"], color="purple", linewidth=0.7)
    ax.set_ylabel("ATR")
    ax.set_title(f"Average True Range ({ATR_PERIOD})", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Equity curve ──
    ax = axes[2]
    ax.plot(equity_df.index, equity_df["equity"], color="blue", linewidth=1)
    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Equity ($)")
    ax.set_title("Equity Curve", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "xauusd_breakout_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = fetch_data(TICKER, START_DATE, END_DATE)
    signals = compute_signals(data, LOOKBACK, ATR_PERIOD)
    trade_log, equity = run_backtest(signals)
    print_stats(trade_log, equity)
    plot_results(signals, equity, trade_log)

    # Save trade log
    if not trade_log.empty:
        trade_log.to_csv("xauusd_trades.csv", index=False)
        print("Trade log saved to xauusd_trades.csv")
