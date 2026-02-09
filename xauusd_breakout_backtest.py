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
ATR_TP_MULTIPLIER = 3.0        # Take-profit = ATR * multiplier (used when trailing disabled)

# ── Improvement: Trend Filter ─────────────────────────────────────────────
TREND_MA_PERIOD = 200           # 200-period SMA for trend direction
USE_TREND_FILTER = True         # Only trade in trend direction

# ── Improvement: Trailing Stop ────────────────────────────────────────────
USE_TRAILING_STOP = True        # Replace fixed TP with trailing stop
TRAIL_ACTIVATION_ATR = 2.0      # Activate trailing after 2x ATR favorable move
TRAIL_DISTANCE_ATR = 3.0        # Trail distance = 3x ATR behind best price

# ── Improvement: Breakeven Stop ───────────────────────────────────────────
USE_BREAKEVEN_STOP = True       # Move SL to entry after favorable move
BREAKEVEN_ATR = 1.5             # ATR multiple to trigger breakeven

# ── Improvement: Cooldown After SL ────────────────────────────────────────
COOLDOWN_BARS = 5               # Bars to wait after SL before new entry

# ── Improvement: ADX Regime Filter ────────────────────────────────────────
ADX_PERIOD = 14                 # ADX calculation period
ADX_THRESHOLD = 20              # Minimum ADX to allow entries
USE_ADX_FILTER = True           # Only trade in trending regimes

# ── Improvement: Long-Only Bias ───────────────────────────────────────────
LONG_ONLY = False               # Set True to disable all short entries


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
    """Compute breakout signals, ATR, ADX, trend filter, and stop/target levels."""
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

    # 200-period SMA trend filter
    df["MA_200"] = df["Close"].rolling(TREND_MA_PERIOD).mean()

    # ADX calculation (Wilder's method)
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    alpha = 1 / ADX_PERIOD
    atr_smooth = tr.ewm(alpha=alpha, min_periods=ADX_PERIOD).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, min_periods=ADX_PERIOD).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=ADX_PERIOD).mean() / atr_smooth)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=alpha, min_periods=ADX_PERIOD).mean()

    # Shift channels by 1 to avoid look-ahead bias
    df["Upper_prev"] = df["Upper"].shift(1)
    df["Lower_prev"] = df["Lower"].shift(1)

    # Base breakout signals
    long_signal = df["Close"] > df["Upper_prev"]
    short_signal = df["Close"] < df["Lower_prev"]

    # Apply trend filter
    if USE_TREND_FILTER:
        long_signal = long_signal & (df["Close"] > df["MA_200"])
        short_signal = short_signal & (df["Close"] < df["MA_200"])

    # Apply ADX filter
    if USE_ADX_FILTER:
        trending = df["ADX"] > ADX_THRESHOLD
        long_signal = long_signal & trending
        short_signal = short_signal & trending

    # Apply long-only bias
    if LONG_ONLY:
        short_signal = pd.Series(False, index=df.index)

    df["Long_Signal"] = long_signal
    df["Short_Signal"] = short_signal

    df.dropna(inplace=True)
    return df


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate the breakout strategy with trailing stop, breakeven, and cooldown."""
    capital = INITIAL_CAPITAL
    position = 0       # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    equity_curve = []

    # Trailing stop / breakeven state
    trailing_active = False
    breakeven_hit = False
    best_price = 0.0
    trail_atr = 0.0

    # Cooldown state
    cooldown_remaining = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        pnl = 0.0

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if position != 0:
            # ── LONG position management ──
            if position == 1:
                # Track best price for trailing/breakeven
                if row["High"] > best_price:
                    best_price = row["High"]

                # Breakeven: move SL to entry after 1x ATR move
                if USE_BREAKEVEN_STOP and not breakeven_hit:
                    if best_price >= entry_price + BREAKEVEN_ATR * trail_atr:
                        stop_loss = entry_price
                        breakeven_hit = True

                # Trailing stop: activate after favorable move, trail behind best
                if USE_TRAILING_STOP and trailing_active:
                    trail_sl = best_price - TRAIL_DISTANCE_ATR * trail_atr
                    stop_loss = max(stop_loss, trail_sl)
                elif USE_TRAILING_STOP and not trailing_active:
                    if best_price >= entry_price + TRAIL_ACTIVATION_ATR * trail_atr:
                        trailing_active = True
                        trail_sl = best_price - TRAIL_DISTANCE_ATR * trail_atr
                        stop_loss = max(stop_loss, trail_sl)

                # Check stop-loss
                if row["Low"] <= stop_loss:
                    pnl = stop_loss - entry_price
                    reason = "TSL" if trailing_active else ("BE" if breakeven_hit else "SL")
                    trades[-1].update(exit_date=idx, exit_price=stop_loss,
                                      pnl=pnl, exit_reason=reason)
                    capital += pnl * trades[-1]["qty"]
                    position = 0
                    if reason == "SL":
                        cooldown_remaining = COOLDOWN_BARS
                # Check fixed TP (only if trailing stop is disabled)
                elif not USE_TRAILING_STOP and row["High"] >= take_profit:
                    pnl = take_profit - entry_price
                    trades[-1].update(exit_date=idx, exit_price=take_profit,
                                      pnl=pnl, exit_reason="TP")
                    capital += pnl * trades[-1]["qty"]
                    position = 0

            # ── SHORT position management ──
            elif position == -1:
                # Track best price (lowest low for shorts)
                if row["Low"] < best_price:
                    best_price = row["Low"]

                # Breakeven
                if USE_BREAKEVEN_STOP and not breakeven_hit:
                    if best_price <= entry_price - BREAKEVEN_ATR * trail_atr:
                        stop_loss = entry_price
                        breakeven_hit = True

                # Trailing stop for shorts
                if USE_TRAILING_STOP and trailing_active:
                    trail_sl = best_price + TRAIL_DISTANCE_ATR * trail_atr
                    stop_loss = min(stop_loss, trail_sl)
                elif USE_TRAILING_STOP and not trailing_active:
                    if best_price <= entry_price - TRAIL_ACTIVATION_ATR * trail_atr:
                        trailing_active = True
                        trail_sl = best_price + TRAIL_DISTANCE_ATR * trail_atr
                        stop_loss = min(stop_loss, trail_sl)

                # Check stop-loss
                if row["High"] >= stop_loss:
                    pnl = entry_price - stop_loss
                    reason = "TSL" if trailing_active else ("BE" if breakeven_hit else "SL")
                    trades[-1].update(exit_date=idx, exit_price=stop_loss,
                                      pnl=pnl, exit_reason=reason)
                    capital += pnl * trades[-1]["qty"]
                    position = 0
                    if reason == "SL":
                        cooldown_remaining = COOLDOWN_BARS
                # Check fixed TP (only if trailing stop is disabled)
                elif not USE_TRAILING_STOP and row["Low"] <= take_profit:
                    pnl = entry_price - take_profit
                    trades[-1].update(exit_date=idx, exit_price=take_profit,
                                      pnl=pnl, exit_reason="TP")
                    capital += pnl * trades[-1]["qty"]
                    position = 0

        # ── Open new position if flat and cooldown expired ──
        if position == 0 and cooldown_remaining == 0:
            atr = row["ATR"]
            if row["Long_Signal"] and atr > 0:
                position = 1
                entry_price = row["Close"]
                stop_loss = entry_price - ATR_SL_MULTIPLIER * atr
                take_profit = entry_price + ATR_TP_MULTIPLIER * atr
                risk_per_unit = ATR_SL_MULTIPLIER * atr
                qty = max(1, int((capital * RISK_PER_TRADE) / risk_per_unit))
                best_price = entry_price
                trail_atr = atr
                trailing_active = False
                breakeven_hit = False
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
                best_price = entry_price
                trail_atr = atr
                trailing_active = False
                breakeven_hit = False
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

    # Sharpe ratio (annualized)
    equity_returns = equity_df["equity"].pct_change().dropna()
    sharpe = (equity_returns.mean() / equity_returns.std() * np.sqrt(252)
              if equity_returns.std() > 0 else 0)

    # Calmar ratio
    calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else float("inf")

    # Active filters summary
    filters = []
    if USE_TREND_FILTER:
        filters.append(f"MA{TREND_MA_PERIOD}")
    if USE_ADX_FILTER:
        filters.append(f"ADX>{ADX_THRESHOLD}")
    if USE_TRAILING_STOP:
        filters.append("Trail")
    if USE_BREAKEVEN_STOP:
        filters.append("BE")
    if COOLDOWN_BARS > 0:
        filters.append(f"CD{COOLDOWN_BARS}")
    if LONG_ONLY:
        filters.append("LongOnly")
    filter_str = ", ".join(filters) if filters else "None"

    print("=" * 60)
    print("      XAUUSD BREAKOUT BACKTEST RESULTS (IMPROVED)")
    print("=" * 60)
    print(f"  Period           : {START_DATE} to {END_DATE}")
    print(f"  Lookback         : {LOOKBACK} periods")
    print(f"  Active Filters   : {filter_str}")
    print(f"  Initial Capital  : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Equity     : ${final_equity:,.2f}")
    print(f"  Total Return     : {total_return:+.2f}%")
    print(f"  Max Drawdown     : {max_dd:.2f}%")
    print(f"  Sharpe Ratio     : {sharpe:.2f}")
    print(f"  Calmar Ratio     : {calmar:.2f}")
    print("-" * 60)
    print(f"  Total Trades     : {len(closed)}")
    print(f"  Winning Trades   : {len(wins)}  ({win_rate:.1f}%)")
    print(f"  Losing Trades    : {len(losses)}")
    print(f"  Avg Win ($)      : ${avg_win:,.2f}")
    print(f"  Avg Loss ($)     : ${avg_loss:,.2f}")
    print(f"  Profit Factor    : {profit_factor:.2f}")
    print(f"  Total P&L ($)    : ${total_pnl:,.2f}")
    print("-" * 60)

    # Per-direction breakdown
    for direction in ["LONG", "SHORT"]:
        dir_trades = closed[closed["direction"] == direction]
        if len(dir_trades) > 0:
            dir_wins = dir_trades[dir_trades["dollar_pnl"] > 0]
            dir_wr = len(dir_wins) / len(dir_trades) * 100
            dir_pnl = dir_trades["dollar_pnl"].sum()
            print(f"  {direction:5s} Trades    : {len(dir_trades)}  "
                  f"({dir_wr:.1f}% win) | P&L: ${dir_pnl:,.2f}")

    # Exit reason breakdown
    print("-" * 60)
    for reason in sorted(closed["exit_reason"].unique()):
        r_trades = closed[closed["exit_reason"] == reason]
        r_pnl = r_trades["dollar_pnl"].mean()
        print(f"  {reason:3s} exits       : {len(r_trades):3d}  |  "
              f"Avg $PnL: ${r_pnl:,.2f}")
    print("=" * 60)


def plot_results(df: pd.DataFrame, equity_df: pd.DataFrame,
                 trade_df: pd.DataFrame) -> None:
    """Generate and save result charts (4 panels: price, ATR, ADX, equity)."""
    has_adx = "ADX" in df.columns
    n_panels = 4 if has_adx else 3
    ratios = [3, 1, 1, 2] if has_adx else [3, 1, 2]
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 14), sharex=False,
                             gridspec_kw={"height_ratios": ratios})

    # ── Panel 1: Price + channels + MA ──
    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Close", linewidth=0.8, color="black")
    ax.plot(df.index, df["Upper_prev"], label=f"{LOOKBACK}-bar High",
            linewidth=0.6, color="green", linestyle="--")
    ax.plot(df.index, df["Lower_prev"], label=f"{LOOKBACK}-bar Low",
            linewidth=0.6, color="red", linestyle="--")

    # 200-MA trend filter line
    if "MA_200" in df.columns:
        ax.plot(df.index, df["MA_200"], label=f"{TREND_MA_PERIOD} MA",
                linewidth=1.2, color="orange")

    # Mark entries
    if not trade_df.empty:
        longs = trade_df[trade_df["direction"] == "LONG"]
        shorts = trade_df[trade_df["direction"] == "SHORT"]
        if not longs.empty:
            ax.scatter(longs["entry_date"], longs["entry_price"],
                       marker="^", color="green", s=60, zorder=5, label="Long Entry")
        if not shorts.empty:
            ax.scatter(shorts["entry_date"], shorts["entry_price"],
                       marker="v", color="red", s=60, zorder=5, label="Short Entry")

        # Mark exits by reason
        closed = trade_df.dropna(subset=["exit_date"])
        for reason, color, marker in [("TP", "blue", "D"), ("TSL", "orange", "s"),
                                       ("BE", "cyan", "o"), ("SL", "red", "x")]:
            exits = closed[closed["exit_reason"] == reason]
            if not exits.empty:
                ax.scatter(exits["exit_date"], exits["exit_price"],
                           marker=marker, color=color, s=40, zorder=5,
                           label=f"{reason} Exit")

    ax.set_title("XAUUSD Breakout Strategy (Improved)", fontsize=14)
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: ATR ──
    ax = axes[1]
    ax.plot(df.index, df["ATR"], color="purple", linewidth=0.7)
    ax.set_ylabel("ATR")
    ax.set_title(f"Average True Range ({ATR_PERIOD})", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: ADX (if computed) ──
    if has_adx:
        ax = axes[2]
        ax.plot(df.index, df["ADX"], color="teal", linewidth=0.7)
        ax.axhline(ADX_THRESHOLD, color="red", linestyle="--", linewidth=0.5,
                    label=f"Threshold={ADX_THRESHOLD}")
        ax.fill_between(df.index, ADX_THRESHOLD, df["ADX"],
                         where=df["ADX"] >= ADX_THRESHOLD,
                         alpha=0.15, color="green", label="Trending")
        ax.set_ylabel("ADX")
        ax.set_title("ADX Trend Strength", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Panel 4: Equity curve ──
    ax = axes[-1]
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
