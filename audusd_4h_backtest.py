#!/usr/bin/env python3
"""
AUDUSD 4-Hour EMA Crossover + RSI + Bollinger Band Strategy Backtest
=====================================================================
Strategy:
  - Trend: 20 EMA / 50 EMA crossover determines direction
  - Entry: Price touches lower Bollinger Band in uptrend (long),
           or upper Bollinger Band in downtrend (short)
  - Filter: RSI must confirm (not overbought for longs, not oversold for shorts)
  - Exit: ATR-based stop-loss, trailing stop, and breakeven mechanisms

Uses yfinance to fetch AUDUSD=X data on a 4-hour timeframe.
Falls back to synthetic data if network is unavailable.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# ── Configuration ──────────────────────────────────────────────────────────
TICKER = "AUDUSD=X"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
INTERVAL = "4h"                    # 4-hour bars (yfinance: limited to ~730 days)
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02              # 2% risk per trade
LOT_SIZE = 100_000                 # Standard forex lot (for PnL calculation in pips->$)

# ── EMA Crossover Settings ─────────────────────────────────────────────────
EMA_FAST = 20                      # Fast EMA period (4H bars)
EMA_SLOW = 50                      # Slow EMA period (4H bars)

# ── RSI Filter (Improvement #4: tighter thresholds) ───────────────────────
RSI_PERIOD = 14
RSI_LONG_MAX = 55                  # Longs only when RSI < 55 (was 70)
RSI_SHORT_MIN = 45                 # Shorts only when RSI > 45 (was 30)
USE_RSI_FILTER = True

# ── Bollinger Bands ────────────────────────────────────────────────────────
BB_PERIOD = 20
BB_STD = 2.0
USE_BB_ENTRY = True                # Use BB touch as entry trigger
USE_BB_BOUNCE = True               # Improvement #1: require bounce confirmation

# ── ATR Stop/Target ───────────────────────────────────────────────────────
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0

# ── Trailing Stop ──────────────────────────────────────────────────────────
USE_TRAILING_STOP = True
TRAIL_ACTIVATION_ATR = 1.5         # Activate after 1.5x ATR move
TRAIL_DISTANCE_ATR = 2.0           # Improvement #3: trail 2x ATR (was 1.0)

# ── Breakeven Stop ─────────────────────────────────────────────────────────
USE_BREAKEVEN_STOP = True
BREAKEVEN_ATR = 2.0                # Improvement #2: relaxed (was 1.0)

# ── Cooldown ───────────────────────────────────────────────────────────────
COOLDOWN_BARS = 3                  # Wait 3 bars (12 hours) after SL hit

# ── ADX Regime Filter ─────────────────────────────────────────────────────
ADX_PERIOD = 14
ADX_THRESHOLD = 20
USE_ADX_FILTER = True

# ── 200 EMA Trend Filter ──────────────────────────────────────────────────
TREND_MA_PERIOD = 200
USE_TREND_FILTER = True

# ── Improvement #5: MACD Histogram Confirmation ─────────────────────────
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
USE_MACD_FILTER = True             # Require MACD histogram direction match

# ── Improvement #6: ATR Minimum Threshold ────────────────────────────────
USE_ATR_FILTER = True              # Skip trades in low-volatility squeeze
ATR_PERCENTILE_MIN = 25            # ATR must be above 25th percentile

# ── Improvement #7: Partial Position Close ───────────────────────────────
USE_PARTIAL_CLOSE = True           # Close 50% at 1.5x ATR, trail remainder
PARTIAL_CLOSE_ATR = 1.5            # ATR multiple to trigger partial close
PARTIAL_CLOSE_PCT = 0.5            # Close 50% of position


def generate_synthetic_audusd_data(start: str, end: str) -> pd.DataFrame:
    """Generate realistic synthetic AUDUSD 4H OHLCV data."""
    print("  Generating synthetic AUDUSD 4H price data (network unavailable) ...")
    np.random.seed(123)

    # Generate 4-hour bars (6 bars per day, weekdays only)
    daily_dates = pd.bdate_range(start=start, end=end)
    timestamps = []
    for d in daily_dates:
        for h in [0, 4, 8, 12, 16, 20]:
            timestamps.append(d + pd.Timedelta(hours=h))
    index = pd.DatetimeIndex(timestamps)
    n = len(index)

    # AUD/USD parameters: ~8% annual vol, slight mean-reversion around 0.68
    initial_price = 0.6850
    annual_drift = 0.01
    annual_vol = 0.08
    dt = 1 / (252 * 6)  # 6 bars per trading day

    # Mean-reverting GBM (Ornstein-Uhlenbeck-like)
    mean_level = 0.6800
    mean_reversion_speed = 0.5  # Annual

    prices = np.zeros(n)
    prices[0] = initial_price

    for i in range(1, n):
        drift = mean_reversion_speed * (mean_level - prices[i - 1]) * dt
        diffusion = annual_vol * np.sqrt(dt) * np.random.randn()
        prices[i] = prices[i - 1] * np.exp(drift + diffusion)

    close = prices

    # Generate OHLV from close
    bar_range = close * np.random.uniform(0.001, 0.004, size=n)
    high = close + bar_range * np.random.uniform(0.3, 0.7, size=n)
    low = close - bar_range * np.random.uniform(0.3, 0.7, size=n)
    open_ = low + (high - low) * np.random.uniform(0.2, 0.8, size=n)
    volume = np.random.randint(10_000, 100_000, size=n)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=index)
    return df


def fetch_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance, fall back to synthetic data.

    Note: yfinance limits 4h data to ~730 days. For longer backtests
    we fall back to synthetic data.
    """
    print(f"Fetching {ticker} ({interval}) data from {start} to {end} ...")
    if HAS_YFINANCE:
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval=interval, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) > 100:
                df.dropna(inplace=True)
                print(f"  Fetched {len(df)} bars from Yahoo Finance.\n")
                return df
            else:
                print(f"  Insufficient data from Yahoo Finance ({len(df)} bars).")
        except Exception as e:
            print(f"  Download failed: {e}")
    else:
        print("  yfinance not available.")

    # Fallback to synthetic data
    df = generate_synthetic_audusd_data(start, end)
    print(f"  Generated {len(df)} synthetic 4H bars.\n")
    return df


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators and entry signals."""
    df = df.copy()

    # ── EMA Crossover ──
    df["EMA_Fast"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_Slow"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["Trend"] = np.where(df["EMA_Fast"] > df["EMA_Slow"], 1, -1)

    # ── RSI ──
    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    # ── Bollinger Bands ──
    df["BB_Mid"] = df["Close"].rolling(BB_PERIOD).mean()
    bb_std = df["Close"].rolling(BB_PERIOD).std()
    df["BB_Upper"] = df["BB_Mid"] + BB_STD * bb_std
    df["BB_Lower"] = df["BB_Mid"] - BB_STD * bb_std

    # ── ATR ──
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()

    # ── 200 EMA Trend Filter ──
    df["EMA_200"] = df["Close"].ewm(span=TREND_MA_PERIOD, adjust=False).mean()

    # ── ADX ──
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

    # ── MACD (Improvement #5) ──
    ema_macd_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_macd_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD_Line"] = ema_macd_fast - ema_macd_slow
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

    # ── ATR percentile threshold (Improvement #6) ──
    df["ATR_Threshold"] = df["ATR"].rolling(window=200, min_periods=50).quantile(
        ATR_PERCENTILE_MIN / 100.0)

    # ── Entry Signals ──
    uptrend = df["Trend"] == 1
    downtrend = df["Trend"] == -1

    if USE_BB_ENTRY:
        if USE_BB_BOUNCE:
            # Improvement #1: prev bar touched BB, current bar closed back inside
            prev_touched_lower = df["Low"].shift(1) <= df["BB_Lower"].shift(1)
            bounced_up = df["Close"] > df["BB_Lower"]
            long_entry_trigger = prev_touched_lower & bounced_up

            prev_touched_upper = df["High"].shift(1) >= df["BB_Upper"].shift(1)
            bounced_down = df["Close"] < df["BB_Upper"]
            short_entry_trigger = prev_touched_upper & bounced_down
        else:
            long_entry_trigger = df["Low"] <= df["BB_Lower"]
            short_entry_trigger = df["High"] >= df["BB_Upper"]
    else:
        long_entry_trigger = (df["EMA_Fast"] > df["EMA_Slow"]) & (
            df["EMA_Fast"].shift(1) <= df["EMA_Slow"].shift(1))
        short_entry_trigger = (df["EMA_Fast"] < df["EMA_Slow"]) & (
            df["EMA_Fast"].shift(1) >= df["EMA_Slow"].shift(1))

    long_signal = uptrend & long_entry_trigger
    short_signal = downtrend & short_entry_trigger

    # RSI filter (Improvement #4: tighter thresholds)
    if USE_RSI_FILTER:
        long_signal = long_signal & (df["RSI"] < RSI_LONG_MAX)
        short_signal = short_signal & (df["RSI"] > RSI_SHORT_MIN)

    # 200 EMA trend filter
    if USE_TREND_FILTER:
        long_signal = long_signal & (df["Close"] > df["EMA_200"])
        short_signal = short_signal & (df["Close"] < df["EMA_200"])

    # ADX filter
    if USE_ADX_FILTER:
        trending = df["ADX"] > ADX_THRESHOLD
        long_signal = long_signal & trending
        short_signal = short_signal & trending

    # MACD histogram filter (Improvement #5)
    # Require histogram direction match OR histogram momentum (rising/falling)
    if USE_MACD_FILTER:
        macd_bullish = (df["MACD_Hist"] > 0) | (df["MACD_Hist"] > df["MACD_Hist"].shift(1))
        macd_bearish = (df["MACD_Hist"] < 0) | (df["MACD_Hist"] < df["MACD_Hist"].shift(1))
        long_signal = long_signal & macd_bullish
        short_signal = short_signal & macd_bearish

    # ATR volatility filter (Improvement #6)
    if USE_ATR_FILTER:
        sufficient_vol = df["ATR"] > df["ATR_Threshold"]
        long_signal = long_signal & sufficient_vol
        short_signal = short_signal & sufficient_vol

    df["Long_Signal"] = long_signal
    df["Short_Signal"] = short_signal

    df.dropna(inplace=True)
    return df


def run_backtest(df: pd.DataFrame) -> tuple:
    """Simulate the strategy with full risk management."""
    capital = INITIAL_CAPITAL
    position = 0       # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    equity_curve = []

    # Trailing / breakeven state
    trailing_active = False
    breakeven_hit = False
    partial_closed = False
    best_price = 0.0
    trail_atr = 0.0

    # Cooldown
    cooldown_remaining = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if position != 0:
            # ── LONG management ──
            if position == 1:
                if row["High"] > best_price:
                    best_price = row["High"]

                # Improvement #7: Partial close at 1.5x ATR
                if USE_PARTIAL_CLOSE and not partial_closed:
                    if best_price >= entry_price + PARTIAL_CLOSE_ATR * trail_atr:
                        close_lots = round(trades[-1]["lots"] * PARTIAL_CLOSE_PCT, 2)
                        partial_pips = PARTIAL_CLOSE_ATR * trail_atr * 10_000
                        partial_dollar = partial_pips * (LOT_SIZE / 10_000) * close_lots
                        capital += partial_dollar
                        trades[-1]["lots"] = round(trades[-1]["lots"] - close_lots, 2)
                        trades[-1]["partial_pnl"] = partial_dollar
                        partial_closed = True

                # Breakeven
                if USE_BREAKEVEN_STOP and not breakeven_hit:
                    if best_price >= entry_price + BREAKEVEN_ATR * trail_atr:
                        stop_loss = entry_price
                        breakeven_hit = True

                # Trailing stop
                if USE_TRAILING_STOP and trailing_active:
                    trail_sl = best_price - TRAIL_DISTANCE_ATR * trail_atr
                    stop_loss = max(stop_loss, trail_sl)
                elif USE_TRAILING_STOP and not trailing_active:
                    if best_price >= entry_price + TRAIL_ACTIVATION_ATR * trail_atr:
                        trailing_active = True
                        trail_sl = best_price - TRAIL_DISTANCE_ATR * trail_atr
                        stop_loss = max(stop_loss, trail_sl)

                # Stop-loss hit
                if row["Low"] <= stop_loss:
                    pnl_pips = (stop_loss - entry_price) * 10_000
                    pnl_dollar = pnl_pips * (LOT_SIZE / 10_000) * trades[-1]["lots"]
                    partial = trades[-1].get("partial_pnl", 0)
                    total_dollar = pnl_dollar + partial
                    total_pips = pnl_pips + (partial / (LOT_SIZE / 10_000) / max(trades[-1]["lots"], 0.01) if partial else 0)
                    reason = "TSL" if trailing_active else ("BE" if breakeven_hit else "SL")
                    trades[-1].update(exit_date=idx, exit_price=stop_loss,
                                      pnl_pips=total_pips, pnl_dollar=total_dollar,
                                      exit_reason=reason)
                    capital += pnl_dollar
                    position = 0
                    if reason == "SL":
                        cooldown_remaining = COOLDOWN_BARS
                # Fixed TP (when trailing disabled)
                elif not USE_TRAILING_STOP and row["High"] >= take_profit:
                    pnl_pips = (take_profit - entry_price) * 10_000
                    pnl_dollar = pnl_pips * (LOT_SIZE / 10_000) * trades[-1]["lots"]
                    partial = trades[-1].get("partial_pnl", 0)
                    total_dollar = pnl_dollar + partial
                    trades[-1].update(exit_date=idx, exit_price=take_profit,
                                      pnl_pips=pnl_pips, pnl_dollar=total_dollar,
                                      exit_reason="TP")
                    capital += pnl_dollar
                    position = 0

            # ── SHORT management ──
            elif position == -1:
                if row["Low"] < best_price:
                    best_price = row["Low"]

                # Improvement #7: Partial close at 1.5x ATR
                if USE_PARTIAL_CLOSE and not partial_closed:
                    if best_price <= entry_price - PARTIAL_CLOSE_ATR * trail_atr:
                        close_lots = round(trades[-1]["lots"] * PARTIAL_CLOSE_PCT, 2)
                        partial_pips = PARTIAL_CLOSE_ATR * trail_atr * 10_000
                        partial_dollar = partial_pips * (LOT_SIZE / 10_000) * close_lots
                        capital += partial_dollar
                        trades[-1]["lots"] = round(trades[-1]["lots"] - close_lots, 2)
                        trades[-1]["partial_pnl"] = partial_dollar
                        partial_closed = True

                # Breakeven
                if USE_BREAKEVEN_STOP and not breakeven_hit:
                    if best_price <= entry_price - BREAKEVEN_ATR * trail_atr:
                        stop_loss = entry_price
                        breakeven_hit = True

                # Trailing stop
                if USE_TRAILING_STOP and trailing_active:
                    trail_sl = best_price + TRAIL_DISTANCE_ATR * trail_atr
                    stop_loss = min(stop_loss, trail_sl)
                elif USE_TRAILING_STOP and not trailing_active:
                    if best_price <= entry_price - TRAIL_ACTIVATION_ATR * trail_atr:
                        trailing_active = True
                        trail_sl = best_price + TRAIL_DISTANCE_ATR * trail_atr
                        stop_loss = min(stop_loss, trail_sl)

                # Stop-loss hit
                if row["High"] >= stop_loss:
                    pnl_pips = (entry_price - stop_loss) * 10_000
                    pnl_dollar = pnl_pips * (LOT_SIZE / 10_000) * trades[-1]["lots"]
                    partial = trades[-1].get("partial_pnl", 0)
                    total_dollar = pnl_dollar + partial
                    total_pips = pnl_pips + (partial / (LOT_SIZE / 10_000) / max(trades[-1]["lots"], 0.01) if partial else 0)
                    reason = "TSL" if trailing_active else ("BE" if breakeven_hit else "SL")
                    trades[-1].update(exit_date=idx, exit_price=stop_loss,
                                      pnl_pips=total_pips, pnl_dollar=total_dollar,
                                      exit_reason=reason)
                    capital += pnl_dollar
                    position = 0
                    if reason == "SL":
                        cooldown_remaining = COOLDOWN_BARS
                # Fixed TP
                elif not USE_TRAILING_STOP and row["Low"] <= take_profit:
                    pnl_pips = (entry_price - take_profit) * 10_000
                    pnl_dollar = pnl_pips * (LOT_SIZE / 10_000) * trades[-1]["lots"]
                    partial = trades[-1].get("partial_pnl", 0)
                    total_dollar = pnl_dollar + partial
                    trades[-1].update(exit_date=idx, exit_price=take_profit,
                                      pnl_pips=pnl_pips, pnl_dollar=total_dollar,
                                      exit_reason="TP")
                    capital += pnl_dollar
                    position = 0

        # ── Open new position ──
        if position == 0 and cooldown_remaining == 0:
            atr = row["ATR"]
            if row["Long_Signal"] and atr > 0:
                position = 1
                entry_price = row["Close"]
                stop_loss = entry_price - ATR_SL_MULTIPLIER * atr
                take_profit = entry_price + ATR_TP_MULTIPLIER * atr
                risk_pips = ATR_SL_MULTIPLIER * atr * 10_000
                risk_dollar = capital * RISK_PER_TRADE
                pip_value = LOT_SIZE / 10_000  # $10 per pip per standard lot
                lots = risk_dollar / (risk_pips * pip_value)
                lots = round(max(0.01, lots), 2)
                best_price = entry_price
                trail_atr = atr
                trailing_active = False
                breakeven_hit = False
                partial_closed = False
                trades.append(dict(
                    entry_date=idx, direction="LONG",
                    entry_price=entry_price, sl=stop_loss, tp=take_profit,
                    lots=lots, exit_date=None, exit_price=None,
                    pnl_pips=None, pnl_dollar=None, exit_reason=None,
                ))
            elif row["Short_Signal"] and atr > 0:
                position = -1
                entry_price = row["Close"]
                stop_loss = entry_price + ATR_SL_MULTIPLIER * atr
                take_profit = entry_price - ATR_TP_MULTIPLIER * atr
                risk_pips = ATR_SL_MULTIPLIER * atr * 10_000
                risk_dollar = capital * RISK_PER_TRADE
                pip_value = LOT_SIZE / 10_000
                lots = risk_dollar / (risk_pips * pip_value)
                lots = round(max(0.01, lots), 2)
                best_price = entry_price
                trail_atr = atr
                trailing_active = False
                breakeven_hit = False
                partial_closed = False
                trades.append(dict(
                    entry_date=idx, direction="SHORT",
                    entry_price=entry_price, sl=stop_loss, tp=take_profit,
                    lots=lots, exit_date=None, exit_price=None,
                    pnl_pips=None, pnl_dollar=None, exit_reason=None,
                ))

        equity_curve.append({"date": idx, "equity": capital})

    # Close open position at end
    if position != 0 and trades:
        last = df.iloc[-1]
        if position == 1:
            pnl_pips = (last["Close"] - entry_price) * 10_000
        else:
            pnl_pips = (entry_price - last["Close"]) * 10_000
        pnl_dollar = pnl_pips * (LOT_SIZE / 10_000) * trades[-1]["lots"]
        partial = trades[-1].get("partial_pnl", 0)
        total_dollar = pnl_dollar + partial
        trades[-1].update(exit_date=df.index[-1], exit_price=last["Close"],
                          pnl_pips=pnl_pips, pnl_dollar=total_dollar,
                          exit_reason="EOD")
        capital += pnl_dollar

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return trade_df, equity_df


def print_stats(trade_df: pd.DataFrame, equity_df: pd.DataFrame) -> None:
    """Print backtest performance summary."""
    if trade_df.empty:
        print("No trades were generated.")
        return

    closed = trade_df.dropna(subset=["pnl_dollar"]).copy()
    total_pnl = closed["pnl_dollar"].sum()
    wins = closed[closed["pnl_dollar"] > 0]
    losses = closed[closed["pnl_dollar"] <= 0]
    win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
    avg_win = wins["pnl_dollar"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_dollar"].mean() if len(losses) > 0 else 0
    avg_win_pips = wins["pnl_pips"].mean() if len(wins) > 0 else 0
    avg_loss_pips = losses["pnl_pips"].mean() if len(losses) > 0 else 0
    profit_factor = (wins["pnl_dollar"].sum() / abs(losses["pnl_dollar"].sum())
                     if len(losses) > 0 and losses["pnl_dollar"].sum() != 0
                     else float("inf"))

    # Max drawdown
    peak = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - peak) / peak
    max_dd = dd.min() * 100

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    # Sharpe ratio (annualized for 4H bars: ~1512 bars/year)
    bars_per_year = 252 * 6
    equity_returns = equity_df["equity"].pct_change().dropna()
    sharpe = (equity_returns.mean() / equity_returns.std() * np.sqrt(bars_per_year)
              if equity_returns.std() > 0 else 0)

    # Calmar ratio
    calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else float("inf")

    # Avg holding period
    closed_with_dates = closed.dropna(subset=["entry_date", "exit_date"])
    if len(closed_with_dates) > 0:
        hold_durations = (pd.to_datetime(closed_with_dates["exit_date"])
                          - pd.to_datetime(closed_with_dates["entry_date"]))
        avg_hold = hold_durations.mean()
        avg_hold_str = str(avg_hold).split(".")[0]  # Remove sub-second
    else:
        avg_hold_str = "N/A"

    # Active filters
    filters = []
    if USE_TREND_FILTER:
        filters.append(f"EMA{TREND_MA_PERIOD}")
    if USE_RSI_FILTER:
        filters.append(f"RSI(<{RSI_LONG_MAX}/{RSI_SHORT_MIN}<)")
    if USE_BB_ENTRY:
        bb_tag = f"BB({BB_PERIOD},{BB_STD})"
        if USE_BB_BOUNCE:
            bb_tag += "+Bounce"
        filters.append(bb_tag)
    if USE_ADX_FILTER:
        filters.append(f"ADX>{ADX_THRESHOLD}")
    if USE_MACD_FILTER:
        filters.append("MACD")
    if USE_ATR_FILTER:
        filters.append(f"ATR>p{ATR_PERCENTILE_MIN}")
    if USE_TRAILING_STOP:
        filters.append(f"Trail({TRAIL_DISTANCE_ATR}x)")
    if USE_BREAKEVEN_STOP:
        filters.append(f"BE({BREAKEVEN_ATR}x)")
    if USE_PARTIAL_CLOSE:
        filters.append(f"Partial({int(PARTIAL_CLOSE_PCT*100)}%@{PARTIAL_CLOSE_ATR}x)")
    if COOLDOWN_BARS > 0:
        filters.append(f"CD{COOLDOWN_BARS}")
    filter_str = ", ".join(filters) if filters else "None"

    print("=" * 65)
    print("    AUDUSD 4H EMA + RSI + BB STRATEGY BACKTEST RESULTS")
    print("=" * 65)
    print(f"  Period             : {START_DATE} to {END_DATE}")
    print(f"  Timeframe          : 4 Hour")
    print(f"  EMA Crossover      : {EMA_FAST} / {EMA_SLOW}")
    print(f"  Active Filters     : {filter_str}")
    print(f"  Initial Capital    : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Equity       : ${final_equity:,.2f}")
    print(f"  Total Return       : {total_return:+.2f}%")
    print(f"  Max Drawdown       : {max_dd:.2f}%")
    print(f"  Sharpe Ratio       : {sharpe:.2f}")
    print(f"  Calmar Ratio       : {calmar:.2f}")
    print("-" * 65)
    print(f"  Total Trades       : {len(closed)}")
    print(f"  Winning Trades     : {len(wins)}  ({win_rate:.1f}%)")
    print(f"  Losing Trades      : {len(losses)}")
    print(f"  Avg Win ($)        : ${avg_win:,.2f}  ({avg_win_pips:.1f} pips)")
    print(f"  Avg Loss ($)       : ${avg_loss:,.2f}  ({avg_loss_pips:.1f} pips)")
    print(f"  Profit Factor      : {profit_factor:.2f}")
    print(f"  Total P&L ($)      : ${total_pnl:,.2f}")
    print(f"  Avg Hold Time      : {avg_hold_str}")
    print("-" * 65)

    # Per-direction breakdown
    for direction in ["LONG", "SHORT"]:
        dir_trades = closed[closed["direction"] == direction]
        if len(dir_trades) > 0:
            dir_wins = dir_trades[dir_trades["pnl_dollar"] > 0]
            dir_wr = len(dir_wins) / len(dir_trades) * 100
            dir_pnl = dir_trades["pnl_dollar"].sum()
            dir_pips = dir_trades["pnl_pips"].sum()
            print(f"  {direction:5s} Trades      : {len(dir_trades)}  "
                  f"({dir_wr:.1f}% win) | P&L: ${dir_pnl:,.2f}  "
                  f"({dir_pips:.1f} pips)")

    # Exit reason breakdown
    print("-" * 65)
    for reason in sorted(closed["exit_reason"].unique()):
        r_trades = closed[closed["exit_reason"] == reason]
        r_pnl = r_trades["pnl_dollar"].mean()
        r_pips = r_trades["pnl_pips"].mean()
        print(f"  {reason:3s} exits         : {len(r_trades):3d}  |  "
              f"Avg: ${r_pnl:,.2f}  ({r_pips:.1f} pips)")
    print("=" * 65)


def plot_results(df: pd.DataFrame, equity_df: pd.DataFrame,
                 trade_df: pd.DataFrame) -> None:
    """Generate 6-panel chart: Price+BB+EMA, RSI, MACD, ADX, ATR, Equity."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 22), sharex=False,
                             gridspec_kw={"height_ratios": [4, 1.2, 1.2, 1, 1, 2]})

    # ── Panel 1: Price + EMAs + Bollinger Bands ──
    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Close", linewidth=0.7, color="black")
    ax.plot(df.index, df["EMA_Fast"], label=f"EMA {EMA_FAST}",
            linewidth=0.8, color="blue", alpha=0.8)
    ax.plot(df.index, df["EMA_Slow"], label=f"EMA {EMA_SLOW}",
            linewidth=0.8, color="red", alpha=0.8)
    ax.plot(df.index, df["EMA_200"], label=f"EMA {TREND_MA_PERIOD}",
            linewidth=1.0, color="orange", alpha=0.7)

    # Bollinger Bands
    ax.plot(df.index, df["BB_Upper"], linewidth=0.5, color="gray", alpha=0.6)
    ax.plot(df.index, df["BB_Lower"], linewidth=0.5, color="gray", alpha=0.6)
    ax.fill_between(df.index, df["BB_Lower"], df["BB_Upper"],
                     alpha=0.05, color="blue")

    # Trade markers
    if not trade_df.empty:
        longs = trade_df[trade_df["direction"] == "LONG"]
        shorts = trade_df[trade_df["direction"] == "SHORT"]
        if not longs.empty:
            ax.scatter(longs["entry_date"], longs["entry_price"],
                       marker="^", color="green", s=50, zorder=5, label="Long Entry")
        if not shorts.empty:
            ax.scatter(shorts["entry_date"], shorts["entry_price"],
                       marker="v", color="red", s=50, zorder=5, label="Short Entry")

        closed = trade_df.dropna(subset=["exit_date"])
        for reason, color, marker in [("TP", "blue", "D"), ("TSL", "orange", "s"),
                                       ("BE", "cyan", "o"), ("SL", "red", "x")]:
            exits = closed[closed["exit_reason"] == reason]
            if not exits.empty:
                ax.scatter(exits["exit_date"], exits["exit_price"],
                           marker=marker, color=color, s=35, zorder=5,
                           label=f"{reason} Exit")

    ax.set_title("AUDUSD 4H — EMA Crossover + RSI + Bollinger Band Strategy",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (AUD/USD)")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: RSI ──
    ax = axes[1]
    ax.plot(df.index, df["RSI"], color="purple", linewidth=0.6)
    ax.axhline(RSI_LONG_MAX, color="red", linestyle="--", linewidth=0.5,
                label=f"Long max={RSI_LONG_MAX}")
    ax.axhline(RSI_SHORT_MIN, color="green", linestyle="--", linewidth=0.5,
                label=f"Short min={RSI_SHORT_MIN}")
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.4)
    ax.fill_between(df.index, RSI_LONG_MAX, 100,
                     alpha=0.05, color="red")
    ax.fill_between(df.index, 0, RSI_SHORT_MIN,
                     alpha=0.05, color="green")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.set_title(f"RSI ({RSI_PERIOD})", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: MACD ──
    ax = axes[2]
    ax.plot(df.index, df["MACD_Line"], color="blue", linewidth=0.6, label="MACD")
    ax.plot(df.index, df["MACD_Signal"], color="orange", linewidth=0.6, label="Signal")
    colors = ["green" if v >= 0 else "red" for v in df["MACD_Hist"]]
    ax.bar(df.index, df["MACD_Hist"], color=colors, alpha=0.4, width=0.15)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.4)
    ax.set_ylabel("MACD")
    ax.set_title(f"MACD ({MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL})", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: ADX ──
    ax = axes[3]
    ax.plot(df.index, df["ADX"], color="teal", linewidth=0.6)
    ax.axhline(ADX_THRESHOLD, color="red", linestyle="--", linewidth=0.5,
                label=f"Threshold={ADX_THRESHOLD}")
    ax.fill_between(df.index, ADX_THRESHOLD, df["ADX"],
                     where=df["ADX"] >= ADX_THRESHOLD,
                     alpha=0.15, color="green", label="Trending")
    ax.set_ylabel("ADX")
    ax.set_title("ADX Trend Strength", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: ATR ──
    ax = axes[4]
    ax.plot(df.index, df["ATR"] * 10_000, color="brown", linewidth=0.6)
    ax.set_ylabel("ATR (pips)")
    ax.set_title(f"ATR ({ATR_PERIOD}) in Pips", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Equity Curve ──
    ax = axes[4]
    ax.plot(equity_df.index, equity_df["equity"], color="blue", linewidth=1)
    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Equity ($)")
    ax.set_title("Equity Curve", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Drawdown fill
    peak = equity_df["equity"].cummax()
    ax.fill_between(equity_df.index, equity_df["equity"], peak,
                     alpha=0.15, color="red", label="Drawdown")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = "audusd_4h_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = fetch_data(TICKER, START_DATE, END_DATE, INTERVAL)
    signals = compute_signals(data)

    print(f"Total bars after indicator warmup: {len(signals)}")
    print(f"Long signals:  {signals['Long_Signal'].sum()}")
    print(f"Short signals: {signals['Short_Signal'].sum()}\n")

    trade_log, equity = run_backtest(signals)
    print_stats(trade_log, equity)
    plot_results(signals, equity, trade_log)

    # Save trade log
    if not trade_log.empty:
        trade_log.to_csv("audusd_4h_trades.csv", index=False)
        print("Trade log saved to audusd_4h_trades.csv")
