#!/usr/bin/env python3
"""
BTC-USD 15-Minute VWAP + RSI Mean-Reversion Strategy Backtest
==============================================================
Strategy:
  - Direction: VWAP determines fair value — buy below, sell above
  - Entry: Price touches Bollinger Band extreme with RSI confirmation
  - Filter: ADX for volatility regime, EMA(200) for macro trend
  - Exit: ATR-based stop-loss, trailing stop, and breakeven mechanisms

Uses CCXT to fetch BTC/USDT data from Binance on a 15-minute timeframe,
enabling multi-year historical data (no 60-day limit like yfinance).
Falls back to synthetic data if network is unavailable.
"""

import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


# ── Configuration ──────────────────────────────────────────────────────────
SYMBOL = "BTC/USDT"                 # CCXT symbol (Binance spot)
EXCHANGE_ID = "binance"             # CCXT exchange identifier
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
TIMEFRAME = "15m"                   # 15-minute bars
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.01              # 1% risk per trade (lower for crypto volatility)

# ── VWAP Settings ─────────────────────────────────────────────────────────
VWAP_SESSION_HOURS = 24             # VWAP resets every 24h (midnight UTC)

# ── RSI Settings ──────────────────────────────────────────────────────────
RSI_PERIOD = 7                      # Fast RSI for intraday scalping
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
USE_RSI_FILTER = True

# ── Bollinger Band Settings ──────────────────────────────────────────────
BB_PERIOD = 20
BB_STD = 2.0
USE_BB_ENTRY = True                 # Require BB touch for entry confirmation

# ── ATR Stop/Target ──────────────────────────────────────────────────────
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5

# ── Trailing Stop ────────────────────────────────────────────────────────
USE_TRAILING_STOP = True
TRAIL_ACTIVATION_ATR = 1.5         # Activate after 1.5x ATR favorable move
TRAIL_DISTANCE_ATR = 1.0           # Trail 1x ATR behind best price

# ── Breakeven Stop ───────────────────────────────────────────────────────
USE_BREAKEVEN_STOP = True
BREAKEVEN_ATR = 1.0                # Move SL to entry after 1x ATR move

# ── Cooldown ─────────────────────────────────────────────────────────────
COOLDOWN_BARS = 4                  # 4 bars = 1 hour on 15m timeframe

# ── ADX Regime Filter ───────────────────────────────────────────────────
ADX_PERIOD = 14
ADX_THRESHOLD = 20
USE_ADX_FILTER = True

# ── 200 EMA Trend Filter ────────────────────────────────────────────────
TREND_EMA_PERIOD = 200             # 200 x 15m = 50 hours ≈ 2 days
USE_TREND_FILTER = True


def generate_synthetic_btc_data(start: str, end: str) -> pd.DataFrame:
    """Generate realistic synthetic BTC-USD 15m OHLCV data using geometric Brownian motion.

    BTC trades 24/7, so we generate bars for every calendar day (including weekends)
    with 96 fifteen-minute bars per day.
    """
    print("  Generating synthetic BTC-USD 15m data (yfinance 15m limit ~60 days) ...")
    np.random.seed(2024)

    # Generate 15-minute timestamps for all calendar days (BTC = 24/7 market)
    daily_dates = pd.date_range(start=start, end=end, freq="D")
    timestamps = []
    for d in daily_dates:
        for h in range(24):
            for m in [0, 15, 30, 45]:
                timestamps.append(d + pd.Timedelta(hours=h, minutes=m))
    index = pd.DatetimeIndex(timestamps)
    n = len(index)

    # BTC GBM parameters (calibrated to historical BTC behavior)
    initial_price = 7200.0       # BTC price ~ January 2020
    annual_drift = 0.30          # Strong upward bias (historical BTC trend)
    annual_vol = 0.70            # ~70% annual volatility (realistic for BTC)
    dt = 1 / (365 * 96)         # 96 bars per day, 365 days per year

    # Generate log-normal returns via GBM
    daily_returns = np.exp(
        (annual_drift - 0.5 * annual_vol**2) * dt
        + annual_vol * np.sqrt(dt) * np.random.randn(n)
    )
    close = initial_price * np.cumprod(daily_returns)

    # Generate OHLV from close prices
    bar_range = close * np.random.uniform(0.001, 0.006, size=n)
    high = close + bar_range * np.random.uniform(0.3, 0.7, size=n)
    low = close - bar_range * np.random.uniform(0.3, 0.7, size=n)
    open_ = low + (high - low) * np.random.uniform(0.2, 0.8, size=n)
    volume = np.random.randint(100, 5000, size=n).astype(float)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=index)
    return df


def fetch_ohlcv_ccxt(symbol: str, exchange_id: str, timeframe: str,
                     start: str, end: str) -> pd.DataFrame:
    """Fetch historical OHLCV data from a CCXT exchange with pagination.

    CCXT returns at most 1000-1500 candles per request, so we loop
    from `start` to `end` in chunks, respecting the exchange rate limit.
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "timeout": 30000})
    exchange.load_markets()

    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    limit = 1000  # candles per request (Binance max = 1000 for 15m)

    all_ohlcv = []
    print(f"  Fetching from {exchange_id.upper()} ({symbol} {timeframe}) ...")
    max_retries = 4
    while since < end_ms:
        ohlcv = None
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                break
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                wait = 2 ** (attempt + 1)
                print(f"\n  Network error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"  Max retries reached. Aborting fetch.")
                    return pd.DataFrame()
            except Exception as e:
                print(f"\n  CCXT fetch error: {e}")
                return pd.DataFrame()
        if ohlcv is None:
            break

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        # Advance past the last fetched candle
        since = last_ts + 1

        # Progress indicator
        fetched_date = pd.Timestamp(last_ts, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        print(f"\r  ... fetched up to {fetched_date}  ({len(all_ohlcv):,} bars)", end="", flush=True)

        # Respect rate limit
        time.sleep(exchange.rateLimit / 1000)

    print()  # newline after progress

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_localize(None)  # Remove tz for consistency

    # Filter to requested date range
    df = df[(df.index >= start) & (df.index <= end)]
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    return df


def fetch_data(symbol: str, exchange_id: str, timeframe: str,
               start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV via CCXT (multi-year capable), fall back to synthetic data."""
    print(f"Fetching {symbol} ({timeframe}) data from {start} to {end} ...")
    if HAS_CCXT:
        try:
            df = fetch_ohlcv_ccxt(symbol, exchange_id, timeframe, start, end)
            if not df.empty and len(df) > 100:
                df.dropna(inplace=True)
                print(f"  Fetched {len(df):,} bars from {exchange_id.upper()}.\n")
                return df
            else:
                print(f"  Insufficient data from {exchange_id.upper()} ({len(df)} bars).")
        except Exception as e:
            print(f"  CCXT download failed: {e}")
    else:
        print("  ccxt not available.")

    # Fallback to synthetic data
    df = generate_synthetic_btc_data(start, end)
    print(f"  Generated {len(df):,} synthetic 15m bars.\n")
    return df


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing method."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP with daily session resets (midnight UTC).

    VWAP = cumulative(TypicalPrice * Volume) / cumulative(Volume)
    within each 24-hour session.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol = tp * df["Volume"]

    # Group by calendar date (session = midnight-to-midnight UTC)
    session = df.index.normalize()
    cum_tp_vol = tp_vol.groupby(session).cumsum()
    cum_vol = df["Volume"].groupby(session).cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators and generate entry signals."""
    df = df.copy()

    # ── VWAP ──
    df["VWAP"] = compute_vwap(df)

    # ── RSI (fast, 7-period) ──
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
    df["EMA_200"] = df["Close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()

    # ── ADX (Wilder's method) ──
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

    # ── Build Entry Signals ──
    # VWAP direction
    below_vwap = df["Close"] < df["VWAP"]
    above_vwap = df["Close"] > df["VWAP"]

    # Bollinger Band touch
    if USE_BB_ENTRY:
        bb_lower_touch = df["Low"] <= df["BB_Lower"]
        bb_upper_touch = df["High"] >= df["BB_Upper"]
    else:
        bb_lower_touch = pd.Series(True, index=df.index)
        bb_upper_touch = pd.Series(True, index=df.index)

    # RSI filter
    if USE_RSI_FILTER:
        rsi_oversold = df["RSI"] < RSI_OVERSOLD
        rsi_overbought = df["RSI"] > RSI_OVERBOUGHT
    else:
        rsi_oversold = pd.Series(True, index=df.index)
        rsi_overbought = pd.Series(True, index=df.index)

    # Trend filter (EMA 200)
    if USE_TREND_FILTER:
        trend_up = df["Close"] > df["EMA_200"]
        trend_down = df["Close"] < df["EMA_200"]
    else:
        trend_up = pd.Series(True, index=df.index)
        trend_down = pd.Series(True, index=df.index)

    # ADX filter
    if USE_ADX_FILTER:
        trending = df["ADX"] > ADX_THRESHOLD
    else:
        trending = pd.Series(True, index=df.index)

    # Combine all conditions
    df["Long_Signal"] = below_vwap & bb_lower_touch & rsi_oversold & trend_up & trending
    df["Short_Signal"] = above_vwap & bb_upper_touch & rsi_overbought & trend_down & trending

    df.dropna(inplace=True)
    return df


def run_backtest(df: pd.DataFrame) -> tuple:
    """Simulate the VWAP mean-reversion strategy with full risk management."""
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
    best_price = 0.0
    trail_atr = 0.0

    # Cooldown
    cooldown_remaining = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if position != 0:
            # ── LONG position management ──
            if position == 1:
                if row["High"] > best_price:
                    best_price = row["High"]

                # Breakeven: move SL to entry after favorable move
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

                # Check stop-loss
                if row["Low"] <= stop_loss:
                    pnl_dollar = (stop_loss - entry_price) * trades[-1]["qty_btc"]
                    reason = "TSL" if trailing_active else ("BE" if breakeven_hit else "SL")
                    trades[-1].update(exit_date=idx, exit_price=stop_loss,
                                      pnl_dollar=pnl_dollar, exit_reason=reason)
                    capital += pnl_dollar
                    position = 0
                    if reason == "SL":
                        cooldown_remaining = COOLDOWN_BARS
                # Check fixed TP (only when trailing stop is disabled)
                elif not USE_TRAILING_STOP and row["High"] >= take_profit:
                    pnl_dollar = (take_profit - entry_price) * trades[-1]["qty_btc"]
                    trades[-1].update(exit_date=idx, exit_price=take_profit,
                                      pnl_dollar=pnl_dollar, exit_reason="TP")
                    capital += pnl_dollar
                    position = 0

            # ── SHORT position management ──
            elif position == -1:
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
                    pnl_dollar = (entry_price - stop_loss) * trades[-1]["qty_btc"]
                    reason = "TSL" if trailing_active else ("BE" if breakeven_hit else "SL")
                    trades[-1].update(exit_date=idx, exit_price=stop_loss,
                                      pnl_dollar=pnl_dollar, exit_reason=reason)
                    capital += pnl_dollar
                    position = 0
                    if reason == "SL":
                        cooldown_remaining = COOLDOWN_BARS
                # Fixed TP
                elif not USE_TRAILING_STOP and row["Low"] <= take_profit:
                    pnl_dollar = (entry_price - take_profit) * trades[-1]["qty_btc"]
                    trades[-1].update(exit_date=idx, exit_price=take_profit,
                                      pnl_dollar=pnl_dollar, exit_reason="TP")
                    capital += pnl_dollar
                    position = 0

        # ── Open new position if flat and cooldown expired ──
        if position == 0 and cooldown_remaining == 0:
            atr = row["ATR"]
            if row["Long_Signal"] and atr > 0:
                position = 1
                entry_price = row["Close"]
                stop_loss = entry_price - ATR_SL_MULTIPLIER * atr
                take_profit = entry_price + ATR_TP_MULTIPLIER * atr
                risk_per_btc = ATR_SL_MULTIPLIER * atr  # USD per BTC risked
                risk_dollar = capital * RISK_PER_TRADE
                qty_btc = risk_dollar / risk_per_btc
                qty_btc = round(max(0.001, qty_btc), 6)
                best_price = entry_price
                trail_atr = atr
                trailing_active = False
                breakeven_hit = False
                trades.append(dict(
                    entry_date=idx, direction="LONG",
                    entry_price=entry_price, sl=stop_loss, tp=take_profit,
                    qty_btc=qty_btc, exit_date=None, exit_price=None,
                    pnl_dollar=None, exit_reason=None,
                ))
            elif row["Short_Signal"] and atr > 0:
                position = -1
                entry_price = row["Close"]
                stop_loss = entry_price + ATR_SL_MULTIPLIER * atr
                take_profit = entry_price - ATR_TP_MULTIPLIER * atr
                risk_per_btc = ATR_SL_MULTIPLIER * atr
                risk_dollar = capital * RISK_PER_TRADE
                qty_btc = risk_dollar / risk_per_btc
                qty_btc = round(max(0.001, qty_btc), 6)
                best_price = entry_price
                trail_atr = atr
                trailing_active = False
                breakeven_hit = False
                trades.append(dict(
                    entry_date=idx, direction="SHORT",
                    entry_price=entry_price, sl=stop_loss, tp=take_profit,
                    qty_btc=qty_btc, exit_date=None, exit_price=None,
                    pnl_dollar=None, exit_reason=None,
                ))

        equity_curve.append({"date": idx, "equity": capital})

    # Close any open position at end of data
    if position != 0 and trades:
        last = df.iloc[-1]
        if position == 1:
            pnl_dollar = (last["Close"] - entry_price) * trades[-1]["qty_btc"]
        else:
            pnl_dollar = (entry_price - last["Close"]) * trades[-1]["qty_btc"]
        trades[-1].update(exit_date=df.index[-1], exit_price=last["Close"],
                          pnl_dollar=pnl_dollar, exit_reason="EOD")
        capital += pnl_dollar

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return trade_df, equity_df


def print_stats(trade_df: pd.DataFrame, equity_df: pd.DataFrame) -> None:
    """Print comprehensive backtest performance summary."""
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
    profit_factor = (wins["pnl_dollar"].sum() / abs(losses["pnl_dollar"].sum())
                     if len(losses) > 0 and losses["pnl_dollar"].sum() != 0
                     else float("inf"))

    # Max drawdown
    peak = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - peak) / peak
    max_dd = dd.min() * 100

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    # Sharpe ratio (annualized for 15m bars: 96 bars/day * 365 days/year)
    bars_per_year = 365 * 96
    equity_returns = equity_df["equity"].pct_change().dropna()
    sharpe = (equity_returns.mean() / equity_returns.std() * np.sqrt(bars_per_year)
              if equity_returns.std() > 0 else 0)

    # Calmar ratio
    calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else float("inf")

    # Average holding period
    closed_with_dates = closed.dropna(subset=["entry_date", "exit_date"])
    if len(closed_with_dates) > 0:
        hold_durations = (pd.to_datetime(closed_with_dates["exit_date"])
                          - pd.to_datetime(closed_with_dates["entry_date"]))
        avg_hold = hold_durations.mean()
        avg_hold_str = str(avg_hold).split(".")[0]  # Remove sub-second
    else:
        avg_hold_str = "N/A"

    # Active filters summary
    filters = []
    filters.append("VWAP")
    if USE_RSI_FILTER:
        filters.append(f"RSI{RSI_PERIOD}({RSI_OVERSOLD}/{RSI_OVERBOUGHT})")
    if USE_BB_ENTRY:
        filters.append(f"BB({BB_PERIOD},{BB_STD})")
    if USE_TREND_FILTER:
        filters.append(f"EMA{TREND_EMA_PERIOD}")
    if USE_ADX_FILTER:
        filters.append(f"ADX>{ADX_THRESHOLD}")
    if USE_TRAILING_STOP:
        filters.append("Trail")
    if USE_BREAKEVEN_STOP:
        filters.append("BE")
    if COOLDOWN_BARS > 0:
        filters.append(f"CD{COOLDOWN_BARS}")
    filter_str = ", ".join(filters)

    print("=" * 70)
    print("   BTC-USD 15M VWAP + RSI + BB MEAN-REVERSION BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Period             : {START_DATE} to {END_DATE}")
    print(f"  Timeframe          : 15 Minute")
    print(f"  Strategy           : VWAP + RSI({RSI_PERIOD}) + BB({BB_PERIOD},{BB_STD})")
    print(f"  Active Filters     : {filter_str}")
    print(f"  Risk per Trade     : {RISK_PER_TRADE*100:.1f}%")
    print(f"  Initial Capital    : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Equity       : ${final_equity:,.2f}")
    print(f"  Total Return       : {total_return:+.2f}%")
    print(f"  Max Drawdown       : {max_dd:.2f}%")
    print(f"  Sharpe Ratio       : {sharpe:.2f}")
    print(f"  Calmar Ratio       : {calmar:.2f}")
    print("-" * 70)
    print(f"  Total Trades       : {len(closed)}")
    print(f"  Winning Trades     : {len(wins)}  ({win_rate:.1f}%)")
    print(f"  Losing Trades      : {len(losses)}")
    print(f"  Avg Win ($)        : ${avg_win:,.2f}")
    print(f"  Avg Loss ($)       : ${avg_loss:,.2f}")
    print(f"  Profit Factor      : {profit_factor:.2f}")
    print(f"  Total P&L ($)      : ${total_pnl:,.2f}")
    print(f"  Avg Hold Time      : {avg_hold_str}")
    print("-" * 70)

    # Per-direction breakdown
    for direction in ["LONG", "SHORT"]:
        dir_trades = closed[closed["direction"] == direction]
        if len(dir_trades) > 0:
            dir_wins = dir_trades[dir_trades["pnl_dollar"] > 0]
            dir_wr = len(dir_wins) / len(dir_trades) * 100
            dir_pnl = dir_trades["pnl_dollar"].sum()
            print(f"  {direction:5s} Trades      : {len(dir_trades)}  "
                  f"({dir_wr:.1f}% win) | P&L: ${dir_pnl:,.2f}")

    # Exit reason breakdown
    print("-" * 70)
    for reason in sorted(closed["exit_reason"].unique()):
        r_trades = closed[closed["exit_reason"] == reason]
        r_pnl = r_trades["pnl_dollar"].mean()
        print(f"  {reason:3s} exits         : {len(r_trades):3d}  |  "
              f"Avg $PnL: ${r_pnl:,.2f}")
    print("=" * 70)


def plot_results(df: pd.DataFrame, equity_df: pd.DataFrame,
                 trade_df: pd.DataFrame) -> None:
    """Generate 5-panel chart: Price+VWAP+BB, RSI, ADX, ATR, Equity."""
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=False,
                             gridspec_kw={"height_ratios": [4, 1.2, 1, 1, 2]})

    # ── Panel 1: Price + VWAP + Bollinger Bands + EMA(200) ──
    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Close", linewidth=0.5, color="black")
    ax.plot(df.index, df["VWAP"], label="VWAP", linewidth=1.0,
            color="blue", linestyle="--", alpha=0.8)
    ax.plot(df.index, df["EMA_200"], label=f"EMA {TREND_EMA_PERIOD}",
            linewidth=0.8, color="orange", alpha=0.7)

    # Bollinger Bands
    ax.plot(df.index, df["BB_Upper"], linewidth=0.4, color="gray", alpha=0.6)
    ax.plot(df.index, df["BB_Lower"], linewidth=0.4, color="gray", alpha=0.6)
    ax.fill_between(df.index, df["BB_Lower"], df["BB_Upper"],
                     alpha=0.05, color="blue")

    # Trade markers
    if not trade_df.empty:
        longs = trade_df[trade_df["direction"] == "LONG"]
        shorts = trade_df[trade_df["direction"] == "SHORT"]
        if not longs.empty:
            ax.scatter(longs["entry_date"], longs["entry_price"],
                       marker="^", color="green", s=40, zorder=5, label="Long Entry")
        if not shorts.empty:
            ax.scatter(shorts["entry_date"], shorts["entry_price"],
                       marker="v", color="red", s=40, zorder=5, label="Short Entry")

        closed = trade_df.dropna(subset=["exit_date"])
        for reason, color, marker in [("TP", "blue", "D"), ("TSL", "orange", "s"),
                                       ("BE", "cyan", "o"), ("SL", "red", "x")]:
            exits = closed[closed["exit_reason"] == reason]
            if not exits.empty:
                ax.scatter(exits["exit_date"], exits["exit_price"],
                           marker=marker, color=color, s=30, zorder=5,
                           label=f"{reason} Exit")

    ax.set_title("BTC-USD 15M — VWAP + RSI Mean-Reversion + Bollinger Band Strategy",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: RSI ──
    ax = axes[1]
    ax.plot(df.index, df["RSI"], color="purple", linewidth=0.5)
    ax.axhline(RSI_OVERBOUGHT, color="red", linestyle="--", linewidth=0.5)
    ax.axhline(RSI_OVERSOLD, color="green", linestyle="--", linewidth=0.5)
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.4)
    ax.fill_between(df.index, RSI_OVERBOUGHT, df["RSI"],
                     where=df["RSI"] >= RSI_OVERBOUGHT,
                     alpha=0.2, color="red")
    ax.fill_between(df.index, RSI_OVERSOLD, df["RSI"],
                     where=df["RSI"] <= RSI_OVERSOLD,
                     alpha=0.2, color="green")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.set_title(f"RSI ({RSI_PERIOD})", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: ADX ──
    ax = axes[2]
    ax.plot(df.index, df["ADX"], color="teal", linewidth=0.5)
    ax.axhline(ADX_THRESHOLD, color="red", linestyle="--", linewidth=0.5,
                label=f"Threshold={ADX_THRESHOLD}")
    ax.fill_between(df.index, ADX_THRESHOLD, df["ADX"],
                     where=df["ADX"] >= ADX_THRESHOLD,
                     alpha=0.15, color="green", label="Trending")
    ax.set_ylabel("ADX")
    ax.set_title("ADX Trend Strength", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: ATR ──
    ax = axes[3]
    ax.plot(df.index, df["ATR"], color="brown", linewidth=0.5)
    ax.set_ylabel("ATR (USD)")
    ax.set_title(f"ATR ({ATR_PERIOD}) in USD", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Equity Curve ──
    ax = axes[4]
    ax.plot(equity_df.index, equity_df["equity"], color="blue", linewidth=0.8)
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
    out_path = "btc_15m_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = fetch_data(SYMBOL, EXCHANGE_ID, TIMEFRAME, START_DATE, END_DATE)
    signals = compute_signals(data)

    print(f"Total bars after indicator warmup: {len(signals)}")
    print(f"Long signals:  {signals['Long_Signal'].sum()}")
    print(f"Short signals: {signals['Short_Signal'].sum()}\n")

    trade_log, equity = run_backtest(signals)
    print_stats(trade_log, equity)
    plot_results(signals, equity, trade_log)

    # Save trade log
    if not trade_log.empty:
        trade_log.to_csv("btc_15m_trades.csv", index=False)
        print("Trade log saved to btc_15m_trades.csv")
